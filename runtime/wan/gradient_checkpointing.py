"""Bounded exact recomputation for the terminal-gradient Wan path."""
from __future__ import annotations

from contextlib import nullcontext


class ReplayLedger:
    """Track forward/recompute calls separately and persist every transition."""

    KINDS = ("transformer_block", "vae_chunk")

    def __init__(self, limits: dict, callback=None):
        self.limits = dict(limits)
        for kind in self.KINDS:
            value = self.limits.get(kind)
            if type(value) is not int or value < 0:
                raise ValueError(f"nonnegative replay limit required for {kind}")
        self.counts = {
            kind: {phase: {"attempted": 0, "completed": 0}
                   for phase in ("forward", "recompute")}
            for kind in self.KINDS
        }
        self.boundaries = {}
        self._seen_storage = {}
        self.callback = callback

    def _emit(self) -> None:
        if self.callback is not None:
            self.callback(self.summary())

    def start(self, kind: str, phase: str) -> None:
        row = self.counts[kind][phase]
        if row["attempted"] >= self.limits[kind]:
            raise RuntimeError(f"REPLAY_LIMIT_{kind}_{phase}")
        row["attempted"] += 1
        self._emit()

    def finish(self, kind: str, phase: str) -> None:
        row = self.counts[kind][phase]
        row["completed"] += 1
        if row["completed"] > row["attempted"]:
            raise RuntimeError("replay completion exceeds attempts")
        self._emit()

    def new_decode(self) -> int:
        index = len(self.boundaries)
        self.boundaries[index] = {
            phase: {"chunks": [], "unique_storage_bytes": 0}
            for phase in ("forward", "recompute")
        }
        self._emit()
        return index

    def record_boundary(self, decode_id: int, phase: str, chunk_index: int, cache) -> None:
        unique, logical = {}, 0
        for value in cache:
            if not hasattr(value, "untyped_storage"):
                continue
            storage = value.untyped_storage()
            key = (value.device.type, value.device.index, storage.data_ptr(), storage.nbytes())
            unique[key] = storage.nbytes()
            logical += value.numel() * value.element_size()
        record = self.boundaries[decode_id][phase]
        seen = self._seen_storage.setdefault((decode_id, phase), set())
        record["unique_storage_bytes"] += sum(
            size for key, size in unique.items() if key not in seen
        )
        seen.update(unique)
        record["chunks"].append(dict(
            index=chunk_index,
            tensor_slots=sum(hasattr(value, "untyped_storage") for value in cache),
            logical_bytes=logical,
            unique_storage_bytes=sum(unique.values()),
        ))
        self._emit()

    def summary(self) -> dict:
        return dict(limits=self.limits, counts=self.counts, cache_boundaries=self.boundaries)


class BoundaryStorage:
    """Pack alias-preserving checkpoint boundary storage on CPU."""

    def __init__(self):
        self.copies = {}

    def pack(self, tensor):
        import torch

        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr(), storage.nbytes())
        if key not in self.copies:
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            self.copies[key] = raw.to(device="cpu", copy=True)
        return (self.copies[key], tensor.device, tensor.dtype, tensor.storage_offset(),
                tuple(tensor.shape), tuple(tensor.stride()))

    @staticmethod
    def unpack(packed):
        import torch
        import weakref

        raw, device, dtype, offset, shape, stride = packed
        previous = getattr(raw, "_restored_view", lambda: None)()
        storage = previous.untyped_storage() if previous is not None else raw.to(
            device=device
        ).untyped_storage()
        result = torch.empty(0, dtype=dtype, device=device).set_(
            storage, offset, shape, stride
        )
        raw._restored_view = weakref.ref(result)
        return result


def checkpoint_call(ledger: ReplayLedger, kind: str, function, *args, boundary=None):
    import torch
    from torch.utils.checkpoint import checkpoint

    calls = 0

    def measured(*inputs):
        nonlocal calls
        phase = "forward" if calls == 0 else "recompute"
        calls += 1
        ledger.start(kind, phase)
        try:
            result = function(*inputs)
        except Exception as exc:
            if phase == "recompute" and type(exc).__name__ == "_StopRecomputationError":
                ledger.finish(kind, phase)
            raise
        if boundary is not None:
            boundary(phase, result)
        ledger.finish(kind, phase)
        return result

    storage = BoundaryStorage() if kind == "vae_chunk" else None
    context = (torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack)
               if storage is not None else nullcontext())
    try:
        with context:
            return checkpoint(measured, *args, use_reentrant=False,
                              preserve_rng_state=True)
    finally:
        if storage is not None:
            storage.copies.clear()


def enable_transformer_checkpointing(transformer, ledger: ReplayLedger) -> None:
    def counted(module, *args):
        return checkpoint_call(ledger, "transformer_block", module, *args)
    transformer.enable_gradient_checkpointing(gradient_checkpointing_func=counted)


def disable_transformer_checkpointing(transformer) -> None:
    disable = getattr(transformer, "disable_gradient_checkpointing", None)
    if callable(disable):
        disable()


def checkpoint_decode(vae, latent, ledger: ReplayLedger):
    """Run native Wan decode with causal-cache chunk checkpoint boundaries."""
    import torch

    if vae.use_tiling:
        raise ValueError("gradient VAE path excludes spatial tiling")
    if latent.shape[0] != 1:
        raise ValueError("single-video checkpoint decoder required")
    decoder = vae.decoder
    original_forward = decoder.forward
    decode_id = ledger.new_decode()
    call_index = 0

    def checkpointed_forward(x, feat_cache=None, feat_idx=None, first_chunk=False):
        nonlocal call_index
        if feat_cache is None or feat_idx is None or feat_idx[0] != 0:
            raise ValueError("native decoder cache/cursor unavailable")
        layout, inputs = [], []
        for item in feat_cache:
            if torch.is_tensor(item):
                layout.append(("tensor", len(inputs)))
                inputs.append(item)
            elif item is None:
                layout.append(("none", None))
            elif item == "Rep":
                layout.append(("rep", None))
            else:
                raise ValueError("unexpected decoder cache input")
        layout = tuple(layout)
        expected_layout = None
        expected_consumed = None
        chunk_index = call_index

        def functional_frame(x_value, *cache_values):
            nonlocal expected_layout, expected_consumed
            local = [
                cache_values[index] if kind == "tensor" else ("Rep" if kind == "rep" else None)
                for kind, index in layout
            ]
            cursor = [0]
            output = original_forward(
                x_value, feat_cache=local, feat_idx=cursor, first_chunk=first_chunk
            )
            consumed = int(cursor[0])
            if consumed <= 0 or consumed > len(local):
                raise ValueError("decoder cache consumption invalid")
            if expected_consumed is None:
                expected_consumed = consumed
            elif consumed != expected_consumed:
                raise ValueError("decoder replay changed cache consumption")
            kinds, tensors = [], []
            for item in local:
                if torch.is_tensor(item):
                    kinds.append("tensor")
                    tensors.append(item)
                elif item is None:
                    kinds.append("none")
                elif item == "Rep":
                    kinds.append("rep")
                else:
                    raise ValueError("unexpected decoder cache output")
            kinds = tuple(kinds)
            if expected_layout is None:
                expected_layout = kinds
            elif kinds != expected_layout:
                raise ValueError("decoder replay changed cache layout")
            return (output, *tensors)

        outputs = checkpoint_call(
            ledger, "vae_chunk", functional_frame, x, *inputs,
            boundary=lambda phase, result: ledger.record_boundary(
                decode_id, phase, chunk_index, result[1:]
            ),
        )
        if expected_layout is None or len(outputs) != 1 + expected_layout.count("tensor"):
            raise ValueError("decoder checkpoint output malformed")
        tensors = iter(outputs[1:])
        feat_cache[:] = [
            next(tensors) if kind == "tensor" else ("Rep" if kind == "rep" else None)
            for kind in expected_layout
        ]
        feat_idx[0] = expected_consumed
        call_index += 1
        return outputs[0]

    decoder.forward = checkpointed_forward
    try:
        decoded = vae.decode(latent, return_dict=False)[0]
        if call_index != latent.shape[2]:
            raise ValueError("native decoder chunk count differs from latent frames")
        return decoded
    finally:
        decoder.forward = original_forward
