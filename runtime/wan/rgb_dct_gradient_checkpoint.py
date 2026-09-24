"""Causal-frame VAE checkpointing with CPU-packed boundary tensors."""
from __future__ import annotations

from typing import Any, Callable


class CheckpointLedger:
    """Count every native decoder-frame forward and backward recomputation."""

    def __init__(self, count: Callable[[str, bool], None],
                 forward_limit: int, recompute_limit: int):
        if (type(forward_limit) is not int or forward_limit < 0
                or type(recompute_limit) is not int or recompute_limit < 0):
            raise ValueError("nonnegative checkpoint call limits required")
        self.count = count
        self.limits = dict(vae_chunk_forward=forward_limit,
                           vae_chunk_recompute=recompute_limit)
        self.boundaries: dict[int, dict] = {}
        self._seen_boundary_storage: dict[tuple[int, str], set] = {}

    def new_decode(self) -> int:
        index = len(self.boundaries)
        self.boundaries[index] = {
            phase: dict(chunks=[], unique_storage_bytes=0)
            for phase in ("forward", "recompute")}
        return index

    def start(self, phase: str) -> None:
        kind = f"vae_chunk_{phase}"
        self.count(kind, False)

    def finish(self, phase: str) -> None:
        kind = f"vae_chunk_{phase}"
        self.count(kind, True)

    def record_boundary(self, decode_id: int, phase: str,
                        chunk_index: int, cache: tuple) -> None:
        unique = {}
        logical = 0
        for value in cache:
            if not hasattr(value, "untyped_storage"):
                continue
            storage = value.untyped_storage()
            key = (value.device.type, value.device.index,
                   storage.data_ptr(), storage.nbytes())
            unique[key] = storage.nbytes()
            logical += value.numel() * value.element_size()
        record = self.boundaries[decode_id][phase]
        seen = self._seen_boundary_storage.setdefault((decode_id, phase), set())
        record["unique_storage_bytes"] += sum(
            size for key, size in unique.items() if key not in seen)
        seen.update(unique)
        record["chunks"].append(dict(index=chunk_index,
                                     tensor_slots=sum(hasattr(x, "untyped_storage")
                                                      for x in cache),
                                     logical_bytes=logical,
                                     unique_storage_bytes=sum(unique.values())))

    def summary(self) -> dict:
        return dict(limits=self.limits, cache_boundaries=self.boundaries)


class BoundaryStorage:
    """One checkpoint input pack with one CPU copy per aliased GPU storage."""

    def __init__(self):
        self.copies = {}

    def pack(self, tensor: Any) -> tuple:
        import torch

        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr(), storage.nbytes())
        if key not in self.copies:
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            self.copies[key] = raw.to(device="cpu", copy=True)
        return (self.copies[key], tensor.device, tensor.dtype,
                tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))

    @staticmethod
    def unpack(packed: tuple) -> Any:
        import torch
        import weakref

        raw, device, dtype, offset, shape, stride = packed
        previous = getattr(raw, "_restored_view", lambda: None)()
        storage = (previous.untyped_storage() if previous is not None
                   else raw.to(device=device).untyped_storage())
        result = torch.empty(0, dtype=dtype, device=device).set_(
            storage, offset, shape, stride)
        raw._restored_view = weakref.ref(result)
        return result


def checkpoint_call(ledger: CheckpointLedger, function: Callable,
                    *args: Any, boundary: Callable | None = None) -> Any:
    import torch
    from torch.utils.checkpoint import checkpoint

    calls = 0
    def measured(*inputs: Any) -> Any:
        nonlocal calls
        phase = "forward" if calls == 0 else "recompute"
        calls += 1
        ledger.start(phase)
        try:
            result = function(*inputs)
        except Exception as exc:
            # Non-reentrant checkpoint may stop once every requested tensor is
            # available. That partial replay is a completed recomputation call.
            if phase == "recompute" and type(exc).__name__ == "_StopRecomputationError":
                ledger.finish(phase)
            raise
        if boundary is not None:
            boundary(phase, result)
        ledger.finish(phase)
        return result

    storage = BoundaryStorage()
    context = torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack)
    try:
        with context:
            return checkpoint(measured, *args, use_reentrant=False,
                              preserve_rng_state=True)
    finally:
        storage.copies.clear()


def checkpoint_decode(vae: Any, latent: Any, ledger: CheckpointLedger) -> Any:
    """Run native Wan decode while checkpointing each causal latent frame."""
    import torch

    if vae.use_tiling:
        raise ValueError("fixed causal-frame checkpoint excludes spatial tiling")
    if latent.shape[0] != 1:
        raise ValueError("single-video checkpoint decoder required")
    decoder = vae.decoder
    original_forward = decoder.forward
    decode_id = ledger.new_decode()
    call_index = 0

    def checkpointed_forward(x, feat_cache=None, feat_idx=None, first_chunk=False):
        nonlocal call_index
        if feat_cache is None or feat_idx is None or feat_idx[0] != 0:
            raise ValueError("native decoder causal cache/cursor unavailable")
        layout = []
        cache_tensors = []
        for item in feat_cache:
            if torch.is_tensor(item):
                layout.append(("tensor", len(cache_tensors)))
                cache_tensors.append(item)
            elif item is None:
                layout.append(("none", None))
            elif item == "Rep":
                layout.append(("rep", None))
            else:
                raise ValueError("unexpected decoder input cache value")
        layout = tuple(layout)
        expected_output_layout = None
        expected_consumed = None
        chunk_index = call_index

        def functional_frame(x_value, *cache_values):
            nonlocal expected_output_layout, expected_consumed
            local = [cache_values[index] if kind == "tensor"
                     else ("Rep" if kind == "rep" else None)
                     for kind, index in layout]
            cursor = [0]
            output = original_forward(x_value, feat_cache=local,
                                      feat_idx=cursor, first_chunk=first_chunk)
            consumed = int(cursor[0])
            if consumed <= 0 or consumed > len(local):
                raise ValueError("decoder cache consumption invalid")
            if expected_consumed is None:
                expected_consumed = consumed
            elif consumed != expected_consumed:
                raise ValueError("replay changed decoder cache consumption")
            kinds = []
            tensors = []
            for item in local:
                if torch.is_tensor(item):
                    kinds.append("tensor"); tensors.append(item)
                elif item is None:
                    kinds.append("none")
                elif item == "Rep":
                    kinds.append("rep")
                else:
                    raise ValueError("unexpected decoder output cache value")
            kinds = tuple(kinds)
            if expected_output_layout is None:
                expected_output_layout = kinds
            elif kinds != expected_output_layout:
                raise ValueError("replay changed decoder cache layout")
            return (output, *tensors)

        outputs = checkpoint_call(
            ledger, functional_frame, x, *cache_tensors,
            boundary=lambda phase, result: ledger.record_boundary(
                decode_id, phase, chunk_index, result[1:]))
        if (expected_output_layout is None
                or len(outputs) != 1 + expected_output_layout.count("tensor")):
            raise ValueError("decoder checkpoint output malformed")
        tensors = iter(outputs[1:])
        feat_cache[:] = [next(tensors) if kind == "tensor"
                         else ("Rep" if kind == "rep" else None)
                         for kind in expected_output_layout]
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
