"""Cache-preserving Wan decoder checkpoint replay for adopted M0. No resource admission quotas."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


TRANSFER_BYTES = 16 * 1024 * 1024


class BoundarySpool:
    """Own exact VAE checkpoint storage until its VJP finishes."""

    def __init__(self):
        kind = self.kind = "VAE"
        root = Path(os.environ.get("M0_BOUNDARY_SPOOL_ROOT") or (
            "/content" if Path("/content").is_dir() else "/tmp"
        ))
        free = shutil.disk_usage(root).free
        self.directory = tempfile.TemporaryDirectory(
            prefix=f"wan-{kind.lower()}-boundary-", dir=root
        )
        self.path = Path(self.directory.name)
        self.disk_root = str(root)
        self.disk_free_at_start_bytes = free
        self.live_disk_bytes = self.peak_disk_bytes = 0
        self.cpu_live_packed_bytes = self.cpu_peak_packed_bytes = 0
        self.d2h_bytes = self.h2d_bytes = 0
        self.disk_written_bytes = self.disk_read_bytes = 0
        self.files = 0
        self.closed = False

    def _cpu_live(self, amount):
        self.cpu_live_packed_bytes = amount
        self.cpu_peak_packed_bytes = max(self.cpu_peak_packed_bytes, amount)

    def pack(self, storage, device):
        import torch

        size = storage.nbytes()
        if self.closed:
            raise RuntimeError("boundary spool already closed")
        fd, name = tempfile.mkstemp(dir=self.path, prefix="storage-")
        raw = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (size,), (1,)
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                for start in range(0, size, TRANSFER_BYTES):
                    chunk = raw[start:start + TRANSFER_BYTES].to("cpu", copy=True)
                    self._cpu_live(chunk.numel())
                    chunk.numpy().tofile(stream)
                    if device.type == "cuda":
                        self.d2h_bytes += chunk.numel()
                    self.disk_written_bytes += chunk.numel()
                    del chunk
                    self._cpu_live(0)
            self.live_disk_bytes += size
            self.peak_disk_bytes = max(self.peak_disk_bytes, self.live_disk_bytes)
            self.files += 1
            return (name, size)
        except BaseException:
            self._cpu_live(0)
            os.unlink(name)
            raise

    def restore(self, record, device):
        import numpy as np
        import torch

        name, size = record
        raw = torch.empty(size, dtype=torch.uint8, device=device)
        try:
            with open(name, "rb") as stream:
                for start in range(0, size, TRANSFER_BYTES):
                    length = min(TRANSFER_BYTES, size - start)
                    chunk = np.fromfile(stream, dtype=np.uint8, count=length)
                    if len(chunk) != length:
                        raise IOError("VAE boundary storage truncated")
                    self._cpu_live(chunk.nbytes)
                    raw[start:start + length].copy_(torch.from_numpy(chunk))
                    if device.type == "cuda":
                        self.h2d_bytes += length
                    self.disk_read_bytes += length
                    del chunk
                    self._cpu_live(0)
        except BaseException:
            self._cpu_live(0)
            raise
        return raw

    def summary(self):
        return dict(kind=self.kind,
                    cpu_live_packed_bytes=self.cpu_live_packed_bytes,
                    cpu_peak_packed_bytes=self.cpu_peak_packed_bytes,
                    cpu_transfer_buffer_limit_bytes=TRANSFER_BYTES,
                    disk_live_bytes=self.live_disk_bytes,
                    disk_peak_bytes=self.peak_disk_bytes,
                    disk_root=self.disk_root,
                    disk_free_at_start_bytes=self.disk_free_at_start_bytes,
                    files=self.files, d2h_bytes=self.d2h_bytes,
                    h2d_bytes=self.h2d_bytes,
                    disk_written_bytes=self.disk_written_bytes,
                    disk_read_bytes=self.disk_read_bytes, closed=self.closed)

    def close(self):
        if not self.closed:
            self.directory.cleanup()
            self.live_disk_bytes = 0
            self.cpu_live_packed_bytes = 0
            self.closed = True


class ReplayLedger:
    """Actual original/replay chunks; one owned spool per sequential objective VJP."""
    def __init__(self, callback=None):
        self.counts = {"vae_chunk": {p: {"attempted": 0, "completed": 0}
                                   for p in ("forward", "recompute")}}
        self.boundaries = {}
        self.boundary_spool = None
        self.boundary_storage_final = None
        self.callback = callback

    def _emit(self):
        if self.callback is not None:
            self.callback(self.summary())

    def start(self, kind, phase):
        self.counts[kind][phase]["attempted"] += 1
        self._emit()

    def finish(self, kind, phase):
        self.counts[kind][phase]["completed"] += 1
        self._emit()

    def new_decode(self):
        self.boundary_spool = BoundarySpool()
        self.boundaries[0] = {p: [] for p in ("forward", "recompute")}
        self._emit()
        return 0

    def record_boundary(self, decode_id, phase, chunk_index, cache):
        unique = {(str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
                  for x in cache if hasattr(x, "untyped_storage")}
        self.boundaries[decode_id][phase].append(dict(index=chunk_index,
            tensor_slots=len(cache), unique_storage_bytes=sum(unique.values())))
        self._emit()

    def summary(self):
        return dict(counts=self.counts, cache_boundaries=self.boundaries,
                    boundary_storage=(self.boundary_spool.summary() if self.boundary_spool
                                      else self.boundary_storage_final))

    def release_boundary_storage(self):
        if self.boundary_spool is not None:
            self.boundary_spool.close()
            self.boundary_storage_final = self.boundary_spool.summary()
            self.boundary_spool = None
            self._emit()


class BoundaryStorage:
    """Pack alias-preserving checkpoint boundary storage to disk."""

    def __init__(self, spool):
        self.spool = spool
        self.copies = {}
        self.restored = {}

    def pack(self, tensor):
        import torch

        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr(), storage.nbytes())
        if key not in self.copies:
            self.copies[key] = (storage, self.spool.pack(storage, tensor.device))
        return (self.copies[key][1], tensor.device, tensor.dtype, tensor.storage_offset(),
                tuple(tensor.shape), tuple(tensor.stride()))

    def unpack(self, packed):
        import torch

        record, device, dtype, offset, shape, stride = packed
        try:
            # Autograd may replace a returned Tensor wrapper. Keep the underlying
            # storage through this one recompute so overlapping/strided aliases
            # survive; measured() releases it on return, early stop or failure.
            if record not in self.restored:
                self.restored[record] = self.spool.restore(record, device).untyped_storage()
            return torch.empty(0, dtype=dtype, device=device).set_(
                self.restored[record], offset, shape, stride)
        except BaseException:
            self.restored.clear()
            raise


def checkpoint_call(ledger: ReplayLedger, kind: str, function, *args, boundary=None):
    import torch
    from torch.utils.checkpoint import checkpoint

    calls = 0

    def measured(*inputs):
        nonlocal calls
        phase = "forward" if calls == 0 else "recompute"
        calls += 1
        try:
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
        finally:
            storage.restored.clear()

    spool = ledger.boundary_spool
    if spool is None:
        raise RuntimeError(f"checkpoint boundary storage missing for {kind}")
    storage = BoundaryStorage(spool)
    # The outer hook packs the non-reentrant checkpoint's saved inputs. Its
    # inner hook still owns the intra-block recomputation placeholders.
    context = torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack)
    try:
        with context:
            return checkpoint(measured, *args, use_reentrant=False,
                              preserve_rng_state=True)
    finally:
        storage.copies.clear()


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
