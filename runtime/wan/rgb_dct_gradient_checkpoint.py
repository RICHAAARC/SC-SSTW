"""Bounded native causal-frame recomputation for local gradients."""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile
import weakref


TRANSFER_BYTES = 16 * 1024 * 1024
DISK_LIMIT_BYTES = 80 * 1024**3
DISK_FREE_MARGIN_BYTES = 2 * 1024**3
# 46 causal chunks: about 0.70 GiB first + 45 * 1.44 GiB thereafter.
# Reserve the complete disk budget before decoding, including headroom.
EXPECTED_BOUNDARY_BYTES = DISK_LIMIT_BYTES


class BoundarySpool:
    """Own exact checkpoint storage until its VAE VJP has finished."""

    def __init__(self):
        root = Path(os.environ.get("RGB_DCT_BOUNDARY_SPOOL_ROOT") or (
            "/content" if Path("/content").is_dir() else "/tmp"
        ))
        free = shutil.disk_usage(root).free
        if free < EXPECTED_BOUNDARY_BYTES + DISK_FREE_MARGIN_BYTES:
            raise RuntimeError("VAE_BOUNDARY_DISK_PREFLIGHT")
        self.directory = tempfile.TemporaryDirectory(
            prefix="wan-vae-boundary-", dir=root
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
        if self.closed or self.live_disk_bytes + size > DISK_LIMIT_BYTES:
            raise RuntimeError("VAE_BOUNDARY_DISK_BUDGET")
        if shutil.disk_usage(self.path).free < size + DISK_FREE_MARGIN_BYTES:
            raise RuntimeError("VAE_BOUNDARY_DISK_FREE")
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
        return dict(cpu_live_packed_bytes=self.cpu_live_packed_bytes,
                    cpu_peak_packed_bytes=self.cpu_peak_packed_bytes,
                    cpu_transfer_buffer_limit_bytes=TRANSFER_BYTES,
                    disk_live_bytes=self.live_disk_bytes,
                    disk_peak_bytes=self.peak_disk_bytes,
                    disk_limit_bytes=DISK_LIMIT_BYTES,
                    preflight_boundary_bytes=EXPECTED_BOUNDARY_BYTES,
                    disk_free_margin_bytes=DISK_FREE_MARGIN_BYTES,
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


class CheckpointLedger:
    """Count native decoder calls and retain each sequential decode receipt."""

    def __init__(self, count, forward_limit: int, recompute_limit: int):
        if (type(forward_limit) is not int or forward_limit < 0
                or type(recompute_limit) is not int or recompute_limit < 0):
            raise ValueError("nonnegative checkpoint call limits required")
        self.count = count
        self.limits = dict(vae_chunk_forward=forward_limit,
                           vae_chunk_recompute=recompute_limit)
        self.boundaries = {}
        self._seen_storage = {}
        self.boundary_spool = None
        self.boundary_storage_history = []

    def start(self, phase: str) -> None:
        self.count(f"vae_chunk_{phase}", False)

    def finish(self, phase: str) -> None:
        self.count(f"vae_chunk_{phase}", True)

    def new_decode(self) -> int:
        if self.boundary_spool is not None:
            raise RuntimeError("previous VAE boundary spool was not released")
        self.boundary_spool = BoundarySpool()
        index = len(self.boundaries)
        self.boundaries[index] = {
            phase: {"chunks": [], "unique_storage_bytes": 0}
            for phase in ("forward", "recompute")
        }
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
    def summary(self) -> dict:
        storage = self.boundary_spool.summary() if self.boundary_spool else None
        receipts = [*self.boundary_storage_history]
        if storage is not None:
            receipts.append(storage)
        aggregate = dict(
            decodes=len(receipts), closed_decodes=len(self.boundary_storage_history),
            d2h_bytes=sum(row["d2h_bytes"] for row in receipts),
            h2d_bytes=sum(row["h2d_bytes"] for row in receipts),
            disk_written_bytes=sum(row["disk_written_bytes"] for row in receipts),
            disk_read_bytes=sum(row["disk_read_bytes"] for row in receipts),
            files=sum(row["files"] for row in receipts),
            disk_peak_bytes=max((row["disk_peak_bytes"] for row in receipts), default=0),
            cpu_peak_packed_bytes=max((row["cpu_peak_packed_bytes"] for row in receipts),
                                      default=0),
            active_disk_live_bytes=storage["disk_live_bytes"] if storage else 0,
            active_cpu_live_packed_bytes=storage["cpu_live_packed_bytes"] if storage else 0,
        )
        return dict(limits=self.limits, cache_boundaries=self.boundaries,
                    boundary_storage=storage,
                    boundary_storage_history=list(self.boundary_storage_history),
                    boundary_storage_aggregate=aggregate)

    def release_boundary_storage(self):
        if self.boundary_spool is not None:
            self.boundary_spool.close()
            self.boundary_storage_history.append(self.boundary_spool.summary())
            self.boundary_spool = None


class BoundaryStorage:
    """Pack alias-preserving checkpoint boundary storage to bounded disk."""

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
        live = [ref for ref in self.restored.get(record, ()) if ref() is not None]
        previous = live[0]() if live else None
        storage = (previous.untyped_storage() if previous is not None else
                   self.spool.restore(record, device).untyped_storage())
        result = torch.empty(0, dtype=dtype, device=device).set_(
            storage, offset, shape, stride
        )
        live.append(weakref.ref(result))
        self.restored[record] = live
        return result


def checkpoint_call(ledger: CheckpointLedger, function, *args, boundary=None):
    import torch
    from torch.utils.checkpoint import checkpoint

    calls = 0

    def measured(*inputs):
        nonlocal calls
        phase = "forward" if calls == 0 else "recompute"
        calls += 1
        ledger.start(phase)
        try:
            result = function(*inputs)
        except Exception as exc:
            if phase == "recompute" and type(exc).__name__ == "_StopRecomputationError":
                ledger.finish(phase)
            raise
        if boundary is not None:
            boundary(phase, result)
        ledger.finish(phase)
        return result

    storage = BoundaryStorage(ledger.boundary_spool) if ledger.boundary_spool else None
    context = (torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack)
               if storage else nullcontext())
    try:
        with context:
            return checkpoint(measured, *args, use_reentrant=False,
                              preserve_rng_state=True)
    finally:
        if storage:
            storage.copies.clear()


def checkpoint_decode(vae, latent, ledger: CheckpointLedger):
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
            ledger, functional_frame, x, *inputs,
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
