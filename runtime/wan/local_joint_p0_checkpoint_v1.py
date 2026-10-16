"""One owned DiT branch graph: native non-reentrant block checkpoint + disk boundaries."""
from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile

from runtime.wan.local_joint_readout_checkpoint_v1 import BoundarySpool, BoundaryStorage, TRANSFER_BYTES


class DiTSpool(BoundarySpool):
    """Reuse storage/view restoration, counting partial writes even on failure."""
    def __init__(self, root=None):
        super().__init__(kind="DiT", root=root)

    def pack(self, storage, device):
        import torch
        if self.closed:
            raise RuntimeError("DiT boundary spool already closed")
        size = storage.nbytes()
        fd, name = tempfile.mkstemp(dir=self.path, prefix="storage-")
        raw = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (size,), (1,))
        written = 0
        try:
            with os.fdopen(fd, "wb", buffering=0) as stream:
                for start in range(0, size, TRANSFER_BYTES):
                    chunk = raw[start:start+TRANSFER_BYTES].to("cpu", copy=True)
                    self._cpu_live(chunk.numel())
                    if device.type == "cuda":
                        self.d2h_bytes += chunk.numel()
                    try:
                        chunk.numpy().tofile(stream)
                    finally:
                        actual = os.fstat(stream.fileno()).st_size
                        increment = actual - written
                        written = actual
                        self.live_disk_bytes += increment
                        self.disk_written_bytes += increment
                        self.peak_disk_bytes = max(self.peak_disk_bytes, self.live_disk_bytes)
                    del chunk
                    self._cpu_live(0)
            self.files += 1
            return name, size
        except BaseException:
            self._cpu_live(0)
            os.unlink(name)
            self.live_disk_bytes -= written
            raise


class BlockCheckpoint:
    """Count every native block entry, with at most one replay per block."""
    def __init__(self, transformer, callback, *, root=None):
        self.blocks = tuple(transformer.blocks)
        self.indices = {id(block): i for i, block in enumerate(self.blocks)}
        self.rows = [{"index": i, "forward": "NOT_RUN", "recompute": "NOT_RUN"}
                     for i in range(len(self.blocks))]
        self.spool = DiTSpool(root)
        self.callback = callback

    def summary(self):
        return dict(blocks=[dict(r) for r in self.rows], boundary_storage=self.spool.summary(),
                    counts={phase: {status: sum(r[phase] == status for r in self.rows)
                                    for status in ("NOT_RUN", "RUNNING", "RETURNED", "FAILED",
                                                   "INTERRUPTED_COMPLETION_UNKNOWN")}
                            for phase in ("forward", "recompute")})

    def emit(self):
        self.callback(self.summary())

    def __call__(self, module, *args):
        import torch
        from torch.utils.checkpoint import checkpoint
        if id(module) not in self.indices:
            raise ValueError("checkpoint received a block outside the loaded Transformer")
        row = self.rows[self.indices[id(module)]]
        storage = BoundaryStorage(self.spool)
        entries = 0

        def measured(*inputs):
            nonlocal entries
            phase = "forward" if entries == 0 else "recompute"
            if entries >= 2 or row[phase] != "NOT_RUN":
                raise RuntimeError("fixed P0 permits each block once plus at most one replay")
            entries += 1
            row[phase] = "RUNNING"
            self.emit()
            try:
                value = module(*inputs)
                row[phase] = "RETURNED"
                return value
            except BaseException as exc:
                early = phase == "recompute" and type(exc).__name__ == "_StopRecomputationError"
                row[phase] = ("RETURNED" if early else "FAILED" if isinstance(exc, Exception)
                              else "INTERRUPTED_COMPLETION_UNKNOWN")
                raise
            finally:
                storage.restored.clear()
                self.emit()

        try:
            with torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack):
                return checkpoint(measured, *args, use_reentrant=False, preserve_rng_state=True)
        finally:
            # Do not retain original GPU backing storages in the pack dedup map.
            storage.copies.clear()

    def close(self):
        self.spool.close()
        self.emit()


@contextmanager
def checkpointed_branch(transformer, callback, *, root=None):
    old_enabled = getattr(transformer, "gradient_checkpointing", False)
    old_function = getattr(transformer, "_gradient_checkpointing_func", None)
    ledger = BlockCheckpoint(transformer, callback, root=root)
    try:
        transformer.enable_gradient_checkpointing(gradient_checkpointing_func=ledger)
        yield ledger
    finally:
        transformer.disable_gradient_checkpointing()
        if old_enabled:
            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=old_function)
        ledger.close()
