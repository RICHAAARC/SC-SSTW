"""Passive adapters for fresh P0/capture, original CPU solve/replay and native Y."""
from pathlib import Path


class Backend:
    def __init__(self, config, output, scratch, event, dump):
        self.config, self.output, self.scratch = config, Path(output), Path(scratch)
        self.event, self.dump = event, dump
        self.native = self.media = self.cpu = None
        self.generation_loaded = False

    def generation_load(self):
        from runtime.wan.b_joint_time_packet_a_native_v1 import Backend as Native
        self.native = Native(self.config["model_id"])
        result = self.native.wan.generation_load(self.config["source"], self.config)
        self.generation_loaded = True
        return result

    def generate(self):
        import torch
        traces = []
        def record(row):
            traces.append(row)
            self.dump(self.output / "private/P0_trace.json", traces)
        try:
            z, receipt = self.native.wan.generate("P0", self.config["writer"], self.event, record)
            torch.save(z, self.output / "private/P0_terminal.pt")
            self.dump(self.output / "private/P0_generation.json", receipt)
            return receipt
        finally:
            self.native.wan.release_generation()
            self.generation_loaded = False

    def vae_load(self):
        if self.native is None:
            from runtime.wan.b_joint_time_packet_a_native_v1 import Backend as Native
            self.native = Native(self.config["model_id"])
        return self.native.load()

    def capture(self):
        return self.native.capture(self.output / "private/P0_terminal.pt", self.scratch / "host", self.event)

    def head(self):
        import torch
        from runtime.wan.b_framewise_qim_v1_operator import validate_head
        from runtime.wan.b_joint_time_packet_a_write_cpu_v1 import validate_head_data
        conv = self.native.wan.vae.decoder.conv_out
        validate_head(conv)
        value = dict(weight=conv.weight.detach().cpu().float(), bias=conv.bias.detach().cpu().float(),
                     causal_contract=dict(padding=list(conv._padding), stride=list(conv.stride),
                                          dilation=list(conv.dilation), groups=conv.groups))
        validate_head_data(value)
        torch.save(value, self.output / "private/head.pt")
        return dict(shape=list(value["weight"].shape), actual_loaded_head=True)

    def q0(self):
        import numpy as np
        import torch
        from runtime.wan.b_joint_time_packet_a_media_v1 import Backend as Media
        from runtime.wan.b_joint_time_packet_a_write_cpu_v1 import Backend as Solver, load_head
        # The 8-GiB receiver observer ends before the original 16-GiB CPU solver.
        medium = Media()
        try:
            pre = np.load(self.scratch / "host/BASE_preclip.npy", mmap_mode="r", allow_pickle=False)
            raw = medium.rgb_raw(pre, "preclip")
            path = self.output / "private/BASE_preclip_raw.npz"
            np.savez(path, **raw)
        finally:
            self.dump(self.output / "q0_resources.json", medium.close())
        head = load_head(self.output / "private/head.pt")
        self.cpu = Solver(head, self.output / "cpu", self.event, self.dump)
        self.cpu.read_q0(dict(file=str(path), role="BASE", layer="preclip", units="255*RGB_luma_DCT", frames=181))
        return dict(saved_raw=str(path), solver_device="cpu", dtype=str(self.cpu.q0.dtype))

    def target(self):
        self.cpu.target()
        return dict(fixed_targets=True)

    def solve(self):
        answer = self.cpu.solve(lambda row: self.dump(self.output / "cpu/solver_progress.json", row))
        if not answer.get("native_handoff_eligible"):
            raise RuntimeError("Original solver produced no finite eligible candidate; no replay/retry")
        return answer

    def predict(self):
        return self.cpu.predict()

    def handoff(self):
        return self.cpu.handoff(self.config["source"]["id"])

    def replay(self):
        import shutil
        temporary = self.scratch / "new"
        def receipt(path, value):
            # Small measured writer/quality/failure receipts are permanent; large pixels are cache.
            self.dump(self.output / "writer_native" / Path(path).relative_to(temporary), value)
        result = self.native.replay(self.output / "cpu", self.scratch / "host/BASE_preclip.npy",
                                    self.scratch / "host/features", temporary, self.event, receipt)
        difference = temporary / "host_q0_difference.npy"
        if difference.is_file():
            shutil.copyfile(difference, self.output / "writer_native/host_q0_difference.npy")
        return result

    def media_backend(self):
        if self.media is None:
            from runtime.wan.b_joint_time_packet_a_media_v1 import Backend as Media
            self.media = Media()
        return self.media

    def preclip(self, arm):
        import numpy as np
        path = self.scratch / ("new/NEW_preclip.npy" if arm == "NEW" else "host/BASE_preclip.npy")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def quantize(self, preclip):
        return self.media_backend().quantize(preclip)

    def encode(self, pixels, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.media_backend().encode(pixels, path)

    def probe(self, path, frames):
        return self.media_backend().probe(path, frames)

    def decode(self, path, frames):
        return self.media_backend().decode(path, frames)

    def decode_y(self, path, frames, probe):
        return self.media_backend().decode_y(path, frames, probe)

    def hard_receive(self, native_y, folder):
        import numpy as np
        from main.tube_state import b_native_y_diagnostic_v1 as native_y_reader
        from main.tube_state import b_joint_time_packet_a_receiver_v1 as receiver
        medium = self.media_backend()
        folder.mkdir(parents=True, exist_ok=True)
        with medium.resources.receiving(600):
            raw = native_y_reader.read_y(native_y, medium.resources.check)
            np.savez(folder / "readout.npz", **raw)
            # Original uncorrected raw_bits/known and original BCH parser, no hinge/global score.
            packet = receiver.diagnostics(raw["soft"], raw["mask"], "NEW", medium.resources.check)
            self.dump(folder / "packets.json", packet)
            return packet

    def close(self):
        result = {}
        if self.generation_loaded and self.native is not None:
            self.native.wan.release_generation()
        if self.native is not None:
            result["native"] = self.native.close()
            result["native"].pop("E41_seconds", None)  # Old source-named empty timer is not a new read.
            result["native"]["identity_encodes_this_stage"] = 0
        if self.media is not None:
            result["media"] = self.media.close()
        return result
