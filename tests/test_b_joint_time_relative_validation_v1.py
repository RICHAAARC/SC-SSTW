"""Artificial stages and IO only; never loads a real model or invokes codec."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main.tube_state.b_joint_time_relative_validation_v1 import (INNER_BUDGET, STAGES, construction_maps,
    denominators, operation_plan, stage_operations, units, validate)
from main.tube_state import b_joint_time_packet_a_v1 as code
from runtime.wan import b_joint_time_relative_validation_v1 as runner
from runtime.wan.native_baseline_storage_v1 import read_json, write_json


class ArtificialBackend:
    instances = []
    fail = False

    def __init__(self, config, output, scratch, event, dump):
        self.output, self.event, self.dump = Path(output), event, dump
        self.saved = {}
        self.calls = Counter()
        self.instances.append(self)

    def events(self, kind, count):
        for _ in range(count):
            self.event(kind, False); self.event(kind, True)

    def generation_load(self): return {"artificial": True}
    def generate(self):
        for kind, n in [("transformer_conditional", 50), ("transformer_unconditional", 50),
                        ("native_step", 50), ("payload_gradient", 25)]:
            self.events(kind, n)
        return {"artificial": True}
    def vae_load(self): self.calls["vae_load"] += 1; return {"artificial": True}
    def capture(self): self.events("D181", 1); self.events("baseline_head", 46); return {}
    def head(self): return {}
    def q0(self): return {}
    def target(self): return {}
    def solve(self): self.events("K", 2); self.events("KT", 2); return {"artificial": True}
    def predict(self): self.events("K", 1); return {}
    def handoff(self): return {}
    def replay(self): self.events("head_replay", 46); return {"artificial": True}
    def preclip(self, arm): return np.broadcast_to(np.arange(181, dtype=np.uint8)[:, None, None, None], (181, 2, 2, 3)).copy()
    def quantize(self, pixels): return pixels.copy()
    def encode(self, pixels, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saved[str(path)] = pixels.copy()
        self.calls["encode"] += 1
    def probe(self, path, frames):
        assert len(self.saved[str(path)]) == frames
        self.calls["probe"] += 1
        return {"artificial": True, "frames": frames}
    def decode(self, path, frames):
        self.calls["decode_RGB"] += 1
        return self.saved[str(path)].copy()
    def decode_y(self, path, frames, probe):
        self.calls["decode_Y"] += 1
        return self.saved[str(path)][..., 0].copy()
    def hard_receive(self, y, folder):
        assert not (self.output / "post_truth_access.json").exists()
        if self.fail: raise OSError("artificial hard-read IO failure")
        folder.mkdir(parents=True, exist_ok=True)
        nodes = []
        for r, pixel in enumerate(y[:, 0, 0]):
            n = int(pixel) + 10
            observations = []
            for lane in range(5):
                j = n - ((n-lane) % 5)
                bits = code.encode_raw24(8*j+n-j)
                for role in ("R0", "R1"):
                    candidate = n if role == "R0" or n % 2 == 0 else n+1
                    cj = candidate - ((candidate-lane) % 5)
                    observations.append(dict(lane=lane, role=role, physical_copy=True, raw_bits=bits,
                        known=[True]*63, status="LEGAL_LOCAL_CANDIDATE", msg=8*cj+candidate-cj, candidate_n=candidate))
            nodes.append(dict(received_frame=r, observations=observations))
        value = dict(coding="NEW", nodes=nodes)
        self.dump(folder / "packets.json", value)
        return value
    def close(self): return {"artificial": True}


class ValidationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read_json(runner.DEFAULT_CONFIG)
        ArtificialBackend.instances = []
        ArtificialBackend.fail = False

    def tearDown(self): self.temp.cleanup()

    def worker(self, stage, output, scratch):
        return runner.worker(stage, output, scratch, backend_factory=ArtificialBackend)

    def test_post_evaluation_separates_failed_missing_and_observed_unresolved(self):
        from runtime.wan.b_joint_time_relative_evaluation_v1 import evaluate_saved
        output = self.root / "accounting"
        us = units(self.config)
        write_json(output / "protocol.json", self.config)
        write_json(output / "private/attack_truth.json", construction_maps())
        write_json(output / "unit_statuses.json", {u["id"]: u for u in us})
        write_json(output / "blind_terminated.json", {"status": "BLIND_TERMINATED"})
        parent_rows = operation_plan(self.config)  # Mostly stale parent snapshot.
        for row in parent_rows:
            if row["id"] == us[6]["id"] + "/relative_compare":
                row["status"] = "RETURNED"  # Must survive a missing stage event.
        write_json(output / "ledger.json", parent_rows)
        stage = output / "stages/receive"
        write_json(stage / "plan.json", stage_operations(self.config, "receive"))
        events = []
        for index, u in enumerate(us):
            for row in stage_operations(self.config, "receive"):
                if not row["id"].startswith(u["id"] + "/"):
                    continue
                op = row["kind"]
                state = None
                if index in (0, 2): state = "RETURNED"
                elif index == 1 and op != "relative_compare":
                    state = "FAILED" if op == "hard_receive" else "RETURNED"
                elif index == 3: state = "STARTED" if op == "relative_compare" else "RETURNED"
                elif index == 5 and op == "encode": state = "RETURNED"
                if state:
                    event = dict(row, status=state)
                    if state == "FAILED": event["error"] = "Artificial read IO failure"
                    events.append(event)
        (stage / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        saved = us[0]
        old_nodes = [dict(W=[0] if r == 0 else []) for r in range(saved["frames"])]
        new_nodes = [dict(conditional_candidates=[0] if r == 0 else [],
                          comparison_performed=False, status="UNCHANGED_SINGLETON" if r == 0 else "UNCHANGED_EMPTY")
                     for r in range(saved["frames"])]
        edges = [dict(exact_singleton_pair=False, pairs=[]) for _ in range(saved["edges"])]
        relative = dict(original_f42=dict(nodes=old_nodes, edges=edges),
                        conditional_nodes=new_nodes, conditional_edges=edges)
        target = output / "blind" / saved["id"] / "relative.json"
        write_json(target, relative)
        before = target.read_bytes()
        result = evaluate_saved(output)
        rows = read_json(output / "node_results.json")
        edge_rows = read_json(output / "edge_results.json")
        self.assertEqual((len(rows), len(edge_rows)), (1644, 1634))
        states = [s["status"] for s in result["summaries"]]
        self.assertEqual(states[:7], ["BLIND_SAVED", "FAILED", "OUTPUT_MISSING", "INTERRUPTED", "NOT_STARTED", "INCOMPLETE", "OUTPUT_MISSING"])
        self.assertEqual(result["evaluated_nodes"], 181)
        self.assertEqual(result["evaluated_edges"], 180)
        self.assertTrue(result["evidence_ceiling"].startswith("partial NEW execution/saved output"))
        self.assertEqual(sum(v["nodes"] for v in result["execution_status_counts"].values()), 1644)
        self.assertEqual(sum(v["edges"] for v in result["execution_status_counts"].values()), 1634)
        good, failed = result["summaries"][:2]
        self.assertEqual(good["counts"]["conditional_correct"], 1)
        self.assertEqual(good["counts"]["conditional_unresolved"], 180)
        self.assertEqual(good["counts"]["edge_conditional_unresolved"], 180)
        self.assertIsNone(failed["counts"]["conditional_unresolved"])
        self.assertIsNone(failed["counts"]["conditional_wrong"])
        self.assertEqual(failed["execution"]["failure_operations"][0]["error"], "Artificial read IO failure")
        self.assertTrue(all(r["conditional_wrong"] is None and r["conditional_correct"] is None
                            and r["conditional_read_status"] == "NOT_EVALUABLE"
                            for r in rows + edge_rows if not r["evaluable"]))
        self.assertEqual(target.read_bytes(), before)

    def test_fixed_plan_and_two_isolated_vae_loads(self):
        validate(self.config)
        ds = denominators(self.config)
        self.assertEqual((ds["units"], ds["nodes"], ds["edges"], ds["physical_hard_bit_slots"]), (10, 1644, 1634, 1035720))
        rows = operation_plan(self.config)
        counts = Counter(r["kind"] for r in rows)
        self.assertEqual([counts[k] for k in ("encode", "probe", "decode_RGB", "decode_Y", "hard_receive", "relative_compare")], [12, 12, 12, 10, 10, 10])
        self.assertEqual(counts["capture_vae_load"]+counts["replay_vae_load"], 2)
        self.assertEqual(len({r["id"] for s in STAGES for r in stage_operations(self.config, s)}), len(rows))
        self.assertEqual(sum(len(stage_operations(self.config, s)) for s in STAGES), len(rows))
        self.assertTrue(all("E41" not in r["id"] for r in rows))
        maps = construction_maps()
        self.assertEqual(maps["CROP"][0:2], [38, 39])
        self.assertEqual(maps["REPEAT1"][44:47], [44, 44, 45])

    def test_artificial_end_to_end_blind_then_evaluation_and_no_reentry_calls(self):
        output = self.root / "attempt"
        with patch("builtins.print"):
            result = runner.run(self.config, output, self.root / "work", worker_runner=self.worker)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(ArtificialBackend.instances), 4)
        self.assertEqual(sum(b.calls["vae_load"] for b in ArtificialBackend.instances), 2)
        self.assertEqual(ArtificialBackend.instances[-1].calls, dict(encode=12, probe=12, decode_RGB=12, decode_Y=10))
        nodes, edges = read_json(output / "node_results.json"), read_json(output / "edge_results.json")
        self.assertEqual((len(nodes), len(edges)), (1644, 1634))
        self.assertTrue(all(r["previous_correct_lost"] is False for r in nodes+edges))
        self.assertTrue(any(r["comparable"] for r in nodes))
        self.assertTrue(any(r["base_output_without_written_time"] for r in nodes))
        internal = read_json(output / "stages/solve/internal_calls.json")
        self.assertEqual(internal["K"], dict(limit=4201, started=3, returned=3))
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
        with patch.object(runner, "evaluate_saved", side_effect=AssertionError("no repeated evaluation")):
            result = runner.run(self.config, output, self.root / "work", worker_runner=lambda *a: self.fail("no stage retry"))
        self.assertEqual(result["mode"], "READ_ONLY")
        self.assertTrue(all(r["status"] == "RETURNED" for r in result["operations"]))
        self.assertEqual(before, {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()})

    def test_io_failure_retains_all_nodes_edges_and_consumes_stage(self):
        ArtificialBackend.fail = True
        output = self.root / "failed"
        with patch("builtins.print"):
            result = runner.run(self.config, output, self.root / "work", worker_runner=self.worker)
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertEqual((len(read_json(output / "node_results.json")), len(read_json(output / "edge_results.json"))), (1644, 1634))
        ledger = read_json(output / "ledger.json")
        self.assertEqual(sum(r["status"] == "FAILED" for r in ledger), 1)
        self.assertGreater(sum(r["status"] == "NOT_STARTED" for r in ledger), 1)
        self.assertEqual(runner.worker("receive", output, self.root / "work")["status"], "READ_ONLY")
        self.assertEqual(read_json(output / "evaluation.json")["evidence_ceiling"], "no completed NEW read; execution incomplete")

    def test_internal_budget_and_post_failure_are_terminal(self):
        events = runner.InnerEvents(self.root / "counter", "solve")
        events.phase = "solver"; events.rows["K"]["started"] = 4200
        with self.assertRaises(RuntimeError): events("K", False)
        events.phase = "prediction"; events("K", False)
        with self.assertRaises(RuntimeError): events("K", False)
        output = self.root / "post"
        with patch.object(runner, "evaluate_saved", side_effect=OSError("artificial post save failure")):
            result = runner.run(self.config, output, self.root / "work", worker_runner=lambda *a: {"status": "STOPPED_FIXED_ATTEMPT"})
        self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
        self.assertTrue((output / "evaluation_failure.json").is_file())

    def test_bootstrap_failure_reserves_full_units_and_notebook_sync(self):
        from runtime.wan import b_joint_time_relative_colab_v1 as entry
        from scripts.build_b_joint_time_relative_validation_v1_notebook import build
        copies = []
        class Cloud:
            def directory(self, create=False): return "folder" if create else None
            def children(self, parent): return []
            def write_json(self, parent, name, value): copies.append((name, deepcopy(value)))
        output = self.root / "cloud"
        with patch.object(entry, "Cloud", Cloud), patch.object(entry, "OUTPUT", output), \
             patch.object(entry, "WORK", self.root / "cache"), patch("builtins.print"), \
             patch.object(entry, "download_source", side_effect=OSError("artificial setup failure")) as download:
            result = entry.run_colab()
            self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
            self.assertEqual(len(read_json(output / "planned_units.json")), 10)
            self.assertEqual(result["planned_nodes"], 1644)
            self.assertEqual(result["planned_edges"], 1634)
            self.assertEqual(len(copies[0][1]["planned_units"]), 10)
            self.assertEqual(entry.run_colab()["status"], "READ_ONLY")
            self.assertEqual(download.call_count, 1)
        (self.root / "notebooks").mkdir()
        nb = build(self.root)
        self.assertEqual(nb.read_bytes(), (runner.DEFAULT_CONFIG.parents[3] / "notebooks/b_joint_time_relative_validation_v1.ipynb").read_bytes())
        self.assertEqual("".join(json.loads(nb.read_text())["cells"][0]["source"]), "from google.colab import drive\ndrive.mount('/content/drive')\n")

    def test_notebook_interrupt_stops_descendant_after_parent_exits(self):
        import subprocess
        import sys
        import time
        from runtime.wan.b_joint_time_relative_colab_v1 import stop_worker_group
        marker = self.root / "child_pid"
        child = ("import os,signal,time; from pathlib import Path; "
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)")
        parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
        process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists(), "Artificial child did not start")
            pid = int(marker.read_text())
            stop_worker_group(process)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] in ("Z", "X"):
                    break
                time.sleep(.02)
            else:
                self.fail("Descendant remains running after parent termination")
            self.assertIsNotNone(process.returncode)
        finally:
            stop_worker_group(process)

    def test_saved_stopped_runner_also_stops_orphan_descendant(self):
        import os
        import subprocess
        import sys
        import time
        from runtime.wan import b_joint_time_relative_colab_v1 as entry
        output, marker = self.root / "stopped_cloud", self.root / "orphan_pid"
        child = ("import os,signal,time; from pathlib import Path; "
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)")
        parent = ("import subprocess,sys,time,json; from pathlib import Path; "
                  f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                  f"marker=Path({str(marker)!r})\n"
                  "while not marker.exists(): time.sleep(.01)\n"
                  f"Path({str(output / 'result.json')!r}).write_text(json.dumps({{'status':'STOPPED_FIXED_ATTEMPT','failure':{{'type':'ArtificialStageExit','message':'retained'}}}}))\n")
        class Cloud:
            def directory(self, create=False): return "folder" if create else None
            def children(self, parent): return []
            def write_json(self, *args): pass
        actual_popen, launched = subprocess.Popen, []
        def launch(*args, **kwargs):
            process = actual_popen([sys.executable, "-c", parent], **kwargs)
            launched.append(process)
            return process
        try:
            with patch.object(entry, "Cloud", Cloud), patch.object(entry, "OUTPUT", output), \
                 patch.object(entry, "WORK", self.root / "stopped_cache"), patch("builtins.print"), \
                 patch.object(entry, "_install_dependencies", return_value=os.environ.copy()), \
                 patch.object(entry.subprocess, "Popen", launch):
                result = entry.run_colab(source_directory=runner.DEFAULT_CONFIG.parents[3])
            self.assertEqual(result["status"], "STOPPED_FIXED_ATTEMPT")
            self.assertEqual(result["failure"]["type"], "ArtificialStageExit")
            pid = int(marker.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] in ("Z", "X"):
                    break
                time.sleep(.02)
            else:
                self.fail("STOPPED result returned while artificial child remains running")
        finally:
            for process in launched:
                entry.stop_worker_group(process)

    def test_native_receipts_are_durable_and_worker_missing_terminal_is_failure(self):
        from runtime.wan.b_joint_time_relative_physical_v1 import Backend
        output, scratch = self.root / "durable", self.root / "scratch"
        backend = Backend(self.config, output, scratch, lambda *a: None, write_json)
        class Native:
            def replay(self, cpu, base, features, destination, event, dump):
                destination.mkdir(parents=True)
                dump(destination / "native_chunk_000.json", {"artificial": True, "returned": True})
                dump(destination / "quality.json", {"artificial": True})
                np.save(destination / "host_q0_difference.npy", np.zeros(2))
                return {"status": "ARTIFICIAL"}
            def close(self): return {"E41_seconds": {"n01": 0, "e02": 0}}
        backend.native = Native()
        backend.replay()
        self.assertTrue((output / "writer_native/native_chunk_000.json").is_file())
        self.assertTrue((output / "writer_native/quality.json").is_file())
        self.assertTrue((output / "writer_native/host_q0_difference.npy").is_file())
        self.assertEqual(backend.close()["native"], {"identity_encodes_this_stage": 0})
        class DeadWorker:
            returncode = 7
            def __init__(self, *a, **k): pass
            def poll(self): return 7
        with patch.object(runner.subprocess, "Popen", DeadWorker):
            with self.assertRaises(FileNotFoundError):
                runner.subprocess_worker("prepare", output, scratch)


if __name__ == "__main__": unittest.main()
