import ast
import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import urllib.request
import zipfile
from pathlib import Path

from experiments.paper_results_v1.build_confirmation_v1_colab import OUTPUT, build_notebook


class ConfirmationNotebookTests(unittest.TestCase):
    def test_notebook_is_fixed_confirmation_companion_based_and_clean(self):
        notebook = build_notebook()
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(code[0], "from google.colab import drive\ndrive.mount('/content/drive')\n")
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        for source in code: ast.parse(source)
        self.assertTrue(all(cell.get("outputs", []) == [] and cell.get("execution_count") is None
                            for cell in notebook["cells"] if cell["cell_type"] == "code"))
        joined = "\n".join(code)
        self.assertIn('CONFIRMATION_CASES = ("confirm_01"', joined)
        self.assertIn('PILOT_CASES = ()', joined)
        self.assertIn('"main_receiver_rows": 2880', joined)
        self.assertIn("ACTIVE_RUN_POINTER", joined)
        self.assertIn("SKIPPED_DURABLE_COMPLETE", joined)
        self.assertIn('"CENTERED_U"', joined)
        self.assertIn("confirmation_v1_cli", joined)
        self.assertIn('SOURCE_REF = "dev/paper-results-v1"', joined)
        self.assertNotIn("PORTABLE_B64", joined)
        self.assertNotIn("expected_digest", joined)
        self.assertNotIn("force_remount", joined)
        companion = Path(__file__).resolve().parents[1] / "notebooks/paper_results_v1_companion.zip"
        with zipfile.ZipFile(companion) as archive:
            self.assertIn("experiments/paper_results_v1/confirmation_v1.adopted.json", archive.namelist())
            self.assertIn("experiments/paper_results_v1/confirmation_v1.py", archive.namelist())
            self.assertIn("experiments/paper_results_v1/confirmation_v1_cli.py", archive.namelist())

    def test_custom_cells_schedule_only_eight_confirmation_cases(self):
        notebook = build_notebook()
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).resolve().parents[1] / "experiments/paper_results_v1/confirmation_v1.adopted.json").read_text())
            logged_calls = []
            def logged(command, stage, **_kwargs): logged_calls.append((stage, list(command))); return 1
            ns = {
                "json": json, "Path": Path, "PYTHON": "python", "PORTABLE_ROOT": Path(__file__).resolve().parents[1],
                "OUTPUT_ROOT": root, "RUN_OUTPUT": root / "generation", "EFFECTIVE_CONFIG": root / "effective.json",
                "WAN_SNAPSHOT": root / "wan",
                "FRAMEWISE_SNAPSHOT": root / "framewise", "VS_ROOT": root / "vs", "VS_CARD": root / "card",
                "VS_WEIGHT": root / "vs.pth", "RIVA_ROOT": root / "riva", "RIVA_WEIGHT": root / "riva.pt",
                "portable_env": {}, "atomic_json": lambda path, value: Path(path).write_text(json.dumps(value)),
                "logged": logged, "CONFIRMATION_CASES": tuple(f"confirm_{i:02d}" for i in range(1, 9)),
                "baseline_pythons": {"videoseal": "vs-python", "rivagan": "rv-python"},
                "record_stage": lambda *_a, **_k: None, "LOG_PATH": root / "log", "STAGE_RECEIPTS": root / "receipts",
                "FIXED_ATTEMPT_SCOPE": {}, "RUN_ID": "stub",
                "time": time, "EVENT_LOG_TARGETS": {},
            }
            helpers = [n for n in ast.parse(cells["portable-setup"]).body if isinstance(n,ast.FunctionDef)
                       and n.name in ("read_event_log","write_event_log")]
            exec(compile(ast.Module(body=helpers,type_ignores=[]),"log-helpers","exec"),ns)
            # Source config is read from PORTABLE_ROOT; the local config remains unchanged.
            exec(compile(cells["confirmation-plans"], "confirmation-plans", "exec"), ns)
            self.assertEqual(len(logged_calls), 4)
            logged_calls.clear()
            exec(compile(cells["fixed-confirmation-execution"], "fixed-execution", "exec"), ns)
            attempts = json.loads((root / "phase_attempts.json").read_text())
            self.assertEqual(len(attempts), 123)
            self.assertEqual({row["case_id"] for row in attempts if row["case_id"]}, set(ns["CONFIRMATION_CASES"]))
            self.assertFalse(any("pilot_" in str(row) for row in attempts))
            generation_phases = {phase: {"cases": {case: {"status": "COMPLETE"} for case in ns["CONFIRMATION_CASES"]}}
                                 for phase in ("generate", "decode", "framewise", "codec")}
            ns["RUN_OUTPUT"].mkdir(parents=True, exist_ok=True)
            (ns["RUN_OUTPUT"] / "run_state.json").write_text(json.dumps({"phases": generation_phases}))
            attack_phases = {phase: {"cases": {case: {"status": "COMPLETE"} for case in ns["CONFIRMATION_CASES"]}}
                             for phase in ns["ATTACK_PHASES"]}
            ns["ATTACK_OUTPUT"].mkdir(parents=True, exist_ok=True)
            (ns["ATTACK_OUTPUT"] / "attack_run_state.json").write_text(json.dumps({"phases": attack_phases}))
            confirm_phases = {phase: {case: {"status": "COMPLETE"} for case in ns["CONFIRMATION_CASES"]}
                              for phase in ("prepare", "read")}
            ns["CONFIRM_OUTPUT"].mkdir(parents=True, exist_ok=True)
            (ns["CONFIRM_OUTPUT"] / "confirmation_state.json").write_text(json.dumps({"phases": confirm_phases}))
            logged_calls.clear()
            exec(compile(cells["fixed-confirmation-execution"], "fixed-execution-rerun", "exec"), ns)
            self.assertEqual([stage for stage, _command in logged_calls], [
                "None:generation:preflight", "None:attack:evaluate", "None:confirmation:report",
            ])
            # Restart enters partial/running stages, retaining the same run directory.
            generation_phases["framewise"]["cases"]["confirm_01"]["status"] = "RUNNING"
            attack_phases["receiver-clock"]["cases"]["confirm_01"]["status"] = "PARTIAL"
            confirm_phases["read"]["confirm_01"]["status"] = "INTERRUPTED"
            (ns["RUN_OUTPUT"] / "run_state.json").write_text(json.dumps({"phases":generation_phases}))
            (ns["ATTACK_OUTPUT"] / "attack_run_state.json").write_text(json.dumps({"phases":attack_phases}))
            (ns["CONFIRM_OUTPUT"] / "confirmation_state.json").write_text(json.dumps({"phases":confirm_phases}))
            logged_calls.clear()
            exec(compile(cells["fixed-confirmation-execution"], "partial-rerun", "exec"), ns)
            self.assertEqual([stage for stage, _ in logged_calls], [
                "None:generation:preflight", "confirm_01:generation:framewise", "confirm_01:attack:receiver-clock",
                "confirm_01:confirmation:read", "None:attack:evaluate", "None:confirmation:report",
            ])
            (ns["ATTACK_OUTPUT"] / "evaluation_report.json").write_text(json.dumps({
                "baseline_rows": [
                    {"case_id": "pilot_01", "status": "FAILED"},
                    {"case_id": "confirm_01", "status": "EVALUATED"},
                ],
                "quality_rows": [
                    {"case_id": "pilot_01", "status": "FAILED"},
                    {"case_id": "confirm_01", "status": "EVALUATED"},
                ],
            }))
            (ns["CONFIRM_OUTPUT"] / "evaluation_report.json").write_text(json.dumps({
                "status": "COMPLETE_WITH_RETAINED_ITEM_STATUSES", "receiver_status_counts": {"EVALUATED": 1},
            }))
            exec(compile(cells["handoff-summary"], "handoff", "exec"), ns)
            handoff = json.loads((root / "handoff_summary.json").read_text())
            self.assertEqual(handoff["execution_scope"], list(ns["CONFIRMATION_CASES"]))
            self.assertEqual(handoff["pilots"]["status"], "NOT_EXECUTED_BY_NOTEBOOK")
            self.assertEqual(handoff["baseline_status_counts"], {"EVALUATED": 1})
            self.assertEqual(handoff["quality_status_counts"], {"EVALUATED": 1})

    def test_source_loader_refetches_when_old_cache_has_only_real_cli(self):
        notebook = build_notebook()
        setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == "portable-setup"))
        tree = ast.parse(setup)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_companion_source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portable = root / "old-source"
            legacy = portable / "experiments/paper_results_v1/real_cli.py"
            legacy.parent.mkdir(parents=True); legacy.write_text("# old cache\n")
            output = root / "output"; output.mkdir()
            companion = Path(__file__).resolve().parents[1] / "notebooks/paper_results_v1_companion.zip"
            fetches = []
            def fetch(_url, target):
                fetches.append(str(target)); shutil.copyfile(companion, target); return str(target), None
            def atomic_json(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8"); return str(path)
            namespace = {
                "Path": Path, "PORTABLE_ROOT": portable, "OUTPUT_ROOT": output,
                "COMPANION_ZIP_URL": "https://example.invalid/companion.zip",
                "SOURCE_REPOSITORY": "https://example.invalid/repo", "SOURCE_REF": "editable-ref",
                "urllib": type("U", (), {"request": type("R", (), {"urlretrieve": staticmethod(fetch)})}),
                "zipfile": zipfile, "atomic_json": atomic_json,
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "source-loader", "exec"), namespace)
            receipt = namespace["load_companion_source"]()
            self.assertEqual(len(fetches), 1)
            self.assertEqual(receipt["status"], "DOWNLOADED_AND_EXTRACTED_AVAILABLE_SOURCE")
            self.assertIn("experiments/paper_results_v1/confirmation_v1_cli.py", receipt["missing_before_fetch"])
            self.assertTrue((portable / "experiments/paper_results_v1/confirmation_v1_cli.py").is_file())
            namespace["load_companion_source"]()
            self.assertEqual(len(fetches), 2) # a complete old cache must still refresh
            def unsafe_fetch(_url,target):
                with zipfile.ZipFile(target,"w") as archive: archive.writestr("../escaped.py","bad")
            namespace["urllib"].request.urlretrieve = staticmethod(unsafe_fetch)
            with self.assertRaisesRegex(RuntimeError,"unsafe companion"):
                namespace["load_companion_source"]()
            self.assertFalse((root/"escaped.py").exists())

    def test_pending_quality_repairs_main_lpips_once_and_keeps_failure_evidence_path(self):
        notebook = build_notebook()
        source = "".join(next(c["source"] for c in notebook["cells"]
                              if c["id"] == "fixed-confirmation-execution"))
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                     and n.name in ("prepare_pending_lpips", "invoke_owned")]
        for metric_status, phase_status, probe_results, expected_stages in (
            ("FAILED", "PARTIAL", [1, 0, 0], ["QUALITY_LPIPS_IMPORT_PROBE", "QUALITY_LPIPS_IMPORT_REPAIR", "QUALITY_LPIPS_IMPORT_REPROBE"]),
            ("FAILED", "PARTIAL", [0], ["QUALITY_LPIPS_IMPORT_PROBE"]),
            ("FAILED", "PARTIAL", [1, 1, 1], ["QUALITY_LPIPS_IMPORT_PROBE", "QUALITY_LPIPS_IMPORT_REPAIR", "QUALITY_LPIPS_IMPORT_REPROBE"]),
            ("EVALUATED", "PARTIAL", [], []),
            ("EVALUATED", "COMPLETE", [], []),
        ):
            with self.subTest(metric=metric_status, phase=phase_status, returns=probe_results):
                calls, invocations, retained = [], [], []
                results = iter(probe_results)
                def logged(command, stage, **kwargs):
                    calls.append((list(command), stage, kwargs))
                    return next(results)
                ns = {
                    "quality_lpips_checked": False, "Path": Path, "PYTHON": "/main/python",
                    "PORTABLE_ROOT": Path("/source"), "portable_env": {}, "ATTACK_OUTPUT": Path("/attack"),
                    "read_state": lambda _path: {"quality_rows": [{"case_id": "confirm_01", "metrics": {"LPIPS": {"status": metric_status}}}]},
                    "phase_status": lambda *_args: phase_status, "logged": logged,
                    "retain_attempt": lambda *args: retained.append(args),
                    "invoke": lambda *args, **kwargs: invocations.append((args, kwargs)),
                }
                exec(compile(ast.Module(body=functions, type_ignores=[]), "quality-preparation", "exec"), ns)
                for _ in range(2):
                    ns["invoke_owned"](["quality"], "confirm_01", "attack", "quality")
                self.assertEqual([stage for _, stage, _ in calls], expected_stages)
                self.assertTrue(all(command[0] == "/main/python" for command, _, _ in calls))
                self.assertTrue(all(kwargs["check"] is False for _, _, kwargs in calls))
                if phase_status == "COMPLETE":
                    self.assertFalse(invocations)
                    self.assertEqual(len(retained), 2)
                else:
                    # Failed dependency repair still reaches the real metric path,
                    # which retains its import/model-load error in the saved row.
                    self.assertEqual(len(invocations), 2)
                installs = [cmd for cmd, _, _ in calls if "install" in cmd]
                self.assertTrue(all(cmd == ["/main/python", "-m", "pip", "install", "lpips"] for cmd in installs))

    def test_active_pointer_recovers_same_run_and_rejects_unsafe_directory_name(self):
        notebook=build_notebook()
        source="".join(next(c["source"] for c in notebook["cells"] if c["id"]=="portable-setup"))
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="saved_run_id")
        ns={"json":json,"Path":Path}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"pointer","exec"),ns)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); pointer=root/"active_confirmation_run.json"
            run=root/"fixed-task"; (run/"run_state").mkdir(parents=True)
            pointer.write_text(json.dumps({"run_id":"fixed-task"}))
            self.assertEqual(ns["saved_run_id"](root,pointer),"fixed-task")
            pointer.write_text("{")
            self.assertEqual(ns["saved_run_id"](root,pointer),"fixed-task")
            pointer.write_text(json.dumps({"run_id":".."}))
            self.assertEqual(ns["saved_run_id"](root,pointer),"fixed-task")
            for value in (None, []):
                pointer.write_text(json.dumps(value))
                self.assertEqual(ns["saved_run_id"](root,pointer),"fixed-task")

    def test_corrupt_auxiliary_logs_preserved_without_blocking_run(self):
        source="".join(next(c["source"] for c in build_notebook()["cells"] if c["id"]=="portable-setup"))
        helpers=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ("read_event_log","write_event_log")]
        ns={"json":json,"Path":Path,"time":time,"EVENT_LOG_TARGETS":{},
            "atomic_json":lambda p,v:Path(p).write_text(json.dumps(v))}
        exec(compile(ast.Module(body=helpers,type_ignores=[]),"log-helpers","exec"),ns)
        with tempfile.TemporaryDirectory() as directory:
            for name in ("stage_receipts.json","phase_attempts.json"):
                path=Path(directory)/name
                path.with_name(name+".previous").write_text('[{"status":"COMPLETE"}]')
                for raw in ("{","null","{}"):
                    path.write_text(raw)
                    rows=ns["read_event_log"](path)
                    self.assertEqual(rows,[{"status":"COMPLETE"}])
                    self.assertTrue(any(p.read_text()==raw for p in Path(directory).glob(name+".unreadable.*")))
                    rows.append({"status":"CONTINUED"})
                    ns["write_event_log"](path,rows)
                    self.assertEqual(json.loads(path.read_text())[-1]["status"],"CONTINUED")

    def test_reused_kernel_drops_old_experiments_package_paths(self):
        source="".join(next(c["source"] for c in build_notebook()["cells"] if c["id"]=="identity-and-init"))
        nodes=[]
        for node in ast.parse(source).body:
            nodes.append(node)
            if isinstance(node,ast.ImportFrom) and node.module=="experiments.paper_results_v1.resume_state": break
        with tempfile.TemporaryDirectory() as directory:
            old={}
            for name in ("experiments","experiments.paper_results_v1","experiments.paper_results_v1.colab_orchestration"):
                mod=types.ModuleType(name); mod.__path__=[str(Path(directory)/name.replace(".","/"))]; old[name]=mod
            ns={"PORTABLE_ROOT":Path(__file__).resolve().parents[1]}
            prior_path=list(sys.path)
            try:
                with mock.patch.dict(sys.modules,old):
                    exec(compile(ast.Module(body=nodes,type_ignores=[]),"source-package-refresh","exec"),ns)
                    loaded=sys.modules["experiments.paper_results_v1.resume_state"]
                    self.assertTrue(str(loaded.__file__).startswith(str(ns["PORTABLE_ROOT"])))
                    self.assertNotIn("experiments.paper_results_v1.colab_orchestration",sys.modules)
            finally:
                sys.path[:]=prior_path


if __name__ == "__main__":
    unittest.main()
