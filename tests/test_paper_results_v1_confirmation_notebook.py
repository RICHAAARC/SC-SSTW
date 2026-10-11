import ast
import json
import shutil
import tempfile
import unittest
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
            }
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
            self.assertEqual(receipt["status"], "DOWNLOADED_AND_EXTRACTED_MISSING_REQUIRED_SOURCE")
            self.assertIn("experiments/paper_results_v1/confirmation_v1_cli.py", receipt["missing_before_fetch"])
            self.assertTrue((portable / "experiments/paper_results_v1/confirmation_v1_cli.py").is_file())


if __name__ == "__main__":
    unittest.main()
