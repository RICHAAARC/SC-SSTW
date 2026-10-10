import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments.paper_results_v1.build_attack_recovery_colab import OUTPUT, build_notebook
from experiments.paper_results_v1.companion_source import OUTPUT as COMPANION
from experiments.paper_results_v1.attack_eval import AttackRunStore, read_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/paper_results_v1/attack_eval.adopted.json"


class RecoveryNotebookTests(unittest.TestCase):
    def test_generated_notebook_is_editable_fixed_scope_and_clean(self):
        build_notebook()
        notebook = json.loads(OUTPUT.read_text())
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(code[0], "from google.colab import drive\ndrive.mount('/content/drive')\n")
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        for source in code:
            ast.parse(source)
        joined = "\n".join(code)
        self.assertIn('SOURCE_REF = "dev/paper-results-v1"', joined)
        self.assertIn("20261010T031617441784Z-2536172d", joined)
        self.assertIn("RECOVERY_POINTER", joined)
        self.assertIn("receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.is_file() else []", joined)
        self.assertIn('PILOTS = ("pilot_01", "pilot_02")', joined)
        self.assertIn('"index-saved-media", "receiver-clock", "receiver-read"', joined)
        self.assertNotIn("baseline-embed-videoseal", joined)
        self.assertNotIn("baseline-codec", joined)
        self.assertNotIn("PORTABLE_B64", joined)
        self.assertNotIn("expected_digest", joined)
        self.assertTrue(all(not cell.get("outputs") and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code"))

    def test_companion_contains_recovery_entrypoints(self):
        receipt = build_notebook()
        import zipfile
        with zipfile.ZipFile(receipt["companion"]["path"]) as archive:
            names = set(archive.namelist())
        self.assertIn("experiments/paper_results_v1/attack_recovery.py", names)
        self.assertIn("experiments/paper_results_v1/attack_recovery_cli.py", names)

    def test_generated_cells_run_in_order_with_no_model_boundary_stub(self):
        build_notebook()
        notebook = json.loads(OUTPUT.read_text())
        cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory); content = sandbox / "content"; drive_root = sandbox / "drive/MyDrive/Video-WM"
            content.mkdir(); drive_root.mkdir(parents=True)
            source = drive_root / "Paper-Results-V1-Temporal-Attack/20261010T031617441784Z-2536172d"
            state = AttackRunStore(source / "run_state", read_json(CONFIG), create=True)
            state.data["evaluation_report"] = {
                "path": "/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack/20261010T031617441784Z-2536172d/run_state/evaluation_report.json",
                "bytes": 1,
            }
            state.save()
            google = types.ModuleType("google"); colab = types.ModuleType("google.colab"); drive = types.ModuleType("google.colab.drive")
            drive.mount = lambda path: None; colab.drive = drive; google.colab = colab
            modules = {"google": google, "google.colab": colab, "google.colab.drive": drive}
            hf = types.ModuleType("huggingface_hub")
            def snapshot_download(*, repo_id, revision, local_dir, allow_patterns):
                target = Path(local_dir); target.mkdir(parents=True, exist_ok=True)
                if repo_id.startswith("Wan-AI/"):
                    (target / "vae").mkdir(); (target / "model_index.json").write_text("{}")
                else:
                    (target / "config.json").write_text("{}"); (target / "diffusion_pytorch_model.safetensors").write_bytes(b"stub")
                return str(target)
            hf.snapshot_download = snapshot_download; modules["huggingface_hub"] = hf
            def adapted(source_code):
                return source_code.replace("/content/drive/MyDrive/Video-WM", str(drive_root)).replace("/content", str(content))
            original_urlretrieve = __import__("urllib.request").request.urlretrieve
            import urllib.request
            def source_download(url, target):
                target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
                if "companion" in str(url): shutil.copyfile(COMPANION, target)
                else: target.write_bytes(b"stub")
                return str(target), None
            urllib.request.urlretrieve = source_download
            ns = {"__name__": "__main__"}
            old_cwd = Path.cwd(); os.chdir(sandbox)
            try:
                with mock.patch.dict(sys.modules, modules):
                    exec(compile(adapted(cells[0]), "drive", "exec"), ns)
                    exec(compile(adapted(cells[1]), "setup", "exec"), ns)
                    commands = []
                    def fake_logged(command, stage, **kwargs):
                        commands.append((stage, list(command)))
                        if command[:2] == ["git", "clone"]:
                            target = Path(command[-1]); target.mkdir(parents=True, exist_ok=True)
                            if "videoseal" in str(target):
                                (target / "videoseal/cards").mkdir(parents=True); (target / "videoseal/cards/videoseal_1.0.yaml").write_text("args: {nbits: 256}")
                            else: (target / "rivagan").mkdir()
                        if "venv" in command and "--without-pip" in command:
                            python = Path(command[-1]) / "bin/python"; python.parent.mkdir(parents=True, exist_ok=True); python.write_text("stub")
                        if "-c" in command: compile(command[command.index("-c") + 1], "generated-probe", "exec")
                        return 0
                    ns["logged"] = fake_logged
                    with mock.patch.object(subprocess, "check_output", return_value=""):
                        exec(compile(adapted(cells[2]), "environment", "exec"), ns)
                    exec(compile(adapted(cells[3]), "execute", "exec"), ns)
                    exec(compile(adapted(cells[4]), "handoff", "exec"), ns)
                self.assertEqual(len([row for row in commands if ":index-saved-media" in row[0]]), 2)
                self.assertEqual(len([row for row in commands if ":receiver-clock" in row[0]]), 2)
                self.assertTrue((ns["OUTPUT_ROOT"] / "handoff_summary.json").is_file())
                self.assertNotIn("torch", sys.modules)
            finally:
                os.chdir(old_cwd); urllib.request.urlretrieve = original_urlretrieve


if __name__ == "__main__":
    unittest.main()
