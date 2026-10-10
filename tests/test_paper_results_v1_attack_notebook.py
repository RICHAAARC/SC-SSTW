import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from experiments.paper_results_v1.build_attack_eval_colab import OUTPUT, build_notebook


class AttackNotebookTests(unittest.TestCase):
    def test_notebook_is_fixed_two_pilot_portable_and_clean(self):
        receipt = build_notebook()
        notebook = json.loads(OUTPUT.read_text())
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(code[0], "from google.colab import drive\ndrive.mount('/content/drive')\n")
        for source in code:
            ast.parse(source)
        self.assertTrue(all(cell.get("outputs", []) == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code"))
        joined = "\n".join(code)
        self.assertIn('PILOTS = ("pilot_01", "pilot_02")', joined)
        self.assertIn('confirmation": "NOT_EXECUTED_BY_NOTEBOOK"', joined)
        self.assertIn("portable_source.zip", joined)
        self.assertIn("record-failure", joined)
        self.assertIn("finally:", joined)
        self.assertNotIn("force_remount", joined)
        self.assertGreater(receipt["portable_files"], 10)

    def test_generated_cells_run_in_order_in_fresh_boundary_stub(self):
        build_notebook()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "boundary.py"
            script.write_text(textwrap.dedent(f'''\
                import json, os, subprocess, sys, types
                from pathlib import Path
                notebook = json.loads(Path({str(OUTPUT)!r}).read_text())
                cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
                sandbox = Path({str(root)!r}); content = sandbox / "content"; drive_root = sandbox / "drive/MyDrive/Video-WM"
                content.mkdir(); drive_root.mkdir(parents=True)
                google=types.ModuleType("google"); colab=types.ModuleType("google.colab"); drive=types.ModuleType("google.colab.drive")
                drive.mount=lambda path: None; colab.drive=drive; colab.userdata=types.SimpleNamespace(get=lambda key: None); google.colab=colab
                sys.modules.update({{"google":google,"google.colab":colab,"google.colab.drive":drive}})
                hf=types.ModuleType("huggingface_hub"); snapshots=[]
                def snapshot_download(*,repo_id,revision,local_dir,allow_patterns):
                    snapshots.append((repo_id,tuple(allow_patterns))); target=Path(local_dir); target.mkdir(parents=True,exist_ok=True)
                    if repo_id.startswith("Wan-AI/"): (target/"vae").mkdir(); (target/"model_index.json").write_text("{{}}",encoding="utf-8"); (target/"vae/config.json").write_text("{{}}",encoding="utf-8")
                    else: (target/"config.json").write_text("{{}}",encoding="utf-8"); (target/"diffusion_pytorch_model.safetensors").write_bytes(b"stub")
                    return str(target)
                hf.snapshot_download=snapshot_download; sys.modules["huggingface_hub"]=hf
                def adapted(source): return source.replace("/content/drive/MyDrive/Video-WM",str(drive_root)).replace("/content",str(content))
                ns={{"__name__":"__main__"}}; os.chdir(sandbox)
                exec(compile(adapted(cells[0]),"drive","exec"),ns)
                exec(compile(adapted(cells[1]),"setup","exec"),ns)
                original_popen=subprocess.Popen; commands=[]
                def urlretrieve(url,target): Path(target).parent.mkdir(parents=True,exist_ok=True); Path(target).write_bytes(b"stub"); return str(target),None
                ns["urllib"].request.urlretrieve=urlretrieve
                entry_attempts={{}}
                class FakeChild:
                    next_pid=91000
                    def __init__(self,command,*args,**kwargs):
                        self.command=list(command); self.pid=FakeChild.next_pid; FakeChild.next_pid+=1; self.done=False; self.returncode=0
                        commands.append(self.command)
                        if self.command[:2]==["git","clone"]:
                            target=Path(self.command[-1]); target.mkdir(parents=True,exist_ok=True)
                            if "videoseal" in str(target): (target/"videoseal/cards").mkdir(parents=True); (target/"videoseal/cards/videoseal_1.0.yaml").write_text("args: {{nbits: 256}}")
                            else: (target/"rivagan").mkdir(); (target/"rivagan/__init__.py").write_text("")
                        if "venv" in self.command and "--without-pip" in self.command:
                            python=Path(self.command[-1])/"bin/python"; python.parent.mkdir(parents=True,exist_ok=True); python.write_text("stub")
                        if "-c" in self.command:
                            code=self.command[self.command.index("-c")+1]; compile(code,"generated-probe","exec")
                            if "VIDEOSEAL_IMPORT_READY" in code:
                                entry_attempts["videoseal"]=entry_attempts.get("videoseal",0)+1; self.returncode=1 if entry_attempts["videoseal"]==1 else 0
                            elif "RIVAGAN_IMPORT_READY" in code:
                                entry_attempts["rivagan"]=entry_attempts.get("rivagan",0)+1; self.returncode=1
                            elif "ATTACK_QUALITY_IMPORT_READY" in code:
                                entry_attempts["quality"]=entry_attempts.get("quality",0)+1; self.returncode=1 if entry_attempts["quality"]==1 else 0
                    def wait(self,timeout=None): self.done=True; return self.returncode
                    def poll(self): return self.returncode if self.done else None
                original_check_output=subprocess.check_output; original_run=subprocess.run
                subprocess.check_output=lambda *a,**k: "" if k.get("text") else b""
                subprocess.run=lambda *a,**k: types.SimpleNamespace(returncode=1,stdout="",stderr="")
                subprocess.Popen=FakeChild
                try: exec(compile(adapted(cells[2]),"environment","exec"),ns)
                finally: subprocess.check_output=original_check_output; subprocess.run=original_run; subprocess.Popen=original_popen
                assert len(snapshots)==2 and "transformer/*" not in snapshots[0][1]
                assert sum("--without-pip" in command for command in commands)==2, ns["baseline_setup"]
                assert sum("--constraint" in command for command in commands)==2
                assert entry_attempts=={{"videoseal":2,"rivagan":2,"quality":2}}
                assert ns["baseline_setup"]["videoseal"]["entry_import_probe_status"]=="IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED"
                assert ns["baseline_setup"]["rivagan"]["entry_import_probe_status"]=="FAILED"
                phase_calls=[]
                class PhaseChild(FakeChild):
                    def __new__(cls,command,*args,**kwargs):
                        phase=command[command.index("--phase")+1] if "--phase" in command else None
                        if phase in ("record-failure","evaluate"): return original_popen(command,*args,**kwargs)
                        return super().__new__(cls)
                    def __init__(self,command,*args,**kwargs):
                        phase=command[command.index("--phase")+1] if "--phase" in command else None
                        FakeChild.__init__(self,command,*args,**kwargs); self.returncode=1; phase_calls.append((phase,list(command)))
                subprocess.Popen=PhaseChild
                exec(compile(adapted(cells[3]),"execute","exec"),ns)
                subprocess.Popen=original_popen; exec(compile(adapted(cells[4]),"handoff","exec"),ns)
                assert len(phase_calls)==20
                report=json.loads((ns["RUN_OUTPUT"]/"evaluation_report.json").read_text())
                assert len(report["receiver_rows"])==2400 and len(report["baseline_rows"])==300
                assert {{row["status"] for row in report["receiver_rows"] if row["case_id"]=="confirm_01"}}=={{"NOT_EXECUTED_BY_NOTEBOOK"}}
                assert "torch" not in sys.modules
                print("ATTACK_BOUNDARY_STUB_COMPLETE")
            '''), encoding="utf-8")
            environment = dict(os.environ); environment.pop("PYTHONPATH", None); environment.pop("PYTHONHOME", None)
            completed = subprocess.run(["/usr/bin/python3", "-I", str(script)], cwd=root, env=environment, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            self.assertIn("ATTACK_BOUNDARY_STUB_COMPLETE", completed.stdout)


if __name__ == "__main__":
    unittest.main()
