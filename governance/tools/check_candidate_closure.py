"""Validate the current GROW chain in a real standalone directory without .git."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
OLD_PATHS=("main/tube_state/state_clock.py","main/tube_state/payload_codec.py",
    "runtime/wan/integrated_core.py","runtime/wan/payload_control.py",
    "experiments/wan_state_clock/run.py","experiments/wan_state_clock/integrated_payload_run.py",
    "notebooks/wan_state_clock_colab.ipynb","notebooks/integrated_payload_v1_colab.ipynb")

def main():
    if any((ROOT/name).exists() for name in OLD_PATHS):
        raise SystemExit("unreleased historical entry remains in current tree")
    with tempfile.TemporaryDirectory(prefix="sc-sstw-detached-") as temporary:
        root=Path(temporary)/"release"
        shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns(".git","__pycache__","*.pyc",".pytest_cache","outputs","local_reports"))
        assert not (root/".git").exists()
        # -I suppresses caller PYTHONPATH/user paths. Standard installed packages remain.
        probe=f"""import sys,json,pathlib
sys.path.insert(0,{str(root)!r})
import main.tube_state.grow_video_reference as m
import runtime.wan.grow_video_reference as r
import experiments.wan_state_clock.grow_video_reference_run as e
assert all(pathlib.Path(x.__file__).is_relative_to({str(root)!r}) for x in (m,r,e))
s=e.Store(pathlib.Path({str(Path(temporary)/'initialized')!r}),create=True)
assert s.data['source_sha'] is None
assert s.data['source_provenance']['manifest_status']=='MATCH'
assert [len(s.data[k]) for k in ('generation','videos','reads','evaluations')]==[3,3,24,48]
print('no-Git source initialization and reference imports PASS')
"""
        subprocess.run([sys.executable,"-I","-c",probe],cwd=temporary,check=True)
        # Current full-flow, voting/blindness, native-step, save/read and failure tests.
        test_code=f"""import os,sys,pytest
os.chdir({str(root)!r});sys.path.insert(0,{str(root)!r})
raise SystemExit(pytest.main(['-q','tests']))
"""
        subprocess.run([sys.executable,"-I","-c",test_code],cwd=temporary,check=True)
        # Missing own adapter must fail, never resolve a sibling or parent checkout.
        adapter=root/"runtime/wan/grow_video_reference.py";adapter.rename(adapter.with_suffix(".removed"))
        bad=f"import sys;sys.path.insert(0,{str(root)!r});import experiments.wan_state_clock.grow_video_reference_run"
        result=subprocess.run([sys.executable,"-I","-c",bad],cwd=temporary,text=True,capture_output=True)
        if result.returncode==0 or "ImportError" not in result.stderr:
            raise SystemExit("missing current adapter unexpectedly resolved")
    print("current GROW closure PASS: detached no-.git full fixture suite and missing-adapter negative")

if __name__=="__main__":main()
