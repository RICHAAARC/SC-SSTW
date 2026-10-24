"""Delivery/import checks only; never start the model or notebook runtime."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import build_b_terminal_three_group_v1_notebook as builder

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_notebook_actual_entry_and_mount():
    nb = json.loads((ROOT/'notebooks'/f'{builder.STEM}.ipynb').read_text())
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert code[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for source in code:
        ast.parse(source)
    assert all(c.get('outputs', []) == [] and c.get('execution_count') is None for c in nb['cells'])
    text = '\n'.join(code)
    assert 'experiments.wan_state_clock.b_terminal_three_group_v1_run' in text
    assert 'SC-SSTW-b-terminal-three-group-v1-' in text
    assert 'b_local_identity_dct_v1.py' in text and 'b_cdc32_bundle_v1.py' in text
    assert 'if not EXISTING_ARTIFACTS:' in text
    assert 'SOURCE_SHA' not in text and 'strict' not in text and 'base64' not in text


def test_editable_no_git_source_cli_and_new_imports(tmp_path):
    archive = ROOT/'notebooks'/f'{builder.STEM}_user_bundle.zip'
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(builder.source_files()) | {f'notebooks/{builder.STEM}.ipynb'}
        assert not any('.git' in Path(n).parts for n in z.namelist())
        z.extractall(tmp_path)
    entry = tmp_path/builder.ENTRY
    result = subprocess.run([sys.executable, '-I', str(entry), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert '--background' in result.stdout
    code = "import sys;sys.path.insert(0,sys.argv[1]);from pathlib import Path;import runtime.wan.b_terminal_three_group_backend_v1 as backend;import main.tube_state.b_local_identity_dct_v1 as b1;import main.tube_state.b_m3_time_path_v1 as b2;import runtime.wan.b_cdc32_bundle_v1 as b3;assert all(Path(x.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()) for x in (backend,b1,b2,b3));print('new modules import from editable source')"
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert 'new modules import' in result.stdout
