from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import build_local_joint_posterior_bridge_v1_notebook as builder


pytestmark = pytest.mark.quick


def code(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def test_notebook_is_direct_fixed_bridge_and_portable():
    import nbformat

    notebook = json.loads((builder.ROOT / "notebooks" / (builder.NAME + "_colab.ipynb")).read_text())
    assert notebook == builder.build_notebook()
    nbformat.validate(notebook)
    assert code(notebook, 0).splitlines() == [
        "from google.colab import drive", "drive.mount('/content/drive')"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    node = next(node for node in ast.parse(code(notebook, 2)).body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "FIXED_CONFIG"
                        for target in node.targets))
    config = ast.literal_eval(node.value)
    assert config == json.loads((builder.ROOT / builder.CONFIG_PATH).read_text())
    assert list(config["inputs"]) == ["bridge_run", "base_terminal", "base_float_rgb"]
    assert config["carrier"]["lambda_residual"] == 1
    assert config["carrier"]["numerical_radius"] == .99998
    source = "\n".join(code(notebook, index) for index in range(len(notebook["cells"])))
    for forbidden in (
        "B64", "base64", "hashlib", "manifest", "git rev-parse", "force_remount",
        "strict", "override", "cvxpy", "clarabel", "NEEDS_SOLVER", "BALL_PROBE",
        "protected_run", "local_joint_ball_probe_v1", "PAUSED_FOR_SESSION_RESTART",
    ):
        assert forbidden not in source
    assert "1 VAE load + 1 ordinary decode (46 chunks)" in source
    assert "0 encode, VJP" in source
    assert "decoder-output clamped float RGB in [0,1]" in source
    assert "full post-clamp difference range" in source
    assert "[-0.25,0.25] applies only to display" in source
    assert "result['status'] != 'COMPLETE'" in code(notebook, 5)
    assert "views=result['views'], counts=result['counts']" in code(notebook, 5)
    assert "views=actual.get('views', fixed_slots['views'])" in code(notebook, 2)
    assert "finalize_interrupted(RUN_OUTPUT" in code(notebook, 2)
    assert code(notebook, 2).index("finalize_interrupted(RUN_OUTPUT") < code(notebook, 2).index("actual = json.loads")
    assert "AutoencoderKLWan" in code(notebook, 4)
    package = builder.portable_archive()
    assert package == (builder.ROOT / "notebooks" / (builder.NAME + "_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        assert "docs/local_joint_posterior_bridge_v1.md" in archive.namelist()
        for name in archive.namelist():
            assert archive.read(name) == (builder.ROOT / name).read_bytes()


def test_portable_archive_real_imports_from_isolated_extraction(tmp_path):
    archive_path = builder.ROOT / "notebooks" / (builder.NAME + "_portable_source.zip")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    script = r'''
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from main.tube_state import local_joint_posterior_bridge_v1 as method
from runtime.wan import local_joint_posterior_bridge_v1 as runtime
from experiments.wan_state_clock import local_joint_posterior_bridge_v1_run as runner
from runtime.wan.local_joint_state_payload_provider_v1 import LatentSupport
import torch
shape = (1, 1, 3, 4, 4)
support = LatentSupport((1, 3), ((0, 1, 0, 1), (0, 1, 3, 4),
                                  (3, 4, 0, 1), (3, 4, 3, 4)))
base = torch.zeros(shape, dtype=torch.float32)
raw = torch.ones(shape, dtype=torch.float32)
residual = torch.zeros(shape, dtype=torch.float32)
terminal, report = method.construct_candidate(base, raw, residual, support)
assert report['status'] == 'FIXED_POSTERIOR_BRIDGE_CANDIDATE'
assert report['fp32']['BASE_radius'] <= 1
assert runtime.EXPECTED['encode'] == runtime.EXPECTED['decoder_vjp'] == 0
assert callable(runner.main)
print('isolated portable imports and lazy method path: PASS')
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(extracted)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
