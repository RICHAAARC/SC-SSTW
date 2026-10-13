from __future__ import annotations

import ast
import io
import json
import zipfile

import pytest

from scripts import build_local_joint_ball_probe_v1_notebook as builder

pytestmark = pytest.mark.quick


def code(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def test_notebook_is_direct_fixed_probe_and_portable():
    import nbformat

    notebook = json.loads((builder.ROOT / "notebooks" / (builder.NAME + "_colab.ipynb")).read_text())
    assert notebook == builder.build_notebook()
    nbformat.validate(notebook)
    assert code(notebook, 0).splitlines() == ["from google.colab import drive", "drive.mount('/content/drive')"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    node = next(node for node in ast.parse(code(notebook, 2)).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "FIXED_CONFIG" for target in node.targets))
    config = ast.literal_eval(node.value)
    assert config == json.loads((builder.ROOT / builder.CONFIG_PATH).read_text())
    assert list(config["inputs"]) == ["protected_run"] and config["carrier"]["cap"] == 1
    source = "\n".join(code(notebook, index) for index in range(len(notebook["cells"])))
    for forbidden in ("B64", "base64", "hashlib", "git rev-parse", "force_remount",
                      "PAUSED_FOR_SESSION_RESTART", "COMPLETE_NO_CANDIDATE", "direction_unit.pt"):
        assert forbidden not in source
    assert "1 ordinary decode (46 chunks)" in source
    assert "result['status'] != 'COMPLETE'" in code(notebook, 5)
    assert "views=result['views'], counts=result['counts']" in code(notebook, 5)
    assert "views=actual.get('views', fixed_slots['views'])" in code(notebook, 2)
    assert "finalize_interrupted(RUN_OUTPUT" in code(notebook, 2)
    assert code(notebook, 2).index("finalize_interrupted(RUN_OUTPUT") < code(notebook, 2).index("actual = json.loads")
    package = builder.portable_archive()
    assert package == (builder.ROOT / "notebooks" / (builder.NAME + "_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (builder.ROOT / name).read_bytes()
