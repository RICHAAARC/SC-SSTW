import ast
import io
import json
import subprocess
import sys
import zipfile
import pytest
from scripts import build_local_joint_same_point_dual_v1_notebook as builder

pytestmark=pytest.mark.quick


def test_real_notebook_builder_and_editable_archive(tmp_path):
    import nbformat
    notebook=json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())
    assert notebook==builder.build_notebook();nbformat.validate(notebook)
    assert "".join(notebook["cells"][0]["source"]).splitlines()==["from google.colab import drive","drive.mount('/content/drive')"]
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]));assert cell["outputs"]==[] and cell["execution_count"] is None
    code="\n".join("".join(c["source"]) for c in notebook["cells"])
    for term in ("B64","base64","hashlib","git rev-parse","force_remount","override"):
        assert term not in code
    node=next(n for n in ast.parse("".join(notebook["cells"][2]["source"])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FIXED_CONFIG" for t in n.targets))
    config=ast.literal_eval(node.value)
    assert config==json.loads((builder.ROOT/builder.CONFIG_PATH).read_text())
    assert config["inputs"]["sum_run"].endswith("resumable-gradient-sum-v1/run")
    assert config["arms"]==["NEXT_MEAN","NEXT_ALLBIT"]
    assert "resumable-same-point-dual-v1" in code
    package=builder.portable_archive()
    assert package==(builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    source=tmp_path/"source"
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert len(archive.namelist())==len(set(archive.namelist()))
        for name in archive.namelist():assert archive.read(name)==(builder.ROOT/name).read_bytes()
        archive.extractall(source)
    assert not (source/".git").exists()
    proc=subprocess.run([sys.executable,"-m","experiments.wan_state_clock.local_joint_same_point_dual_v1_run","--help"],cwd=source,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr
    assert "--config" in proc.stdout and "--output" in proc.stdout
