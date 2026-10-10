"""Ordinary ZIP source loading reaches each real CLI; model boundary is stubbed."""
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import pytest

from scripts import build_local_joint_state_payload_v1_notebook as original
from scripts import build_local_joint_terminal_bridge_v1_notebook as bridge

pytestmark = pytest.mark.quick


def test_optional_saved_file_metadata_read_failure_is_not_admission(tmp_path):
    from runtime.wan import fixed_rgb_media, local_joint_state_payload_experiment_v1
    missing = tmp_path/"metadata-no-longer-readable"
    assert fixed_rgb_media.file_sha256(missing) is None
    assert local_joint_state_payload_experiment_v1._sha(missing) is None


@pytest.mark.parametrize("kind", ["original", "bridge"])
@pytest.mark.parametrize("broken_import", [False, True])
def test_downloaded_editable_source_reaches_cli_without_identity_admission(tmp_path, monkeypatch, kind, broken_import):
    builder = original if kind == "original" else bridge
    package = builder.portable_archive()
    requests = []
    def download(url):
        requests.append(url)
        return io.BytesIO(package)
    monkeypatch.setattr(urllib.request, "urlopen", download)
    workspace = tmp_path/"source"
    workspace.mkdir()
    # Even an unrelated/unusable Git metadata entry must not be consulted.
    (workspace/".git").write_text("unrelated local metadata")
    notes = []
    def write(path, value):
        notes.append(value)
    namespace = dict(WORKSPACE=workspace, OUTPUT=tmp_path, SOURCE_REF="editable-source-ref",
                     write_json=write, record_failure=lambda stage, exc: pytest.fail(str(exc)), sys=sys)
    nb = builder.build_notebook()
    source_index = 4 if kind == "original" else 3
    exec(compile("".join(nb["cells"][source_index]["source"]), "actual-source-cell", "exec"), namespace)
    assert requests and "/editable-source-ref/notebooks/" in requests[0]
    assert not list(workspace.rglob("*manifest*"))
    assert notes == [dict(url=requests[0], workspace=str(workspace))]
    module = ("experiments.wan_state_clock.local_joint_state_payload_v1_run" if kind == "original"
              else "experiments.wan_state_clock.local_joint_terminal_bridge_v1_run")
    source = workspace/Path(*module.split('.')).with_suffix('.py')
    source.write_text(("raise ImportError('sentinel actual import failure')\n" if broken_import else "# User-edited source.\n")
                      + source.read_text().replace("from __future__ import annotations", ""))
    config = workspace/builder.CONFIG_PATH
    config.write_text(json.dumps(json.loads(config.read_text()), indent=4))
    marker = tmp_path/"actual_cli_boundary.json"
    harness = """
import importlib, json, pathlib, sys
entry = importlib.import_module(sys.argv[1])
def no_model_run(config, output, **kwargs):
    pathlib.Path(sys.argv[4]).write_text(json.dumps(dict(config=config, output=str(output), kwargs=kwargs)))
    return dict(status='COMPLETE', counts={})
entry.run = no_model_run
raise SystemExit(entry.main(['--config',sys.argv[2],'--output',sys.argv[3]]))
"""
    child = subprocess.run([sys.executable, "-c", harness, module, str(config), str(tmp_path/"run"), str(marker)],
                           cwd=workspace, text=True, capture_output=True)
    if broken_import:
        assert child.returncode != 0 and "sentinel actual import failure" in child.stderr
        assert not marker.exists()
    else:
        assert child.returncode == 0, child.stderr
        called = json.loads(marker.read_text())
        assert called["config"]["carrier"]["rho"] == .5
        assert called["config"]["carrier"]["cap"] == 1.
        assert not called["kwargs"].get("preflight_only", False)
        assert not any(word in json.dumps(called["kwargs"]) for word in ("sha256", "manifest", "git_commit"))
