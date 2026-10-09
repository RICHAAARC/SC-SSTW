from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile

import pytest

from scripts import build_local_joint_state_payload_v1_notebook as builder


pytestmark = pytest.mark.quick
NOTEBOOK = builder.ROOT / "notebooks/local_joint_state_payload_v1_colab.ipynb"
COMPANION_PACKAGE = builder.ROOT / "notebooks/local_joint_state_payload_v1_portable_source.zip"


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict[str, object], index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def _install_colab_stub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    mounts: list[str] = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: mounts.append(path))
    colab.userdata = types.SimpleNamespace(get=lambda _name: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return mounts


def _execute_setup(notebook: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    namespace: dict[str, object] = {"__name__": "__notebook_fixture__"}
    mounts = _install_colab_stub(monkeypatch)
    exec(compile(_code(notebook, 0), "cell-0", "exec"), namespace)
    setup = _code(notebook, 2).replace(
        "/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1",
        str(tmp_path / "drive"),
    ).replace(
        "/content/Video-WM-Local-Joint-State-Payload-V1-",
        str(tmp_path / "workspace-"),
    )
    exec(compile(setup, "cell-2", "exec"), namespace)
    return namespace, mounts


def _dependency_receipt(namespace: dict[str, object]) -> dict[str, object]:
    return dict(
        python=sys.version,
        versions={
            "torch": "2.11.0",
            "diffusers": "0.39.0",
            "transformers": "4.57.6",
            "numpy": "2.1.3",
            "accelerate": "1.15.0",
            "safetensors": "0.8.0",
            "huggingface-hub": "0.36.2",
            "tokenizers": "0.22.2",
            "sentencepiece": "0.2.2",
            "ftfy": "6.3.1",
        },
        torch_cuda_runtime="13.0",
        cuda_available=True,
        device="FAKE GPU",
        total_device_memory=1,
        ffmpeg="/fake/ffmpeg",
        ffprobe="/fake/ffprobe",
    )


def _write_fake_run(namespace: dict[str, object]) -> None:
    run_output = namespace["RUN_OUTPUT"]
    run_output.mkdir(parents=True, exist_ok=True)
    namespace["write_json"](
        run_output / "result.json",
        dict(
            status="COMPLETE",
            stage="COMPLETE",
            execution=dict(attempted=True, completed=True),
            actual_model_calls=True,
            source_identity=dict(
                kind="unversioned_directory",
                git_commit=None,
                content_sha256=namespace["RUNNER_CONTENT_SHA256"],
            ),
            arms={"OFF": {"status": "COMPLETE"}, "JOINT": {"status": "COMPLETE"}},
        ),
    )
    namespace["write_json"](run_output / "raw_observation_manifest.json", {"entries": {}})


def _write_fake_posthoc(namespace: dict[str, object]) -> None:
    posthoc = namespace["POSTHOC_OUTPUT"]
    posthoc.mkdir(parents=True, exist_ok=True)
    namespace["write_json"](
        posthoc / "raw_observation_seal.json",
        dict(truth_loaded=False, entries={f"entry-{index}": {"status": "SEALED"} for index in range(12)}),
    )
    namespace["write_json"](
        posthoc / "posthoc_result.json",
        dict(
            status="COMPLETE",
            outcome_classification="FINITE_DESCRIPTIVE_RESULT",
            mp4_attribution="UNRESOLVED",
            scientific_pass=False,
        ),
    )


def _execute_environment_source_and_run(
    notebook: dict[str, object],
    namespace: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner_returncode: int = 0,
    fail_execution_receipt: bool = False,
) -> list[str]:
    stages: list[str] = []

    def fake_logged(_command, stage, **_kwargs):
        stages.append(stage)
        if stage == "DEPENDENCY_PROBE":
            namespace["write_json"](namespace["OUTPUT"] / "dependency_probe.json", _dependency_receipt(namespace))
            return 0
        if stage == "DEPENDENCY_REPORT":
            return 1  # historical pip-check conflicts are recorded, not a hard gate
        if stage == "REAL_FIXED_RUN":
            _write_fake_run(namespace)
            return runner_returncode
        if stage == "SEALED_POSTHOC":
            _write_fake_posthoc(namespace)
            if fail_execution_receipt:
                original_write = namespace["write_json"]

                def fail_one_receipt(path, value):
                    if Path(path).name == "execution_receipt.json":
                        raise OSError("sentinel receipt failure")
                    return original_write(path, value)

                namespace["write_json"] = fail_one_receipt
            return 0
        raise AssertionError(f"unexpected external command: {stage}")

    namespace["logged"] = fake_logged
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "fake==1\n")
    for index in (3, 4):
        exec(compile(_code(notebook, index), f"cell-{index}", "exec"), namespace)
    exec(compile(_code(notebook, 5), "cell-5", "exec"), namespace)
    return stages


def test_notebook_is_deterministic_empty_single_file_handoff() -> None:
    notebook = _load_notebook()
    assert notebook == builder.build_notebook()
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    assert _code(notebook, 0) == "from google.colab import drive\ndrive.mount('/content/drive')"
    all_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "force_remount" not in all_source
    assert "RUN =" not in all_source
    assert "git clone" not in all_source
    assert "automatic_retry=False" in all_source
    assert "local_joint_state_payload_posthoc_v1_run" in all_source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))

    package, manifest = builder.portable_archive()
    assert COMPANION_PACKAGE.read_bytes() == package
    assert hashlib.sha256(package).hexdigest() in all_source
    assert manifest["git_commit"] is None
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES) | {"portable_source_manifest.json"}
        assert not any(name == ".git" or name.startswith(".git/") for name in archive.namelist())
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    runner_tree = ast.parse(
        (builder.ROOT / "experiments/wan_state_clock/local_joint_state_payload_v1_run.py").read_text()
    )
    runner_closure = next(
        ast.literal_eval(node.value)
        for node in runner_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_CLOSURE" for target in node.targets)
    )
    assert tuple(runner_closure) == builder.RUNNER_FILES


def test_stubbed_run_all_success_preserves_fixed_identity_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _load_notebook()
    namespace, mounts = _execute_setup(notebook, tmp_path, monkeypatch)
    stages = _execute_environment_source_and_run(notebook, namespace, monkeypatch)
    exec(compile(_code(notebook, 6), "cell-6", "exec"), namespace)

    assert mounts == ["/content/drive"]
    assert stages == ["DEPENDENCY_PROBE", "DEPENDENCY_REPORT", "REAL_FIXED_RUN", "SEALED_POSTHOC"]
    assert not (namespace["WORKSPACE"] / ".git").exists()
    audit = json.loads((namespace["OUTPUT"] / "notebook_audit.json").read_text())
    assert audit["run"]["source_identity"]["git_commit"] is None
    assert audit["raw_seal"]["entries"] == 12
    assert audit["posthoc"]["scientific_pass"] is False
    slots = json.loads((namespace["OUTPUT"] / "fixed_slots.json").read_text())
    assert slots["status"] == "SUPERSEDED_BY_RUN_AND_POSTHOC"
    assert set(slots["arms"]) == {"OFF", "JOINT"}
    assert all(len(arm["steps"]) == 50 for arm in slots["arms"].values())


def test_runner_nonzero_retains_primary_and_still_seals_posthoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _load_notebook()
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="real runner exited 7"):
        _execute_environment_source_and_run(
            notebook, namespace, monkeypatch, runner_returncode=7
        )

    assert (namespace["RUN_OUTPUT"] / "result.json").is_file()
    assert (namespace["POSTHOC_OUTPUT"] / "raw_observation_seal.json").is_file()
    receipt = json.loads((namespace["OUTPUT"] / "execution_receipt.json").read_text())
    assert receipt["run_returncode"] == 7
    assert receipt["posthoc_returncode"] == 0
    failure = json.loads((namespace["OUTPUT"] / "notebook_failure.json").read_text())
    assert failure["stage"] == "REAL_RUN_OR_POSTHOC"
    assert "real runner exited 7" in failure["reason"]


def test_dependency_failure_preserves_primary_and_fixed_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _load_notebook()
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    primary = RuntimeError("sentinel dependency failure")

    def fail_logged(_command, stage, **_kwargs):
        assert stage == "DEPENDENCY_PROBE"
        raise primary

    namespace["logged"] = fail_logged
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")
    with pytest.raises(RuntimeError) as caught:
        exec(compile(_code(notebook, 3), "cell-3", "exec"), namespace)
    assert caught.value is primary
    slots = json.loads((namespace["OUTPUT"] / "fixed_slots.json").read_text())
    assert slots["status"] == "SETUP_NOT_COMPLETED"
    assert set(slots["arms"]) == {"OFF", "JOINT"}
    failure = json.loads((namespace["OUTPUT"] / "notebook_failure.json").read_text())
    assert failure["stage"] == "ENVIRONMENT_SETUP"
    assert failure["retry"] is False


def test_receipt_failure_does_not_replace_runner_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = _load_notebook()
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="real runner exited 9") as caught:
        _execute_environment_source_and_run(
            notebook,
            namespace,
            monkeypatch,
            runner_returncode=9,
            fail_execution_receipt=True,
        )
    assert type(caught.value) is RuntimeError
    failure = json.loads((namespace["OUTPUT"] / "notebook_failure.json").read_text())
    assert failure["reason"] == "RuntimeError: real runner exited 9"
    assert (namespace["OUTPUT"] / "notebook_audit.json").is_file()
