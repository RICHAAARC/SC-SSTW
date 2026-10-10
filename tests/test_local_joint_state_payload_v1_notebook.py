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
        if "-c" in _command:
            compile(_command[_command.index("-c") + 1], f"{stage}-child", "exec")
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
    # This is the already-executed historical handoff. New optional runtime
    # observers do not rewrite its embedded source snapshot.
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

    source_tree = ast.parse(_code(notebook, 4))
    encoded = next(ast.literal_eval(node.value) for node in source_tree.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "SOURCE_PACKAGE_B64"
                       for target in node.targets))
    package = base64.b64decode("".join(encoded))
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        manifest = json.loads(archive.read("portable_source_manifest.json"))
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


@pytest.mark.parametrize("repair_required,version_difference", [(False, False), (False, True), (True, False)])
def test_generated_dependency_probe_runs_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair_required: bool, version_difference: bool
) -> None:
    notebook = _load_notebook()
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    expected = _dependency_receipt(namespace)
    stages: list[str] = []
    probe_outputs: list[subprocess.CompletedProcess] = []
    repaired = False

    def run_probe(command, stage, *, check=True, **_kwargs):
        nonlocal repaired
        stages.append(stage)
        if stage in ("DEPENDENCY_PROBE", "DEPENDENCY_REPROBE"):
            versions = dict(expected["versions"])
            versions["torch"] += "+cu130"
            if version_difference:
                versions["diffusers"] = "0.40.0"
            assert command[1:3] == ["-u", "-c"]
            # Execute the actual generated child source in a fresh interpreter.
            # Only external dependency/GPU interfaces are replaced; syntax and
            # the JSON receipt writer are real, with no pip/model/Drive work.
            harness = "\n".join([
                "import importlib.metadata, shutil, sys, types",
                f"versions = {versions!r}",
                "importlib.metadata.version = versions.__getitem__",
                "shutil.which = lambda name: '/fake/' + name",
                "torch = types.ModuleType('torch')",
                "torch.version = types.SimpleNamespace(cuda='13.0')",
                "torch.cuda = types.SimpleNamespace(is_available=lambda: True, "
                "get_device_name=lambda index: 'FAKE GPU', "
                "get_device_properties=lambda index: types.SimpleNamespace(total_memory=1))",
                "diffusers = types.ModuleType('diffusers')",
                "diffusers.AutoencoderKLWan = object",
                "pass" if repair_required and not repaired else "diffusers.WanPipeline = object",
                "sys.modules.update(torch=torch, diffusers=diffusers)",
                f"exec(compile({command[3]!r}, '<actual-dependency-probe>', 'exec'))",
            ])
            result = subprocess.run(
                [sys.executable, "-I", "-c", harness],
                text=True, capture_output=True, check=False,
            )
            probe_outputs.append(result)
            if check and result.returncode:
                raise subprocess.CalledProcessError(
                    result.returncode, command, output=result.stdout, stderr=result.stderr
                )
            return result.returncode
        if stage == "DEPENDENCY_REPAIR":
            assert repair_required and not repaired
            repaired = True
            return 0
        assert stage == "DEPENDENCY_REPORT"
        return 1

    namespace["logged"] = run_probe
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "fake==1\n")
    exec(compile(_code(notebook, 3), "cell-3", "exec"), namespace)

    assert stages == (
        ["DEPENDENCY_PROBE", "DEPENDENCY_REPAIR", "DEPENDENCY_REPROBE", "DEPENDENCY_REPORT"]
        if repair_required else ["DEPENDENCY_PROBE", "DEPENDENCY_REPORT"]
    )
    assert [row.returncode for row in probe_outputs] == ([1, 0] if repair_required else [0])
    if repair_required:
        assert "ImportError" in probe_outputs[0].stderr
    receipt_path = namespace["OUTPUT"] / "dependency_probe.json"
    assert receipt_path.read_bytes().endswith(b"\n")
    receipt = json.loads(receipt_path.read_text())
    expected["versions"]["torch"] += "+cu130"
    expected["version_differences"] = {}
    if version_difference:
        expected["versions"]["diffusers"] = "0.40.0"
        expected["version_differences"] = {"diffusers": {"reference": "0.39.0", "actual": "0.40.0"}}
    assert receipt == expected
    environment = json.loads((namespace["OUTPUT"] / "environment_receipt.json").read_text())
    assert environment["dependency_check_returncode"] == 1
    assert environment["cuda_available"] is True


@pytest.mark.parametrize("manifest_mode", ["missing", "malformed"])
def test_portable_provenance_differences_do_not_block_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_mode: str
) -> None:
    notebook = _load_notebook()
    original, _ = builder.portable_archive()
    edited = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(edited, "w") as archive:
        for name in source.namelist():
            if name == "portable_source_manifest.json":
                if manifest_mode == "malformed":
                    archive.writestr(name, b"not-json")
                continue
            data = source.read(name)
            if name == "main/tube_state/local_joint_state_payload_v1.py":
                data += b"\n# Local source annotation.\n"
            if name == builder.CONFIG_PATH:
                data = json.dumps(json.loads(data), indent=4).encode() + b"\n"
            archive.writestr(name, data)
        archive.writestr("local_notes.txt", "Additional user notes are allowed.\n")
    source_tree = ast.parse(_code(notebook, 4))
    package_assignment = next(n for n in source_tree.body if isinstance(n, ast.Assign))
    assert package_assignment.targets[0].id == "SOURCE_PACKAGE_B64"
    package_assignment.value = ast.Constant(base64.b64encode(edited.getvalue()).decode())
    notebook["cells"][4]["source"] = [ast.unparse(source_tree)]
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    _execute_environment_source_and_run(notebook, namespace, monkeypatch)
    receipt = json.loads((namespace["OUTPUT"] / "portable_source_receipt.json").read_text())
    assert receipt["blocking"] is False
    assert receipt["package_sha256"] != receipt["package_reference_sha256"]
    assert receipt["config_sha256"] != namespace["CONFIG_SHA256"]
    assert (namespace["WORKSPACE"] / "local_notes.txt").is_file()
    assert (namespace["POSTHOC_OUTPUT"] / "posthoc_result.json").is_file()


@pytest.mark.parametrize("unsafe_path", [False, True])
def test_unusable_or_unsafe_source_archive_retains_real_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_path: bool
) -> None:
    notebook = _load_notebook()
    if unsafe_path:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../outside.py", b"# Must not escape extraction directory.\n")
        package = stream.getvalue()
    else:
        package = b"This is not a ZIP archive."
    source_tree = ast.parse(_code(notebook, 4))
    assignment = next(n for n in source_tree.body if isinstance(n, ast.Assign))
    assignment.value = ast.Constant(base64.b64encode(package).decode())
    namespace, _ = _execute_setup(notebook, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError if unsafe_path else zipfile.BadZipFile):
        exec(compile(ast.unparse(source_tree), "cell-4", "exec"), namespace)
    assert not (tmp_path / "outside.py").exists()
    failure = json.loads((namespace["OUTPUT"] / "notebook_failure.json").read_text())
    assert failure["stage"] == "PORTABLE_SOURCE"


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
