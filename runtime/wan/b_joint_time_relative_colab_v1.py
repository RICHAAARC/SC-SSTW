"""Standalone fresh Joint-A temporal confirmation bootstrap; stdlib imports only until explicitly called.

This file may be read from an editable source URL before the rest of the project.
Cloud receipt is a consumed-attempt record, not a code/weight integrity check.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

RESULT_ROOT_ID = "1JvuNbrLs6Mo0yoPPgeYBuiVvd2UvCglA"
COMPONENTS = ("B-Joint-Time-Relative-Consistency-V1", "fresh-r01-v1", "attempt01")
OUTPUT = Path("/content/drive/MyDrive/Video-WM").joinpath(*COMPONENTS)
WORK = Path("/content/video-wm-joint-relative-v1")
SOURCE_REPOSITORY = "RICHAAARC/SC-SSTW"
SOURCE_REF = "dev/local-joint-state-payload-v1"


def reservation_units():
    return [dict(id=f"joint-relative-r01/{arm}/{view}", arm=arm, view=view, frames=n, edges=n-1,
                 status="NOT_STARTED") for arm in ("NEW", "BASE")
            for view, n in zip(("FULL", "CROP", "DELETE1", "REPEAT1", "DELETE3"), (181, 101, 180, 182, 178))]


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = member.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if (name.startswith("/") or any(":" in p for p in name.split("/"))
                    or not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("unsafe source archive path: " + name)
        zipped.extractall(destination)


def download_source(repository, destination, ref="main"):
    from urllib.parse import quote
    destination = Path(destination)
    if destination.exists():
        return destination
    stage = Path(tempfile.mkdtemp(prefix="source-", dir=WORK))
    archive = stage / "source.zip"
    url = f"https://codeload.github.com/{repository}/zip/refs/heads/{quote(ref, safe='/')}"
    urllib.request.urlretrieve(url, archive)
    safe_extract(archive, stage / "unpacked")
    folders = list((stage / "unpacked").iterdir())
    if len(folders) != 1 or not folders[0].is_dir():
        raise ValueError("source archive has no single repository directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(folders[0]), str(destination))
    return destination


class Cloud:
    def __init__(self):
        from google.colab import auth
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        auth.authenticate_user()
        credentials, _ = google.auth.default()
        self.session = AuthorizedSession(credentials)

    def request(self, method, url, **kwargs):
        response = self.session.request(method, url, timeout=(20, 120), **kwargs)
        response.raise_for_status()
        return response

    def children(self, parent):
        rows, token = [], None
        while True:
            params = dict(q=f"'{parent}' in parents and trashed=false", pageSize=1000,
                          fields="nextPageToken,incompleteSearch,files(id,name,mimeType,size)")
            if token:
                params["pageToken"] = token
            value = self.request("GET", "https://www.googleapis.com/drive/v3/files", params=params).json()
            if value.get("incompleteSearch"):
                raise RuntimeError("Drive directory listing incomplete; attempt state unknown")
            rows.extend(value.get("files", []))
            token = value.get("nextPageToken")
            if not token:
                return rows

    def directory(self, create=False):
        parent = RESULT_ROOT_ID
        for name in COMPONENTS:
            found = [f for f in self.children(parent) if f["name"] == name]
            if len(found) > 1:
                raise RuntimeError(f"Multiple result folders named {name}; location ambiguous, do not start")
            if found:
                if found[0]["mimeType"] != "application/vnd.google-apps.folder":
                    raise ValueError("Result directory path is a file: " + name)
                parent = found[0]["id"]
            elif not create:
                return None
            else:
                parent = self.request("POST", "https://www.googleapis.com/drive/v3/files",
                                      json=dict(name=name, parents=[parent], mimeType="application/vnd.google-apps.folder")).json()["id"]
        return parent

    def write_json(self, parent, name, value):
        import uuid
        boundary = "native-baseline-" + uuid.uuid4().hex
        metadata = json.dumps(dict(name=name, parents=[parent], mimeType="application/json"))
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{metadata}\r\n"
                f"--{boundary}\r\nContent-Type: application/json\r\n\r\n"
                + json.dumps(value, ensure_ascii=False, allow_nan=False) + f"\r\n--{boundary}--\r\n").encode()
        record = self.request("POST", "https://www.googleapis.com/upload/drive/v3/files",
                              params=dict(uploadType="multipart", fields="id,name"), data=body,
                              headers={"Content-Type": "multipart/related; boundary=" + boundary}).json()
        # Read the receipt as actual JSON; no hash/source equality check.
        saved = self.request("GET", f"https://www.googleapis.com/drive/v3/files/{record['id']}", params={"alt": "media"}).json()
        if not isinstance(saved, dict):
            raise ValueError("Cloud result receipt is not readable JSON")
        return record


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _install_dependencies(source):
    import importlib.util
    path = source / "scripts/b_framewise_qim_v1_setup.py"
    spec = importlib.util.spec_from_file_location("joint_relative_dependency_setup", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    logs = OUTPUT / "preparation"
    logs.mkdir(parents=True, exist_ok=True)
    setup.ensure_imports("import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; from diffusers import WanPipeline,AutoencoderKLWan; import runtime.wan.b_joint_time_relative_physical_v1",
        ["numpy", "torch", "psutil", "accelerate", "safetensors", "huggingface-hub", "diffusers", "transformers", "sentencepiece", "ftfy"],
        "fresh Joint-A actual imports", source, logs / "setup.log")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        setup.command(["apt-get", "-qq", "update"], "Prepare media dependency", source, logs / "setup.log")
        setup.command(["apt-get", "-qq", "install", "-y", "ffmpeg"], "Install media dependency", source, logs / "setup.log")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    return environment


def stop_worker_group(worker):
    """Stop the reserved runner and every inherited stage/codec process on Colab."""
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A parent may exit before a child that ignores SIGTERM. Close that group too.
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    worker.wait()


def run_colab(source_directory=None, source_ref=SOURCE_REF):
    """One user Run-all. Existing cloud or mounted artifacts are read-only."""
    cloud = Cloud()
    parent = cloud.directory(create=False)
    cloud_rows = cloud.children(parent) if parent else []
    local_rows = list(OUTPUT.iterdir()) if OUTPUT.exists() else []
    if cloud_rows or local_rows:
        print("Consumed attempt: read-only; no setup/model/media rerun.")
        show_results(cloud, parent)
        return dict(status="READ_ONLY", output=str(OUTPUT))
    parent = cloud.directory(create=True)
    # This guard is for single-notebook use, not a distributed mutual-exclusion service.
    if cloud.children(parent):
        print("Attempt appeared while preparing; read-only.")
        show_results(cloud, parent)
        return dict(status="READ_ONLY", output=str(OUTPUT))
    receipt = dict(status="RESERVED", consumed=True, reserved_at=time.time(),
                   experiment="joint-time-relative-consistency-fresh-r01-v1", attempt="attempt01",
                   planned_units=reservation_units(), planned_nodes=1644, planned_edges=1634,
                   planned_hard_bit_slots=1035720, E41=0, identity_new_reads=0)
    cloud.write_json(parent, "cloud_start.json", receipt)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _save(OUTPUT / "attempt.json", receipt)
    worker = None
    outcome = dict(status="RESERVED", consumed=True)
    try:
        _save(OUTPUT / "planned_units.json", reservation_units())
        WORK.mkdir(parents=True, exist_ok=True)
        os.environ.update(HF_HOME=str(WORK / "cache/huggingface"), TORCH_HOME=str(WORK / "cache/torch"))
        source = Path(source_directory) if source_directory else download_source(SOURCE_REPOSITORY, WORK / "project", source_ref)
        runner = source / "experiments/wan_state_clock/b_joint_time_relative_validation_v1_run.py"
        if not runner.is_file():
            raise FileNotFoundError("Available source lacks fresh Joint-A runner: " + str(runner))
        sys.path.insert(0, str(source))
        from main.tube_state.b_joint_time_relative_validation_v1 import denominators, operation_plan, units, validate
        config = json.loads((source / "experiments/wan_state_clock/configs/b_joint_time_relative_validation_v1.json").read_text())
        validate(config)
        _save(OUTPUT / "planned_units.json", units(config))
        _save(OUTPUT / "planned_calls.json", operation_plan(config))
        _save(OUTPUT / "protocol.json", config)
        environment = _install_dependencies(source)
        args = [sys.executable, "-u", "-c", "from experiments.wan_state_clock.b_joint_time_relative_validation_v1_run import main; raise SystemExit(main(notebook_reservation=True))",
                "--output", str(OUTPUT), "--scratch", str(WORK / "work")]
        with (OUTPUT / "runner.log").open("a") as log:
            worker = subprocess.Popen(args, cwd=source, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            while worker.poll() is None:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = OUTPUT / "progress.json"
                    print(progress.read_text() if progress.is_file() else "Same worker running; see runner.log", flush=True)
        print("Runner exit:", worker.returncode)
        # A stage can die while its codec child survives; even a normal STOPPED
        # parent result must close that inherited group before handing back control.
        stop_worker_group(worker)
        terminal_path = OUTPUT / "result.json"
        terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else None
        if not isinstance(terminal, dict) or terminal.get("status") not in ("COMPLETED", "STOPPED_FIXED_ATTEMPT"):
            raise RuntimeError(f"Worker exited {worker.returncode} without a readable final result; attempt remains consumed")
        # Keep an already-saved specific failure; do not replace it with a generic exit error.
        if worker.returncode != 0 and terminal["status"] != "STOPPED_FIXED_ATTEMPT":
            raise RuntimeError(f"Worker exited {worker.returncode} despite a saved completed result")
        outcome = dict(status=terminal["status"], consumed=True, worker_returncode=worker.returncode,
                       failure=terminal.get("failure"), evaluation_failure=terminal.get("evaluation_failure"))
        previous = json.loads((OUTPUT / "attempt.json").read_text())
        _save(OUTPUT / "attempt.json", dict(previous, status=terminal["status"], consumed=True,
                                           worker_returncode=worker.returncode))
    except BaseException as error:
        if worker is not None:
            stop_worker_group(worker)
        failure = dict(status="STOPPED_FIXED_ATTEMPT", consumed=True, error=f"{type(error).__name__}: {error}",
                       planned_units=10, planned_nodes=1644, planned_edges=1634,
                       planned_hard_bit_slots=1035720, E41=0, identity_new_reads=0, retry=False,
                       worker_returncode=worker.returncode if worker is not None else None,
                       retained_accounting=["planned_units.json", "planned_calls.json", "events.jsonl", "ledger.json"])
        outcome = failure
        try:
            _save(OUTPUT / "preparation_failure.json", failure)
            previous = json.loads((OUTPUT / "attempt.json").read_text())
            _save(OUTPUT / "attempt.json", dict(previous, status="STOPPED_FIXED_ATTEMPT", consumed=True,
                                               stopped_at=time.time(), bootstrap_failure=failure))
        except Exception as storage_error:
            failure["mounted_save_error"] = f"{type(storage_error).__name__}: {storage_error}"
        try:
            cloud.write_json(parent, "preparation_failure_cloud.json", failure)
        except Exception as storage_error:
            failure["cloud_save_error"] = f"{type(storage_error).__name__}: {storage_error}"
        print(failure)
    # Persist compact core results through the API as well as the mounted directory.
    # Copies are ordinary backups; no source/hash validation or model call follows.
    for name in ("result.json", "evaluation.json", "attempt.json"):
        path = OUTPUT / name
        if path.is_file():
            try:
                cloud.write_json(parent, name.replace(".json", "_cloud_copy.json"), json.loads(path.read_text()))
            except Exception as error:
                print("Cloud copy failed; mounted results retained:", name, type(error).__name__, str(error))
    show_results(cloud, parent)
    return dict(output=str(OUTPUT), **outcome)


def show_results(cloud=None, parent=None):
    print("Retained result directory:", OUTPUT)
    found = False
    for name in ("result.json", "evaluation.json", "preparation_failure.json"):
        path = OUTPUT / name
        try:
            if not path.is_file():
                continue
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print("Saved file unreadable; retained without retry:", name, type(error).__name__, str(error))
        else:
            print(name, json.dumps(value, ensure_ascii=False, indent=2))
            found = True
    if not found and cloud and parent:
        try:
            cloud_rows = cloud.children(parent)
        except Exception as error:
            print("Cloud result listing unavailable:", type(error).__name__, str(error))
            cloud_rows = []
        for row in cloud_rows:
            if row["name"] in ("result_cloud_copy.json", "evaluation_cloud_copy.json", "preparation_failure_cloud.json", "cloud_start.json"):
                try:
                    value = cloud.request("GET", f"https://www.googleapis.com/drive/v3/files/{row['id']}", params={"alt": "media"}).json()
                    print(row["name"], json.dumps(value, ensure_ascii=False, indent=2))
                except Exception as error:
                    print("Cloud receipt unreadable:", row["name"], type(error).__name__, str(error))
    if not found:
        print("Final mounted result not readable yet. Existing reservation stays consumed; no automatic retry.")
