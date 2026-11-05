"""Small standalone Colab bootstrap; stdlib imports only until explicitly called.

This file may be read from an editable source URL before the rest of the project.
Cloud receipt is a consumed-attempt record, not a code/weight integrity check.
"""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

RESULT_ROOT_ID = "1Vbw0Gl2vtG4UFzvtfGBTit9cZQz4gpOU"
COMPONENTS = ("baselines", "native-wam-pixelseal-temporal-dev-v2", "attempt01")
OUTPUT = Path("/content/drive/MyDrive/Video-WM_Result").joinpath(*COMPONENTS)
WORK = Path("/content/video-wm-native-baselines-temporal-v2")
SOURCE_REPOSITORY = "RICHAAARC/SC-SSTW"
SOURCE_REF = "dev/local-joint-state-payload-v1"


def reservation_plan():
    """Stdlib fallback slots retained even when project download/setup fails."""
    lengths = {"IDENTITY": 181, **{f"TRIM_S{s}_L129": 129 for s in range(26, 30)},
               "DROP18": 163, "DROP54": 127, "BURST18": 163, "BURST54": 127,
               "REPEAT_FRAME18": 199, "REPLAY_BLOCK18": 199}
    return [dict(id=f"{m}/b-rank3r-m{s:02d}/{a}/{c}/{v}", method=m, source=f"b-rank3r-m{s:02d}",
                 arm=a, channel=c, condition=v, frames=n, nbits=bits, status="NOT_STARTED")
            for m, bits in (("wam", 32), ("pixelseal", 256)) for s in range(1, 5)
            for a in ("WM", "OFF") for c in ("G1_EDIT_RGB", "G2_EDIT_H264") for v, n in lengths.items()]


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


def _install_dependencies(source, vendor):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(source), str(vendor / "videoseal"), str(vendor / "watermark-anything")])
    mapping = dict(numpy="numpy", torch="torch", torchvision="torchvision", omegaconf="omegaconf",
                   pandas="pandas", einops="einops", lpips="lpips", timm="timm", pycocotools="pycocotools",
                   tqdm="tqdm", pytorch_msssim="pytorch-msssim", tensorboard="tensorboard", calflops="calflops",
                   transformers="transformers", safetensors="safetensors", scipy="scipy", av="av",
                   huggingface_hub="huggingface-hub", ftfy="ftfy", sentencepiece="sentencepiece",
                   accelerate="accelerate", cv2="opencv-python-headless", pywt="PyWavelets",
                   skimage="scikit-image", sklearn="scikit-learn", matplotlib="matplotlib", PIL="pillow",
                   yaml="PyYAML", ffmpeg="ffmpeg-python", requests="requests", decord="decord")
    probe = """import json
try:
 import numpy, torch, torchvision, videoseal.utils.cfg
 import watermark_anything.models, watermark_anything.data.metrics
 import importlib.util
 from pathlib import Path
 p=Path(__import__('sys').argv[1])/'watermark-anything/notebooks/inference_utils.py'
 s=importlib.util.spec_from_file_location('wam_import_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 print(json.dumps({'ok':True}))
except ModuleNotFoundError as e:
 print(json.dumps({'missing':e.name}));raise SystemExit(3)
"""
    installed = set()
    for _ in range(len(mapping)+1):
        checked = subprocess.run([sys.executable, "-c", probe, str(vendor)], cwd=vendor / "videoseal",
                                 env=environment, capture_output=True, text=True)
        if checked.returncode == 0:
            break
        if checked.returncode != 3:
            raise RuntimeError("Actual dependency import failed: " + checked.stderr[-6000:])
        missing = json.loads(checked.stdout.splitlines()[-1])["missing"].split(".")[0]
        if missing not in mapping or missing in installed:
            raise RuntimeError("Unresolved dependency import: " + missing + " " + checked.stderr[-2000:])
        subprocess.run([sys.executable, "-m", "pip", "install", mapping[missing]], check=True)
        installed.add(missing)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
    return environment


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
                   experiment="native-wam-pixelseal-temporal-dev-v2", attempt="attempt01",
                   planned_read_tasks=reservation_plan(), planned_frame_rows=53600,
                   planned_frame_bits=dict(wam=857600, pixelseal=6860800),
                   planned_video_messages=176, planned_video_bits=45056)
    cloud.write_json(parent, "cloud_start.json", receipt)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _save(OUTPUT / "attempt.json", receipt)
    worker = None
    outcome = dict(status="RESERVED", consumed=True)
    try:
        _save(OUTPUT / "planned_read_tasks.json", reservation_plan())
        WORK.mkdir(parents=True, exist_ok=True)
        os.environ.update(HF_HOME=str(WORK / "cache/huggingface"), TORCH_HOME=str(WORK / "cache/torch"))
        source = Path(source_directory) if source_directory else download_source(SOURCE_REPOSITORY, WORK / "project", source_ref)
        runner = source / "experiments/wan_state_clock/native_baselines_temporal_v2_run.py"
        if not runner.is_file():
            raise FileNotFoundError("Available source lacks native baseline runner: " + str(runner))
        sys.path.insert(0, str(source))
        from main.baselines.native_paper_temporal_v2 import denominators, operations, read_tasks, physical_budget, validate_config
        config = json.loads((source / "experiments/wan_state_clock/configs/native_baselines_temporal_v2.json").read_text())
        validate_config(config)
        _save(OUTPUT / "planned_read_tasks.json", read_tasks(config))
        _save(OUTPUT / "physical_budget.json", physical_budget(config))
        _save(OUTPUT / "planned_calls.json", operations(config))
        _save(OUTPUT / "protocol.json", config)
        vendor = WORK / "vendor"
        download_source("facebookresearch/videoseal", vendor / "videoseal")
        download_source("facebookresearch/watermark-anything", vendor / "watermark-anything")
        environment = _install_dependencies(source, vendor)
        weight = vendor / "watermark-anything/checkpoints/wam_mit.pth"
        if not weight.is_file():
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth", weight)
        args = [sys.executable, "-u", "-c", "from experiments.wan_state_clock.native_baselines_temporal_v2_run import main; raise SystemExit(main(notebook_reservation=True))",
                "--output", str(OUTPUT), "--scratch", str(WORK / "work"), "--vendor", str(vendor)]
        with (OUTPUT / "runner.log").open("a") as log:
            worker = subprocess.Popen(args, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while worker.poll() is None:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = OUTPUT / "progress.json"
                    print(progress.read_text() if progress.is_file() else "Same worker running; see runner.log", flush=True)
        print("Runner exit:", worker.returncode)
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
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        failure = dict(status="STOPPED_FIXED_ATTEMPT", consumed=True, error=f"{type(error).__name__}: {error}",
                       planned_read_tasks=352, planned_frame_rows=53600,
                       planned_frame_bits=dict(wam=857600, pixelseal=6860800), retry=False,
                       planned_video_messages=176, planned_video_bits=45056,
                       worker_returncode=worker.returncode if worker is not None else None,
                       retained_accounting=["planned_read_tasks.json", "planned_calls.json", "events.jsonl", "ledger.json"])
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
            print(name, json.dumps(_display_value(value), ensure_ascii=False, indent=2))
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
                    print(row["name"], json.dumps(_display_value(value), ensure_ascii=False, indent=2))
                except Exception as error:
                    print("Cloud receipt unreadable:", row["name"], type(error).__name__, str(error))
    if not found:
        print("Final mounted result not readable yet. Existing reservation stays consumed; no automatic retry.")


def _display_value(value):
    if not isinstance(value, dict):
        return value
    large = {"source_condition_channel_details", "source_retention", "video_messages", "comparisons", "planned_read_tasks"}
    return {k: {"saved_rows": len(v), "detail": "Full rows retained in JSON/CSV"}
            if k in large and isinstance(v, list) else v for k, v in value.items()}
