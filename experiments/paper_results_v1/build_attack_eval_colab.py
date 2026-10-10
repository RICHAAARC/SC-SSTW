"""Build the self-contained fixed two-pilot temporal-attack Colab notebook.

Building and testing this notebook never imports or downloads model packages.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/paper_results_v1_temporal_attack_two_pilot_colab.ipynb"


def _files():
    paths = [
        ROOT / "experiments/__init__.py", ROOT / "main/__init__.py", ROOT / "runtime/__init__.py",
        ROOT / "experiments/paper_results_v1/attack_eval.adopted.json",
    ]
    paths.extend(sorted((ROOT / "experiments/paper_results_v1").glob("*.py")))
    paths.extend(sorted((ROOT / "main/tube_state").rglob("*.py")))
    paths.extend(sorted((ROOT / "runtime/wan").rglob("*.py")))
    unique = {path.relative_to(ROOT).as_posix(): path for path in paths}
    return [(name, unique[name]) for name in sorted(unique)]


def _payload():
    files = _files()
    manifest = {
        "schema_version": "paper-results-v1-temporal-attack-portable-v1",
        "files": {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for name, path in files},
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16; archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("portable_manifest.json", (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16; archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    raw = stream.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest(), manifest


def _cell(kind, source, ident):
    row = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": ident}
    if kind == "code": row.update(execution_count=None, outputs=[])
    return row


def build_notebook():
    encoded, digest, manifest = _payload()
    drive = "from google.colab import drive\ndrive.mount('/content/drive')\n"
    setup = textwrap.dedent(f'''\
        # Extract the embedded source and initialize the complete fixed plan before model setup.
        import base64, datetime, hashlib, importlib, json, os, shutil, signal, subprocess, sys, time, traceback, urllib.request, uuid, zipfile
        from pathlib import Path

        PORTABLE_B64 = {encoded!r}
        PORTABLE_SHA256 = {digest!r}
        PILOTS = ("pilot_01", "pilot_02")
        PHASES = (
            "import-main", "baseline-embed-videoseal", "baseline-embed-rivagan",
            "baseline-codec", "attack-media", "receiver-clock", "receiver-read",
            "baseline-extract-videoseal", "baseline-extract-rivagan", "quality",
        )
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        RUN_ID = stamp + "-" + uuid.uuid4().hex[:8]
        OUTPUT_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack") / RUN_ID
        RUN_OUTPUT = OUTPUT_ROOT / "run_state"
        TEMP_ROOT = Path("/content") / ("paper-results-v1-attack-temp-" + RUN_ID)
        PORTABLE_ROOT = Path("/content") / ("paper-results-v1-attack-source-" + RUN_ID)
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Two-Pilot/20261009T123350716100Z-f5e1f880/run_state")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False); CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        LOG = OUTPUT_ROOT / "execution.log"; RECEIPTS = OUTPUT_ROOT / "stage_receipts.json"

        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\\n"); os.replace(tmp, path)
        receipts = []
        def record(stage, status, **fields):
            row = {{"stage": stage, "status": status, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}}
            receipts.append(row); atomic_json(RECEIPTS, receipts); return row
        PYTHON = sys.executable
        def logged(command, stage, *, cwd=None, env=None, check=True):
            print(f"[{{stage}}] start; log={{LOG}}", flush=True); child = None; rc = None
            started = time.monotonic()
            try:
                with LOG.open("a") as stream:
                    stream.write("COMMAND " + repr(command) + "\\n"); stream.flush()
                    child = subprocess.Popen(command, cwd=cwd or PORTABLE_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    rc = child.wait()
                record(stage, "COMPLETE" if rc == 0 else "FAILED", returncode=rc, seconds=time.monotonic()-started, command=command)
                if check and rc: raise RuntimeError(f"{{stage}} exited {{rc}}")
                return rc
            except BaseException as exc:
                record(stage, "FAILED", returncode=rc, seconds=time.monotonic()-started, command=command, reason=f"{{type(exc).__name__}}: {{exc}}")
                raise
            finally:
                if child is not None and child.poll() is None:
                    try: os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=5)
                    except BaseException:
                        try: os.killpg(child.pid, signal.SIGKILL)
                        except BaseException: pass
                        try: child.wait(timeout=5)
                        except BaseException: pass
                print(f"[{{stage}}] end rc={{rc}}", flush=True)

        raw = base64.b64decode(PORTABLE_B64); archive = OUTPUT_ROOT / "portable_source.zip"; archive.write_bytes(raw)
        if hashlib.sha256(raw).hexdigest() != PORTABLE_SHA256: record("portable", "RECORDED_DIFFERENCE", expected=PORTABLE_SHA256)
        PORTABLE_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                target = (PORTABLE_ROOT / item.filename).resolve()
                if PORTABLE_ROOT.resolve() not in target.parents and target != PORTABLE_ROOT.resolve(): raise RuntimeError("unsafe portable path")
            zf.extractall(PORTABLE_ROOT)
        sys.path.insert(0, str(PORTABLE_ROOT))
        CONFIG = OUTPUT_ROOT / "effective_attack_config.json"
        cfg = json.loads((PORTABLE_ROOT / "experiments/paper_results_v1/attack_eval.adopted.json").read_text())
        cfg["source_run"]["run_dir"] = str(SOURCE_RUN)
        VIDEOSEAL_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"; RIVAGAN_COMMIT = "efffa72a4ca46d4d5051f6970c96424c2cdab441"
        WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"; FRAMEWISE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / WAN_REVISION
        FRAMEWISE_SNAPSHOT = CACHE_ROOT / "models/sd-vae-ft-mse" / FRAMEWISE_REVISION
        VS_ROOT = CACHE_ROOT / "sources" / ("videoseal-" + VIDEOSEAL_COMMIT); VS_CARD = VS_ROOT / "videoseal/cards/videoseal_1.0.yaml"; VS_WEIGHT = CACHE_ROOT / "checkpoints/videoseal/y_256b_img.pth"
        RIVA_ROOT = CACHE_ROOT / "sources" / ("rivagan-" + RIVAGAN_COMMIT); RIVA_WEIGHT = CACHE_ROOT / "checkpoints/rivagan/rivagan_32bit_model.pt"
        cfg["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        cfg["models"]["framewise"]["local_snapshot_path"] = str(FRAMEWISE_SNAPSHOT)
        cfg["models"]["videoseal"].update(source_root=str(VS_ROOT), card_path=str(VS_CARD), checkpoint_path=str(VS_WEIGHT))
        cfg["models"]["rivagan"].update(source_root=str(RIVA_ROOT), checkpoint_path=str(RIVA_WEIGHT))
        atomic_json(CONFIG, cfg)
        ENV = {{**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}}
        portable_env = ENV; EFFECTIVE_CONFIG = CONFIG
        record_stage = record
        def observe_sha256(path):
            try:
                with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest(), None
            except OSError as exc: return None, f"{{type(exc).__name__}}: {{exc}}"
        CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG)]
        logged([*CLI, "--output", str(OUTPUT_ROOT / "fixed_plan.json"), "--phase", "plan"], "fixed-plan", env=ENV, check=True)
        logged([*CLI, "--output", str(RUN_OUTPUT), "--temp-root", str(TEMP_ROOT), "--phase", "init"], "initialize", env=ENV, check=True)
        print("Initialized", RUN_ID, "with all 10 cases; only pilot_01/pilot_02 will be attempted.")
    ''')
    accepted_notebook = json.loads((ROOT / "notebooks/paper_results_v1_two_pilot_colab.ipynb").read_text(encoding="utf-8"))
    accepted_environment = next(
        "".join(cell["source"]) for cell in accepted_notebook["cells"]
        if cell.get("id") == "environment-and-models"
    )
    # Reuse the accepted 544d577 interpreter recovery, import-first repair,
    # per-baseline failure receipts and pip --python isolation. Only the Wan
    # snapshot is narrowed to the VAE required by this receiver-only run.
    accepted_environment = accepted_environment.replace(
        'allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"],',
        'allow_patterns=["model_index.json", "vae/*"],',
    )
    source_setup = textwrap.dedent('''\
        # Source/checkpoint preparation is isolated per external method.
        VIDEOSEAL_REPO = "https://github.com/facebookresearch/videoseal.git"
        VIDEOSEAL_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth"
        RIVAGAN_REPO = "https://github.com/DAI-Lab/RivaGAN.git"
        RIVAGAN_WEIGHT_COMMIT = "4d9928350509f808b6b57d48d2f958aef811d332"
        RIVAGAN_CHECKPOINT_URL = "https://raw.githubusercontent.com/Peachypie98/RivaGAN/4d9928350509f808b6b57d48d2f958aef811d332/model_weight/rivagan_32bit_model.pt"
        source_checkout_receipts = {}
        def optional_sha256(label, path):
            digest, error = observe_sha256(path)
            source_checkout_receipts[label] = {"actual_sha256": digest, "error": error, "blocking": False}
            return digest
        def prepare_source(label, url, commit, destination):
            destination = Path(destination); created = not destination.exists()
            if created:
                if logged(["git", "clone", "--filter=blob:none", url, str(destination)], label + "_CLONE", check=False):
                    raise RuntimeError(label + " clone failed")
                logged(["git", "-C", str(destination), "fetch", "origin", commit], label + "_FETCH", check=False)
                if logged(["git", "-C", str(destination), "checkout", "--detach", commit], label + "_CHECKOUT", check=False):
                    raise RuntimeError(label + " checkout failed")
            try:
                head = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
                dirty = subprocess.run(["git", "-C", str(destination), "status", "--porcelain"], text=True, capture_output=True, check=False)
                source_checkout_receipts[label] = {"requested": commit, "actual": head.stdout.strip() if head.returncode == 0 else None, "dirty": dirty.stdout.splitlines() if dirty.returncode == 0 else None, "blocking": False}
            except OSError as exc:
                source_checkout_receipts[label] = {"requested": commit, "actual": None, "reason": f"{type(exc).__name__}: {exc}", "blocking": False}
            return destination
        def cached_download(label, url, destination):
            destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                temporary = destination.with_suffix(destination.suffix + ".download"); temporary.unlink(missing_ok=True)
                urllib.request.urlretrieve(url, temporary); os.replace(temporary, destination)
            return destination, optional_sha256(label, destination)
        baseline_setup = {}
        try:
            prepare_source("VIDEOSEAL_SOURCE", VIDEOSEAL_REPO, VIDEOSEAL_COMMIT, VS_ROOT)
            VS_WEIGHT, vs_weight_sha = cached_download("VIDEOSEAL_CHECKPOINT", VIDEOSEAL_CHECKPOINT_URL, VS_WEIGHT)
            vs_card_sha = optional_sha256("VIDEOSEAL_CARD", VS_CARD)
            baseline_setup["videoseal"] = {"status": "READY", "card_sha256": vs_card_sha, "checkpoint_sha256": vs_weight_sha}
        except Exception as exc:
            vs_card_sha = vs_weight_sha = None; baseline_setup["videoseal"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        try:
            prepare_source("RIVAGAN_SOURCE", RIVAGAN_REPO, RIVAGAN_COMMIT, RIVA_ROOT)
            RIVA_WEIGHT, riva_weight_sha = cached_download("RIVAGAN_CHECKPOINT", RIVAGAN_CHECKPOINT_URL, RIVA_WEIGHT)
            baseline_setup["rivagan"] = {"status": "READY", "checkpoint_sha256": riva_weight_sha, "checkpoint_provenance": "Peachypie98 community 32-bit checkpoint; not DAI-Lab official weights"}
        except Exception as exc:
            riva_weight_sha = None; baseline_setup["rivagan"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        atomic_json(OUTPUT_ROOT / "baseline_setup_receipts.json", baseline_setup)
    ''')
    environment = source_setup + accepted_environment + textwrap.dedent('''\
        # Attack-quality imports are checked independently of the core stack.
        quality_environment_status = "READY"
        try:
            if logged([PYTHON, "-c", "import lpips,cv2; print('ATTACK_QUALITY_IMPORT_READY')"], "ATTACK_QUALITY_IMPORT_PROBE", check=False):
                logged([PYTHON, "-m", "pip", "install", "lpips", "opencv-python-headless"], "ATTACK_QUALITY_IMPORT_FAILURE_REPAIR")
                logged([PYTHON, "-c", "import lpips,cv2; print('ATTACK_QUALITY_IMPORT_READY')"], "ATTACK_QUALITY_IMPORT_REPROBE")
        except Exception as exc:
            quality_environment_status = "FAILED_RETAINED"
            record_stage("ATTACK_QUALITY_ENVIRONMENT", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        VS_PYTHON = baseline_pythons["videoseal"]
        RV_PYTHON = baseline_pythons["rivagan"]
    ''')
    execute = textwrap.dedent('''\
        # Fixed stage order. A failed baseline stage is retained and does not stop independent main stages.
        attempts = []
        interpreters = {"baseline-embed-videoseal": VS_PYTHON, "baseline-extract-videoseal": VS_PYTHON, "baseline-embed-rivagan": RV_PYTHON, "baseline-extract-rivagan": RV_PYTHON}
        interrupted = None
        try:
            for case_id in PILOTS:
                for phase in PHASES:
                    python = interpreters.get(phase, sys.executable)
                    command = [python, "-u", "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG), "--output", str(RUN_OUTPUT), "--temp-root", str(TEMP_ROOT), "--phase", phase, "--case-id", case_id]
                    try:
                        rc = logged(command, case_id + ":" + phase, env=ENV, check=False)
                        attempts.append({"case_id": case_id, "phase": phase, "returncode": rc})
                        atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
                        if rc:
                            logged([sys.executable, "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG), "--output", str(RUN_OUTPUT), "--phase", "record-failure", "--failed-phase", phase, "--case-id", case_id, "--reason", "subprocess exited " + str(rc)], case_id + ":retain-failure", env=ENV)
                    except KeyboardInterrupt as exc:
                        interrupted = exc; attempts.append({"case_id": case_id, "phase": phase, "status": "INTERRUPTED"}); atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
                        try: logged([sys.executable, "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG), "--output", str(RUN_OUTPUT), "--phase", "record-failure", "--failed-phase", phase, "--case-id", case_id, "--reason", "KeyboardInterrupt"], case_id + ":retain-interrupt", env=ENV)
                        except BaseException: pass
                        raise
                    except BaseException as exc:
                        attempts.append({"case_id": case_id, "phase": phase, "status": "FAILED_RETAINED", "reason": f"{type(exc).__name__}: {exc}"}); atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
                        try: logged([sys.executable, "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG), "--output", str(RUN_OUTPUT), "--phase", "record-failure", "--failed-phase", phase, "--case-id", case_id, "--reason", repr(exc)], case_id + ":retain-launch-failure", env=ENV)
                        except BaseException: pass
        finally:
            try: logged([sys.executable, "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG), "--output", str(RUN_OUTPUT), "--phase", "evaluate"], "final-evaluate", env=ENV)
            except BaseException as exc: record("final-evaluate", "FAILED", reason=f"{type(exc).__name__}: {exc}")
            summary = {"run_id": RUN_ID, "scope": {"attempted": list(PILOTS), "confirmation": "NOT_EXECUTED_BY_NOTEBOOK"}, "run_output": str(RUN_OUTPUT), "report": str(RUN_OUTPUT / "evaluation_report.json"), "interrupted": interrupted is not None}
            atomic_json(OUTPUT_ROOT / "handoff_summary.json", summary)
            print(json.dumps(summary, indent=2))
        if interrupted is not None: raise interrupted
    ''')
    closing = textwrap.dedent('''\
        # Return this unique Drive directory. Edited RGB and decoded readback stayed in /content temporary storage.
        print("Return directory:", OUTPUT_ROOT)
        print("Required: handoff_summary.json, execution.log, stage_receipts.json, attempts.json, fixed_plan.json, run_state/attack_run_state.json, run_state/evaluation_report.json, CSV files, records/, and media/.")
        print("The eight confirmation cases remain NOT_EXECUTED_BY_NOTEBOOK and are not pilot failures.")
    ''')
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "colab": {"name": OUTPUT.name},
        },
        "cells": [
            _cell("code", drive, "drive-mount"),
            _cell("markdown", "# Fixed two-pilot temporal attack evaluation\n\nRun all. The first cell asks you to authorize the Drive mount. This notebook reuses the specified prior two-pilot run and never regenerates a missing main arm.\n", "intro"),
            _cell("code", setup, "plan-init"), _cell("code", environment, "environment"),
            _cell("code", execute, "execute"), _cell("code", closing, "handoff"),
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "portable_sha256": digest, "portable_files": len(manifest["files"])}


if __name__ == "__main__":
    print(json.dumps(build_notebook(), indent=2))
