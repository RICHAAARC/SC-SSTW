"""Build the fixed saved-residual clean-MP4/delete_g0 joint Run-all notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/paper_results_v1_residual_joint_v1_colab.ipynb"
ZIP = ROOT / "notebooks/paper_results_v1_residual_joint_v1_companion.zip"


def companion_files():
    files = [ROOT / path for path in (
        "main/__init__.py", "runtime/__init__.py", "experiments/__init__.py",
        "experiments/paper_results_v1/__init__.py",
        "experiments/paper_results_v1/resume_state.py",
        "experiments/paper_results_v1/attack_matrix.py",
        "experiments/paper_results_v1/native_adapters.py",
        "experiments/paper_results_v1/real_backends.py",
        "experiments/paper_results_v1/residual_joint_v1.py",
        "experiments/paper_results_v1/residual_joint_v1_cli.py",
    )]
    files += sorted((ROOT / "main/tube_state").rglob("*.py"))
    files += sorted((ROOT / "runtime/wan").rglob("*.py"))
    return sorted(set(files))


def cell(kind, source, ident):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": ident}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in companion_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    setup = textwrap.dedent('''\
        import datetime, json, os, shutil, signal, subprocess, sys, time, urllib.request, uuid, zipfile
        from pathlib import Path

        SOURCE_REPOSITORY = "https://github.com/RICHAAARC/SC-SSTW"
        SOURCE_BRANCH = "dev/paper-results-v1"
        ZIP_URL = SOURCE_REPOSITORY + "/raw/refs/heads/" + SOURCE_BRANCH + "/notebooks/paper_results_v1_residual_joint_v1_companion.zip"
        SOURCE_RUN_ID = "20261014T110044944207Z-c569a935"
        SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Residual-Clock-V1") / SOURCE_RUN_ID
        PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Residual-Joint-V1")
        POINTER = PARENT / ("active-" + SOURCE_RUN_ID + ".json")
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        PORTABLE_ROOT = Path("/content/paper-results-v1-residual-joint-source")
        PARENT.mkdir(parents=True, exist_ok=True)

        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\\n")
            os.replace(temporary, path)

        if POINTER.is_file():
            OUTPUT_ROOT = Path(json.loads(POINTER.read_text())["output_root"])
        else:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            OUTPUT_ROOT = PARENT / (stamp + "-" + uuid.uuid4().hex[:8])
            atomic_json(POINTER, {"output_root": str(OUTPUT_ROOT), "source_run": str(SOURCE_RUN)})
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_OUTPUT = OUTPUT_ROOT / "run_state"
        LOG = OUTPUT_ROOT / "execution.log"
        RECEIPTS = OUTPUT_ROOT / "stage_receipts.json"
        receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.is_file() else []

        def record(stage, status, **fields):
            receipts.append({"stage": stage, "status": status, "time": time.time(), **fields})
            atomic_json(RECEIPTS, receipts)

        def logged(command, stage, *, check=True):
            child = None; returncode = None; started = time.monotonic()
            print(stage, "log:", LOG, flush=True)
            try:
                with LOG.open("a") as stream:
                    stream.write("COMMAND " + repr(command) + "\\n"); stream.flush()
                    child = subprocess.Popen(command, cwd=PORTABLE_ROOT, env=ENV, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    returncode = child.wait()
                record(stage, "COMPLETE" if returncode == 0 else "FAILED", returncode=returncode,
                       seconds=time.monotonic() - started)
                if returncode and check:
                    raise RuntimeError(stage + " exited " + str(returncode))
                return returncode
            except BaseException as exc:
                record(stage, "FAILED", returncode=returncode, reason=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if child is not None and child.poll() is None:
                    try: os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=5)
                    except BaseException:
                        try: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
                        except BaseException: pass

        required = PORTABLE_ROOT / "experiments/paper_results_v1/residual_joint_v1_cli.py"
        if not required.is_file():
            archive = OUTPUT_ROOT / "companion_source.zip"
            urllib.request.urlretrieve(ZIP_URL, archive)
            PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as source_zip:
                for item in source_zip.infolist():
                    target = (PORTABLE_ROOT / item.filename).resolve()
                    if PORTABLE_ROOT.resolve() not in target.parents:
                        raise ValueError("unsafe companion archive path")
                source_zip.extractall(PORTABLE_ROOT)
        if not required.is_file():
            raise FileNotFoundError(required)
        ENV = {**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}
        CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.residual_joint_v1_cli", "--output", str(RUN_OUTPUT)]

        source_config = SOURCE_RUN / "effective_residual_clock_config.json"
        config = json.loads(source_config.read_text())
        wan_revision = config["models"]["wan"].get("revision", "0fad780a534b6463e45facd96134c9f345acfa5b")
        framewise_revision = config["models"]["framewise"].get("revision", "31f26fdeee1355a5c34592e401dd41e45d25a493")
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / wan_revision
        FRAMEWISE_SNAPSHOT = CACHE_ROOT / "models/sd-vae-ft-mse" / framewise_revision
        config["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        config["models"]["framewise"]["local_snapshot_path"] = str(FRAMEWISE_SNAPSHOT)
        CONFIG = OUTPUT_ROOT / "effective_residual_joint_config.json"
        atomic_json(CONFIG, config)
        logged([*CLI, "--source-run", str(SOURCE_RUN), "--config", str(CONFIG), "--phase", "init"], "initialize")

        if any(shutil.which(name) is None for name in ("ffmpeg", "ffprobe")):
            logged(["apt-get", "update"], "MEDIA_TOOL_UPDATE", check=False)
            logged(["apt-get", "install", "-y", "ffmpeg"], "MEDIA_TOOL_REPAIR")
        cpu_probe = "import numpy,torch; from PIL import Image; print('RESIDUAL_JOINT_CPU_READY')"
        if logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_PROBE", check=False):
            logged([sys.executable, "-m", "pip", "install", "numpy", "torch", "pillow"], "CPU_IMPORT_REPAIR")
            logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_REPROBE")
        logged([*CLI, "--phase", "quality"], "saved-rgb-peak-quality", check=False)
        logged([*CLI, "--phase", "media"], "fixed-codec-and-delete-g0", check=False)
        PENDING_MODEL_UNITS = json.loads((RUN_OUTPUT / "summary.json").read_text())["pending_model_units"]
        atomic_json(OUTPUT_ROOT / "pending_model_units.json", {"units": PENDING_MODEL_UNITS, "count": len(PENDING_MODEL_UNITS)})
        print("Fixed denominator: four clean-MP4 joint rows plus four delete_g0 joint rows (256 bits).")
        print("Pending model units:", len(PENDING_MODEL_UNITS))
    ''')

    environment = textwrap.dedent('''\
        environment_status = "SKIPPED_NO_PENDING_MODEL_UNIT"
        if PENDING_MODEL_UNITS:
            environment_status = "READY"
            try:
                probe = "import numpy,torch,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan,AutoencoderKL; print('MODEL_READ_READY')"
                if logged([sys.executable, "-c", probe], "MODEL_IMPORT_PROBE", check=False):
                    logged([sys.executable, "-m", "pip", "install", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub"], "MODEL_IMPORT_REPAIR")
                    logged([sys.executable, "-c", probe], "MODEL_IMPORT_REPROBE")
                if not (WAN_SNAPSHOT / "vae/config.json").is_file() or not list((WAN_SNAPSHOT / "vae").glob("*.safetensors")):
                    upstream = config["models"]["wan"].get("upstream_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
                    command = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=" + repr(upstream) + ", revision=" + repr(wan_revision) + ", local_dir=" + repr(str(WAN_SNAPSHOT)) + ", allow_patterns=['model_index.json','vae/*'])"
                    logged([sys.executable, "-c", command], "WAN_VAE_CACHE_PREPARATION")
                if not (FRAMEWISE_SNAPSHOT / "config.json").is_file() or not list(FRAMEWISE_SNAPSHOT.glob("*.safetensors")):
                    upstream = config["models"]["framewise"].get("upstream_id", "stabilityai/sd-vae-ft-mse")
                    command = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=" + repr(upstream) + ", revision=" + repr(framewise_revision) + ", local_dir=" + repr(str(FRAMEWISE_SNAPSHOT)) + ", allow_patterns=['config.json','*.safetensors'])"
                    logged([sys.executable, "-c", command], "FRAMEWISE_VAE_CACHE_PREPARATION")
                record("environment", "READY")
            except Exception as exc:
                environment_status = "FAILED"
                record("environment", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        else:
            record("environment", environment_status, pending_model_units=0)
    ''')

    quality_display = textwrap.dedent('''\
        from IPython.display import Image as NotebookImage, display
        quality_state = json.loads((RUN_OUTPUT / "residual_joint_state.json").read_text()).get("quality_inspection", {})
        print(json.dumps(quality_state, indent=2))
        contact_sheet = quality_state.get("contact_sheet")
        if contact_sheet and Path(contact_sheet).is_file():
            display(NotebookImage(filename=contact_sheet))
    ''')

    execute = textwrap.dedent('''\
        attempts_path = OUTPUT_ROOT / "attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.is_file() else []
        caught = None
        for phase, stage in (("clock", "framewise-centered-u"), ("wan", "wan-payload-read")):
            try:
                code = logged([*CLI, "--phase", phase], stage, check=False)
                attempts.append({"phase": phase, "returncode": code, "time": time.time()})
                atomic_json(attempts_path, attempts)
                if code:
                    logged([*CLI, "--phase", "record-failure", "--reason", stage + " subprocess exited " + str(code)],
                           "retain-" + phase + "-failure", check=False)
            except BaseException as exc:
                caught = caught or exc
                attempts.append({"phase": phase, "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                                 "reason": f"{type(exc).__name__}: {exc}", "time": time.time()})
                atomic_json(attempts_path, attempts)
                try:
                    logged([*CLI, "--phase", "record-failure", "--reason", f"{type(exc).__name__}: {exc}"],
                           "retain-" + phase + "-exception", check=False)
                except BaseException: pass
        try:
            logged([*CLI, "--phase", "report"], "final-report", check=False)
        except BaseException as exc:
            record("final-report", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        summary_path = RUN_OUTPUT / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"status": "MISSING_REPORT"}
        atomic_json(OUTPUT_ROOT / "handoff_summary.json", {
            "output_root": str(OUTPUT_ROOT), "source_run": str(SOURCE_RUN),
            "summary": summary, "environment_status": environment_status,
            "execution_boundary": "four saved residual candidates; 4 fixed MP4 publications; same-MP4 delete_g0; 8 blind CENTERED_U joint rows; no writer/generation/Wan decode/VJP/scan",
        })
        print(json.dumps(summary, indent=2))
        if caught is not None:
            raise caught
    ''')

    intro = """# Paper Results V1 — saved residual joint MP4/delete_g0 validation

Select a GPU runtime and **Run all**. This fixed task reuses only the four saved `residual_clock_pre.rgb8` candidates from run `20261014T110044944207Z-c569a935`; it does not reconstruct or retune the writer. Each candidate receives one publication with the adopted libx264/8 fps/CRF18/yuv420p transport. The clean decoded RGB produces one joint row. The fixed `delete_g0` edit is applied to that same decoded MP4 RGB and produces one attacked joint row. There is no second codec after the deletion.

Both groups use the real received RGB as the only media input. Each row performs a fresh framewise encode, `CENTERED_U` blind path selection from received evidence plus K0 and the public protocol, `decode_visible_span`, then a Wan VAE encode and the unchanged payload reader. The saved delete map and payload answer do not select, correct, retry, or exclude a row. They join only after blind selection and payload read are durable, for path-error and bit-error reporting.

Before codec execution, a bounded CPU-only quality cell rereads the saved candidate and NATIVE_PRE frames in the recorded spatial-peak, temporal-peak, and frame-179/180 neighborhoods. It saves per-frame changed-pixel extent and new boundary-value channel-fraction rows and displays a native/candidate/amplified-difference contact sheet. The boundary-value measure counts candidate channels newly equal to 0 or 255 relative to NATIVE_PRE; it is not an exact clipping measurement. This is an inspection artifact with no quality threshold and cannot block, select, or remove a joint row.

The fixed denominator is eight rows / 256 payload bits: clean MP4 4×32 and delete_g0 4×32. Clock failure, unsupported visible span, model failure, missing input, and payload failure remain in those eight rows. There is no NO_SYNC group in this minimal joint validation. An uninterrupted complete run has four codec encodes plus four readbacks, four CPU delete edits, one framewise load and eight framewise clips (176 batch8 batches: 92 clean + 84 delete), eight `CENTERED_U` selections, and one Wan load with at most eight Wan clip encodes. Interrupted unfinished units may repeat; every actual attempt remains recorded. There is no generation/DiT, writer reconstruction, Wan decode, VJP, new sample, attack scan, threshold, FPR claim, or scientific PASS. No real model, GPU, codec, Colab, or Drive task was executed while building this notebook.
"""

    handoff = textwrap.dedent('''\
        print("Return this directory:", OUTPUT_ROOT)
        print("Include handoff_summary.json, execution.log, attempts.json, stage_receipts.json, and the complete run_state directory.")
        print("Keep clean MP4 and delete_g0 as separate fixed four-row reports; both are the new saved residual candidate.")
    ''')
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [
            cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
            cell("markdown", intro, "intro"),
            cell("code", setup, "initialize-media"),
            cell("code", quality_display, "peak-quality-display"),
            cell("code", environment, "environment"),
            cell("code", execute, "execute-report"),
            cell("code", handoff, "handoff"),
        ],
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"notebook": str(NOTEBOOK), "companion": str(ZIP), "files": len(companion_files())}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
