"""Build the fixed four-source layer-readout notebook and source companion."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/paper_results_v1_layer_readout_v1_colab.ipynb"
ZIP = ROOT / "notebooks/paper_results_v1_layer_readout_v1_companion.zip"


def companion_files():
    files = [ROOT / path for path in (
        "main/__init__.py", "runtime/__init__.py", "experiments/__init__.py",
        "experiments/paper_results_v1/__init__.py", "experiments/paper_results_v1/resume_state.py",
        "experiments/paper_results_v1/layer_readout_v1.py",
        "experiments/paper_results_v1/layer_readout_v1_cli.py",
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
        import datetime, json, os, signal, subprocess, sys, time, urllib.request, uuid, zipfile
        from pathlib import Path

        SOURCE_REPOSITORY = "https://github.com/RICHAAARC/SC-SSTW"
        SOURCE_REF = "dev/paper-results-v1"
        ZIP_URL = SOURCE_REPOSITORY + "/raw/refs/heads/" + SOURCE_REF + "/notebooks/paper_results_v1_layer_readout_v1_companion.zip"
        SOURCE_RUN_ID = "20261012T032440550166Z-c3026cfb"
        SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1") / SOURCE_RUN_ID
        PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Layer-Readout-V1")
        POINTER = PARENT / ("source-" + SOURCE_RUN_ID + ".json")
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        PORTABLE_ROOT = Path("/content/paper-results-v1-layer-readout-source")
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
                    child = subprocess.Popen(command, cwd=PORTABLE_ROOT, env=ENV,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    returncode = child.wait()
                record(stage, "COMPLETE" if returncode == 0 else "FAILED",
                       returncode=returncode, seconds=time.monotonic() - started)
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

        required = PORTABLE_ROOT / "experiments/paper_results_v1/layer_readout_v1_cli.py"
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
        CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.layer_readout_v1_cli", "--output", str(RUN_OUTPUT)]

        source_config = SOURCE_RUN / "effective_confirmation_config.json"
        config = json.loads(source_config.read_text())
        wan = config["models"]["wan"]
        revision = wan.get("revision", "0fad780a534b6463e45facd96134c9f345acfa5b")
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / revision
        wan["local_snapshot_path"] = str(WAN_SNAPSHOT)
        CONFIG = OUTPUT_ROOT / "effective_layer_readout_config.json"
        atomic_json(CONFIG, config)
        logged([*CLI, "--source-run", str(SOURCE_RUN), "--config", str(CONFIG), "--phase", "init"], "initialize")
        cpu_probe = "import numpy,torch; print('CPU_FIXED_READ_READY')"
        if logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_PROBE", check=False):
            logged([sys.executable, "-m", "pip", "install", "numpy", "torch"], "CPU_IMPORT_REPAIR")
            logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_REPROBE")
        # CPU-only reuse: saved writer terminal and saved M05 POST identity vote sidecar.
        logged([*CLI, "--phase", "reuse"], "reuse-saved-fixed-reads", check=False)
        PENDING_MODEL_ROWS = json.loads((RUN_OUTPUT / "summary.json").read_text())["pending_model_rows"]
        atomic_json(OUTPUT_ROOT / "pending_model_rows.json", {"slot_ids": PENDING_MODEL_ROWS, "count": len(PENDING_MODEL_ROWS)})
        print("Fixed denominator: 4 sources x 5 layers = 20 rows; planned new Wan encodes: 12.")
        print("Physical Wan encode rows still pending:", len(PENDING_MODEL_ROWS))
        print((RUN_OUTPUT / "summary.json").read_text())
    ''')

    environment = textwrap.dedent('''\
        if not PENDING_MODEL_ROWS:
            environment_status = "SKIPPED_NO_PENDING_ENCODE"
            record("environment", environment_status, pending_model_rows=0)
            print("Wan VAE dependency probe and cache preparation skipped: no physical encode remains.")
        else:
            environment_status = "READY"
            try:
                probe = "import numpy,torch,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan; print('WAN_VAE_READ_READY')"
                if logged([sys.executable, "-c", probe], "WAN_IMPORT_PROBE", check=False):
                    logged([sys.executable, "-m", "pip", "install", "torch", "numpy", "diffusers==0.39.0", "transformers==4.57.6", "accelerate", "safetensors", "huggingface-hub<1"], "WAN_IMPORT_REPAIR")
                    logged([sys.executable, "-c", probe], "WAN_IMPORT_REPROBE")
                if not (WAN_SNAPSHOT / "vae/config.json").is_file() or not list((WAN_SNAPSHOT / "vae").glob("*.safetensors")):
                    upstream = config["models"]["wan"].get("upstream_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
                    download = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=" + repr(upstream) + ", revision=" + repr(revision) + ", local_dir=" + repr(str(WAN_SNAPSHOT)) + ", allow_patterns=['model_index.json','vae/*'])"
                    logged([sys.executable, "-c", download], "WAN_VAE_CACHE_PREPARATION")
                pip_check = logged([sys.executable, "-m", "pip", "check"], "PIP_CHECK_NONFATAL", check=False)
                record("environment", "READY", pip_check_nonfatal_returncode=pip_check)
            except Exception as exc:
                environment_status = "FAILED"
                record("environment", "FAILED", reason=f"{type(exc).__name__}: {exc}")
    ''')

    execute = textwrap.dedent('''\
        attempts_path = OUTPUT_ROOT / "attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.is_file() else []
        caught = None
        try:
            returncode = logged([*CLI, "--phase", "read"], "fixed-layer-wan-encodes", check=False)
            attempts.append({"phase": "read", "returncode": returncode, "time": time.time()})
            atomic_json(attempts_path, attempts)
            if returncode:
                logged([*CLI, "--phase", "record-failure", "--reason", "read subprocess exited " + str(returncode)], "retain-child-failure", check=False)
        except BaseException as exc:
            caught = exc
            attempts.append({"phase": "read", "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", "reason": f"{type(exc).__name__}: {exc}", "time": time.time()})
            atomic_json(attempts_path, attempts)
            try: logged([*CLI, "--phase", "record-failure", "--reason", f"{type(exc).__name__}: {exc}"], "retain-interruption", check=False)
            except BaseException: pass
        finally:
            try: logged([*CLI, "--phase", "report"], "final-report", check=False)
            except BaseException as exc: record("final-report", "FAILED", reason=f"{type(exc).__name__}: {exc}")
            summary_path = RUN_OUTPUT / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"status": "MISSING_REPORT"}
            atomic_json(OUTPUT_ROOT / "handoff_summary.json", {
                "output_root": str(OUTPUT_ROOT), "source_run": str(SOURCE_RUN), "summary": summary,
                "environment_status": environment_status,
                "execution_boundary": "user-run GPU layer readout; no generation, decode, VJP, framewise model, or codec",
            })
            print(json.dumps(summary, indent=2))
        if caught is not None:
            raise caught
    ''')

    intro = """# Paper Results V1 — fixed layer readout

Select a GPU runtime and **Run all**. This diagnostic reads four fixed sources (`confirm_01`, `confirm_02`, `confirm_07`, and successful control `confirm_03`) at five fixed layers using K0, R44, and identity coordinates 0..180. It reuses saved writer-terminal and M05-POST votes, then performs exactly the unfinished subset of 12 planned Wan VAE encodes for saved NATIVE-PRE, RECON-PRE, and M05-PRE RGB8 artifacts.

One successful uninterrupted read process loads one Wan VAE and makes 12 encode forwards. It performs zero Wan decode, VJP, generation/DiT, framewise-model load, codec, or media reconstruction calls. Each encode commits its normalized latent before vote/report work; later Run all resumes from durable latents and skips complete rows. All 20 rows, including missing and failed rows, remain in the report. Payload truth is joined only after the fixed reads for evaluation. The GPU experiment was not run while this notebook was built.
"""
    handoff = textwrap.dedent('''\
        print("Return this directory:", OUTPUT_ROOT)
        print("Include handoff_summary.json, execution.log, attempts.json, stage_receipts.json, and the complete run_state directory.")
        print("The fixed report contains 20 layer rows and retains every missing or failed row.")
    ''')
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [
            cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
            cell("markdown", intro, "intro"),
            cell("code", setup, "initialize-reuse"),
            cell("code", environment, "environment"),
            cell("code", execute, "execute-report"),
            cell("code", handoff, "handoff"),
        ],
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"notebook": str(NOTEBOOK), "companion": str(ZIP), "files": len(companion_files())}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
