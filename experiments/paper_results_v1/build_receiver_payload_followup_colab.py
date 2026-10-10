"""Build the fixed receiver follow-up and its ordinary source companion."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/paper_results_v1_receiver_payload_followup_colab.ipynb"
ZIP = ROOT / "notebooks/paper_results_v1_receiver_payload_followup_companion.zip"


def companion_files():
    files = [ROOT / p for p in ("main/__init__.py", "runtime/__init__.py", "experiments/__init__.py",
        "experiments/paper_results_v1/__init__.py", "experiments/paper_results_v1/receiver_controls.py",
        "experiments/paper_results_v1/receiver_controls_cli.py", "experiments/paper_results_v1/attack_matrix.py",
        "experiments/paper_results_v1/receiver_payload_followup.py",
        "experiments/paper_results_v1/receiver_payload_followup_cli.py")]
    files += sorted((ROOT / "main/tube_state").rglob("*.py"))
    files += sorted((ROOT / "runtime/wan").rglob("*.py"))
    return sorted(set(files))


def cell(kind, source, ident):
    row = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True), id=ident)
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def build():
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in companion_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    setup = textwrap.dedent('''\
        import datetime, json, os, signal, subprocess, sys, time, urllib.request, uuid, zipfile
        from pathlib import Path
        SOURCE_REPOSITORY = "https://github.com/RICHAAARC/SC-SSTW"
        SOURCE_REF = "dev/paper-results-v1"
        ZIP_URL = SOURCE_REPOSITORY + "/raw/refs/heads/" + SOURCE_REF + "/notebooks/paper_results_v1_receiver_payload_followup_companion.zip"
        SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack-Recovery/20261010T063205774076Z-74da28d6")
        PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Receiver-Payload-Followup")
        POINTER = PARENT / "source-20261010T063205774076Z-74da28d6.json"
        PARENT.mkdir(parents=True, exist_ok=True)
        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\\n"); os.replace(temp, path)
        if POINTER.is_file():
            OUTPUT_ROOT = Path(json.loads(POINTER.read_text())["output_root"])
        else:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            OUTPUT_ROOT = PARENT / (stamp + "-" + uuid.uuid4().hex[:8])
            atomic_json(POINTER, {"output_root": str(OUTPUT_ROOT), "source_run": str(SOURCE_RUN)})
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_OUTPUT = OUTPUT_ROOT / "run_state"
        TEMP_ROOT = Path("/content") / ("receiver-followup-rgb-" + OUTPUT_ROOT.name)
        PORTABLE_ROOT = Path("/content/paper-results-v1-receiver-followup-source")
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        LOG = OUTPUT_ROOT / "execution.log"; RECEIPTS = OUTPUT_ROOT / "stage_receipts.json"
        receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.is_file() else []
        def record(stage, status, **fields):
            receipts.append({"stage": stage, "status": status, "time": time.time(), **fields})
            atomic_json(RECEIPTS, receipts)
        def logged(command, stage, *, check=True):
            child = None; rc = None; start = time.monotonic()
            print(stage, "log:", LOG, flush=True)
            try:
                with LOG.open("a") as stream:
                    stream.write("COMMAND " + repr(command) + "\\n"); stream.flush()
                    child = subprocess.Popen(command, cwd=PORTABLE_ROOT, env=ENV,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    rc = child.wait()
                record(stage, "COMPLETE" if rc == 0 else "FAILED", returncode=rc, seconds=time.monotonic()-start)
                if rc and check: raise RuntimeError(stage + " exited " + str(rc))
                return rc
            except BaseException as exc:
                record(stage, "FAILED", returncode=rc, reason=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if child is not None and child.poll() is None:
                    try: os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=5)
                    except BaseException:
                        try: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
                        except BaseException: pass
        required = PORTABLE_ROOT / "experiments/paper_results_v1/receiver_payload_followup_cli.py"
        if not required.is_file():
            archive = OUTPUT_ROOT / "companion_source.zip"
            urllib.request.urlretrieve(ZIP_URL, archive)
            PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                for item in zf.infolist():
                    target = (PORTABLE_ROOT / item.filename).resolve()
                    if PORTABLE_ROOT.resolve() not in target.parents:
                        raise ValueError("unsafe companion archive path")
                zf.extractall(PORTABLE_ROOT)
        if not required.is_file(): raise FileNotFoundError(required)
        ENV = {**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}
        CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.receiver_payload_followup_cli", "--output", str(RUN_OUTPUT)]
        if logged([sys.executable, "-c", "import numpy"], "CPU_IMPORT_PROBE", check=False):
            logged([sys.executable, "-m", "pip", "install", "numpy"], "CPU_IMPORT_REPAIR")
            logged([sys.executable, "-c", "import numpy"], "CPU_IMPORT_REPROBE")
        cfg = json.loads((SOURCE_RUN / "effective_attack_config.json").read_text())
        WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / WAN_REVISION
        cfg["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        CONFIG = OUTPUT_ROOT / "effective_followup_config.json"; atomic_json(CONFIG, cfg)
        logged([*CLI, "--source-run", str(SOURCE_RUN), "--config", str(CONFIG), "--temp-root", str(TEMP_ROOT), "--phase", "init"], "initialize")
        logged([*CLI, "--phase", "prepare"], "prepare-saved-observations", check=False)
        print("Fixed 90 rows; expected 34 reused and 30 new physical Wan inputs. Actual plan:")
        print((RUN_OUTPUT / "summary.json").read_text())
    ''')
    environment = textwrap.dedent('''\
        environment_status = "READY"
        try:
            probe = "import numpy,torch,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan; print('WAN_READ_IMPORT_READY')"
            if logged([sys.executable, "-c", probe], "WAN_IMPORT_PROBE", check=False):
                logged([sys.executable, "-m", "pip", "install", "torch", "numpy", "diffusers==0.39.0", "transformers==4.57.6", "accelerate", "safetensors", "huggingface-hub<1"], "WAN_IMPORT_FAILURE_REPAIR")
                logged([sys.executable, "-c", probe], "WAN_IMPORT_REPROBE")
            # This reuses the successful Wan cache, not a version admission gate.
            if not (WAN_SNAPSHOT / "vae/config.json").is_file() or not list((WAN_SNAPSHOT / "vae").glob("*.safetensors")):
                download = "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers', revision=" + repr(WAN_REVISION) + ", local_dir=" + repr(str(WAN_SNAPSHOT)) + ", allow_patterns=['model_index.json','vae/*'])"
                logged([sys.executable, "-c", download], "WAN_VAE_CACHE_PREPARATION")
            diagnostic_rc = logged([sys.executable, "-m", "pip", "check"], "PIP_CHECK_NONFATAL", check=False)
            record("environment", "READY", pip_check_nonfatal_returncode=diagnostic_rc)
        except Exception as exc:
            environment_status = "FAILED"
            record("environment", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        # No framewise model, baseline or quality dependencies are prepared.
    ''')
    execute = textwrap.dedent('''\
        attempts_path = OUTPUT_ROOT / "attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.is_file() else []
        caught = None
        try:
            rc = logged([*CLI, "--phase", "read"], "physical-wan-reads", check=False)
            attempts.append({"phase": "read", "returncode": rc}); atomic_json(attempts_path, attempts)
            if rc:
                logged([*CLI, "--phase", "record-failure", "--reason", "read subprocess exited " + str(rc)], "retain-child-failure", check=False)
        except BaseException as exc:
            caught = exc
            attempts.append({"phase": "read", "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", "reason": f"{type(exc).__name__}: {exc}"})
            atomic_json(attempts_path, attempts)
            try: logged([*CLI, "--phase", "record-failure", "--reason", f"{type(exc).__name__}: {exc}"], "retain-interruption", check=False)
            except BaseException: pass
        finally:
            try: logged([*CLI, "--phase", "report"], "final-report", check=False)
            except BaseException as exc: record("final-report", "FAILED", reason=str(exc))
            summary_path = RUN_OUTPUT / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"status": "MISSING_REPORT"}
            atomic_json(OUTPUT_ROOT / "handoff_summary.json", {"output_root": str(OUTPUT_ROOT), "source_run": str(SOURCE_RUN), "summary": summary, "environment_status": environment_status, "evidence_ceiling": "same-batch pilot development; confirmation NOT_EXECUTED"})
            print(json.dumps(summary, indent=2))
        if caught is not None: raise caught
    ''')
    intro = """# M05 receiver payload follow-up

Select a GPU runtime and Run all. Reuse the saved two-pilot recovery observations and original U reads. Compare RAW_U, CENTERED_U and CENTERED_D4 on all 30 conditions (90 rows); expected 34 reused rows and 30 new Wan encodes for 56 logical rows. Existing MP4s are decoded only when temporary RGB is gone. No generation, media publication, framewise, baseline or quality model is run. Later Run all resumes the same new Drive output; unfinished model attempts are retained without silent repeats. Real models/media were not executed while building this notebook.
"""
    handoff = "print('Return this directory:', OUTPUT_ROOT)\nprint('Include handoff_summary.json, execution.log, attempts.json, stage_receipts.json and all run_state files/sidecars.')\nprint('Same two-pilot development batch, not confirmation or wrong-key/OFF validation.')\n"
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python")),
        cells=[cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
            cell("markdown", intro, "intro"), cell("code", setup, "initialize-prepare"),
            cell("code", environment, "environment"), cell("code", execute, "execute-report"),
            cell("code", handoff, "handoff")])
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return dict(notebook=str(NOTEBOOK), companion=str(ZIP), files=len(companion_files()))


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
