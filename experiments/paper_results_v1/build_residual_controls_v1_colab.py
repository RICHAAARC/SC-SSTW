"""Build the fixed post-MP4 quality and receiver-control Run-all notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/paper_results_v1_residual_controls_v1_colab.ipynb"
ZIP = ROOT / "notebooks/paper_results_v1_residual_controls_v1_companion.zip"


def companion_files():
    files = [ROOT / path for path in (
        "main/__init__.py", "runtime/__init__.py", "experiments/__init__.py",
        "experiments/paper_results_v1/__init__.py", "experiments/paper_results_v1/resume_state.py",
        "experiments/paper_results_v1/attack_matrix.py", "experiments/paper_results_v1/native_adapters.py",
        "experiments/paper_results_v1/real_backends.py", "experiments/paper_results_v1/residual_joint_v1.py",
        "experiments/paper_results_v1/residual_controls_v1.py",
        "experiments/paper_results_v1/residual_controls_v1_cli.py",
    )]
    files += sorted((ROOT / "main/tube_state").rglob("*.py"))
    files += sorted((ROOT / "runtime/wan").rglob("*.py"))
    return sorted(set(files))


def cell(kind, source, ident):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": ident}
    if kind == "code": result.update(execution_count=None, outputs=[])
    return result


def build():
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in companion_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    setup = textwrap.dedent('''\
        import datetime, json, os, shutil, signal, subprocess, sys, time, urllib.request, uuid, zipfile
        from pathlib import Path

        SOURCE_REPOSITORY = "https://github.com/RICHAAARC/SC-SSTW"
        SOURCE_BRANCH = "dev/paper-results-v1"
        ZIP_URL = SOURCE_REPOSITORY + "/raw/refs/heads/" + SOURCE_BRANCH + "/notebooks/paper_results_v1_residual_controls_v1_companion.zip"
        SOURCE_JOINT_ID = "20261014T162818228961Z-dd0f70f0"
        SOURCE_JOINT_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Residual-Joint-V1") / SOURCE_JOINT_ID
        CHECKPOINT_ID = "20261012T032440550166Z-c3026cfb"
        CHECKPOINT_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1") / CHECKPOINT_ID
        PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Residual-Controls-V1")
        POINTER = PARENT / ("active-" + SOURCE_JOINT_ID + ".json")
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        PORTABLE_ROOT = None
        PARENT.mkdir(parents=True, exist_ok=True)

        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\\n")
            os.replace(temporary, path)

        def load_list_history(path, label):
            path = Path(path)
            if not path.is_file(): return []
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, list): raise ValueError(label + " must be a JSON list")
                return value
            except Exception as exc:
                backup = path.with_name(path.name + ".unreadable-" + str(time.time_ns()))
                os.replace(path, backup)
                return [{"stage": label + "-history", "status": "UNREADABLE_PRESERVED",
                         "time": time.time(), "backup": str(backup), "reason": f"{type(exc).__name__}: {exc}"}]

        if POINTER.is_file():
            OUTPUT_ROOT = Path(json.loads(POINTER.read_text())["output_root"])
        else:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            OUTPUT_ROOT = PARENT / (stamp + "-" + uuid.uuid4().hex[:8])
            atomic_json(POINTER, {"output_root": str(OUTPUT_ROOT), "source_joint_run": str(SOURCE_JOINT_RUN),
                                  "checkpoint_run": str(CHECKPOINT_RUN)})
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_OUTPUT = OUTPUT_ROOT / "run_state"
        LOG = OUTPUT_ROOT / "execution.log"
        RECEIPTS = OUTPUT_ROOT / "stage_receipts.json"
        receipts = load_list_history(RECEIPTS, "stage_receipts")

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
                if returncode and check: raise RuntimeError(stage + " exited " + str(returncode))
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

        archive = OUTPUT_ROOT / "companion_source.zip"
        downloaded = archive.with_name(archive.name + ".download")
        urllib.request.urlretrieve(ZIP_URL, downloaded)
        os.replace(downloaded, archive)
        staged_root = Path("/content") / ("paper-results-v1-residual-controls-source-" + uuid.uuid4().hex[:10])
        staged_root.mkdir(parents=True, exist_ok=False)
        required_members = {
            "experiments/paper_results_v1/residual_controls_v1.py",
            "experiments/paper_results_v1/residual_controls_v1_cli.py",
            "experiments/paper_results_v1/resume_state.py",
        }
        with zipfile.ZipFile(archive) as source_zip:
            members = {item.filename for item in source_zip.infolist() if not item.is_dir()}
            missing = sorted(required_members - members)
            if missing: raise FileNotFoundError("companion ZIP missing: " + ", ".join(missing))
            for item in source_zip.infolist():
                target = (staged_root / item.filename).resolve()
                if target != staged_root.resolve() and staged_root.resolve() not in target.parents:
                    raise ValueError("unsafe companion archive path")
            source_zip.extractall(staged_root)
        required = staged_root / "experiments/paper_results_v1/residual_controls_v1_cli.py"
        if not required.is_file(): raise FileNotFoundError(required)
        probe_env = {**os.environ, "PYTHONPATH": str(staged_root)}
        source_probe = subprocess.run([sys.executable, "-m", "experiments.paper_results_v1.residual_controls_v1_cli", "--help"],
                                      cwd=staged_root, env=probe_env, capture_output=True, text=True)
        if source_probe.returncode:
            raise RuntimeError("fresh companion CLI import failed: " + source_probe.stderr[-2000:])
        PORTABLE_ROOT = staged_root
        ENV = probe_env
        CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.residual_controls_v1_cli", "--output", str(RUN_OUTPUT)]
        record("portable-source", "COMPLETE", root=str(PORTABLE_ROOT), members=len(members))

        source_config = SOURCE_JOINT_RUN / "effective_residual_joint_config.json"
        config = json.loads(source_config.read_text())
        wan_revision = config["models"]["wan"].get("revision", "0fad780a534b6463e45facd96134c9f345acfa5b")
        framewise_revision = config["models"]["framewise"].get("revision", "31f26fdeee1355a5c34592e401dd41e45d25a493")
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / wan_revision
        FRAMEWISE_SNAPSHOT = CACHE_ROOT / "models/sd-vae-ft-mse" / framewise_revision
        config["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        config["models"]["framewise"]["local_snapshot_path"] = str(FRAMEWISE_SNAPSHOT)
        CONFIG = OUTPUT_ROOT / "effective_residual_controls_config.json"
        atomic_json(CONFIG, config)
        logged([*CLI, "--source-joint-run", str(SOURCE_JOINT_RUN), "--checkpoint-run", str(CHECKPOINT_RUN),
                "--config", str(CONFIG), "--phase", "init"], "initialize")

        cpu_probe = "import numpy,torch,PIL; print('RESIDUAL_CONTROLS_CPU_READY')"
        if logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_PROBE", check=False):
            logged([sys.executable, "-m", "pip", "install", "numpy", "torch", "pillow"], "CPU_IMPORT_REPAIR")
            logged([sys.executable, "-c", cpu_probe], "CPU_IMPORT_REPROBE")
        logged([*CLI, "--phase", "quality"], "post-mp4-quality", check=False)
        logged([*CLI, "--phase", "media"], "reuse-off-post-and-delete-g0", check=False)
        PENDING_MODEL_UNITS = json.loads((RUN_OUTPUT / "summary.json").read_text())["pending_model_units"]
        atomic_json(OUTPUT_ROOT / "pending_model_units.json", {"units": PENDING_MODEL_UNITS, "count": len(PENDING_MODEL_UNITS)})
        print("Fixed denominator: 24 rows / 768 bits (8 reused WM_K0 + 8 WM_K1 + 8 OFF_K0).")
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
                environment_status = "FAILED"; record("environment", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        else:
            record("environment", environment_status, pending_model_units=0)
    ''')

    execute = textwrap.dedent('''\
        attempts_path = OUTPUT_ROOT / "attempts.json"
        attempts = load_list_history(attempts_path, "attempts")
        caught = None
        for phase, stage in (("clock", "k1-and-off-clock"), ("wan", "operation-isolated-payload-controls")):
            try:
                code = logged([*CLI, "--phase", phase], stage, check=False)
                attempts.append({"phase": phase, "returncode": code, "time": time.time()}); atomic_json(attempts_path, attempts)
                if code:
                    logged([*CLI, "--phase", "record-failure", "--reason", stage + " subprocess exited " + str(code)],
                           "retain-" + phase + "-failure", check=False)
            except BaseException as exc:
                caught = caught or exc
                attempts.append({"phase": phase, "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                                 "reason": f"{type(exc).__name__}: {exc}", "time": time.time()}); atomic_json(attempts_path, attempts)
                try: logged([*CLI, "--phase", "record-failure", "--reason", f"{type(exc).__name__}: {exc}"],
                            "retain-" + phase + "-exception", check=False)
                except BaseException: pass
        logged([*CLI, "--phase", "report"], "final-report", check=False)
        summary_path = RUN_OUTPUT / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"status": "MISSING_REPORT"}
        atomic_json(OUTPUT_ROOT / "handoff_summary.json", {"output_root": str(OUTPUT_ROOT),
                    "source_joint_run": str(SOURCE_JOINT_RUN), "checkpoint_run": str(CHECKPOINT_RUN),
                    "summary": summary, "environment_status": environment_status,
                    "execution_boundary": "reuse 8 WM_K0; run 8 WM_K1 and 8 OFF_K0 controls; 24 CPU post-MP4 quality pairs; no generation/writer/codec/threshold/FPR"})
        print(json.dumps(summary, indent=2))
        if caught is not None: raise caught
    ''')

    intro = """# Paper Results V1 — post-MP4 quality and fixed receiver controls

Select a GPU runtime and **Run all**. This fixed follow-up reuses the completed residual-joint run and the original checkpoint. It performs no generation, writer reconstruction, image reconstruction, or codec call.

The denominator is 24 rows / 768 compared bits: eight already completed watermarked K0 rows, eight new watermarked K1 wrong-key controls, and eight new unwatermarked K0 controls. Each arm has the same four cases and CLEAN_MP4 / fixed DELETE_G0 conditions. K1 recomputes the key-dependent clock evidence and blind CENTERED_U operation from the saved receiver-visible framewise latent. K0 truth, selector state, and writer state are not K1 inputs. A saved K0 Wan latent is reused only if the actual received media is shared and the complete selected operation maps are equal; otherwise K1 receives a fresh Wan encode. OFF uses the saved OFF_NATIVE MP4 readback and applies the same RGB deletion without a second codec.

A targeted audit found no saved raw key-independent `[T,4,40,64]` OFF POST latent in the checkpoint or recovery records. Existing score sidecars are key-dependent, and PRE, other-attack, or Wan temporal-16-channel tensors are not equivalent inputs. The current budget therefore includes eight OFF framewise encodes / 176 batch-8 calls; once written, those durable tensors are reused across kernel restarts.

OFF has no inserted payload truth. Its comparison with the known probe bits is an absence/false-match control, and K1 is a wrong-key control. There is no presence threshold, detected-presence label, FPR estimate, or quality gate.

The CPU quality phase compares every actual watermarked received clip against three explicit references: CANDIDATE_PRE (codec-only delta), saved NATIVE_PRE (candidate plus codec delta), and saved NATIVE_POST (separate same-codec native baseline). DELETE_G0 applies the fixed deletion map to every reference. Each of the 24 pair records contains full-clip and every-frame rows plus four separately selected peaks: frame RMSE, frame maximum pixel/channel absolute error, transition RMSE, and transition maximum pixel/channel residual-delta. Every peak saves its neighborhood and actual candidate/reference/absolute-difference frame and crop triptychs. Difference display is fixed to 0..255 with no auto-normalization, and none of these values is a quality gate.

All 24 receiver rows and 24 quality slots are materialized before source rows or OFF metadata are joined. Missing or unreadable source inputs remain fixed failed slots. Probe-bit and clock-truth comparisons remain null/NOT_JOINED until both clock and Wan phases are complete and no model unit remains pending; final K0 payload recovery and K1/OFF false-target matches are reported separately from descriptive post-hoc clock-path equality. Each Run all downloads and probes a fresh ordinary companion ZIP extraction. Unreadable auxiliary receipt or attempt JSON is preserved under a `.unreadable-*` name and replaced without blocking resume.

When all 16 new blind operations are supported, an uninterrupted incremental run has four CPU OFF delete edits, one framewise load and eight new framewise clips (176 batch8 batches), 16 new CENTERED_U selections, at most one Wan load, at most 16 new Wan clip encodes, and 16 new payload reads. Equal complete K1/K0 corrected-RGB operations can reuse the corresponding readable K0 Wan latent. Unsupported clocks, unsupported visible spans, failures, and missing invalid cached tensors reduce actual completed calls without changing or replacing the fixed rows. The journal saves committed state every 60 seconds and Run all resumes the same Drive task.
"""

    handoff = textwrap.dedent('''\
        print("Return this directory:", OUTPUT_ROOT)
        print("Include handoff_summary.json, execution.log, attempts.json, stage_receipts.json, and the complete run_state directory.")
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
                          cell("markdown", intro, "intro"), cell("code", setup, "initialize-cpu"),
                          cell("code", environment, "environment"), cell("code", execute, "execute-report"),
                          cell("code", handoff, "handoff")]}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"notebook": str(NOTEBOOK), "companion": str(ZIP), "files": len(companion_files())}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
