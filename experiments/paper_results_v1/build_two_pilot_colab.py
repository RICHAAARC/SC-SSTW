"""Build the self-contained, fixed two-pilot Paper Results V1 Colab notebook.

The generated notebook embeds the exact outer evaluator plus its main/runtime
closure.  It intentionally performs no model import or download while being
built.  Colab execution records the embedded content manifest as the source
identity; a Git checkout is not required.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/paper_results_v1_two_pilot_colab.ipynb"


def _portable_files():
    paths = [
        ROOT / "experiments/__init__.py",
        ROOT / "main/__init__.py",
        ROOT / "runtime/__init__.py",
        ROOT / "experiments/paper_results_v1/real_eval.adopted.json",
        ROOT / "experiments/paper_results_v1/source_identity_audit.json",
    ]
    paths.extend(sorted((ROOT / "experiments/paper_results_v1").glob("*.py")))
    paths.extend(sorted((ROOT / "main/tube_state").rglob("*.py")))
    paths.extend(sorted((ROOT / "runtime/wan").rglob("*.py")))
    unique = {path.relative_to(ROOT).as_posix(): path for path in paths}
    return [(name, unique[name]) for name in sorted(unique)]


def _git_head():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _payload():
    files = _portable_files()
    manifest = {
        "schema_version": "paper-results-v1-portable-source-v1",
        "execution_identity": "ZIP_SHA256_PLUS_PER_FILE_SHA256",
        "built_from_git_head_context_only": _git_head(),
        "files": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for name, path in files
        },
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("portable_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    value = stream.getvalue()
    return base64.b64encode(value).decode(), hashlib.sha256(value).hexdigest(), manifest


def _source(value):
    return value.splitlines(keepends=True)


def _cell(cell_type, source, cell_id):
    cell = {"cell_type": cell_type, "metadata": {}, "source": _source(source), "id": cell_id}
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook():
    payload_b64, payload_sha, manifest = _payload()
    setup = textwrap.dedent(f'''\
        # Fixed scope and self-contained source identity. This cell imports only the standard library.
        from pathlib import Path
        import base64, datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback, urllib.request, uuid, zipfile

        PILOT_CASES = ("pilot_01", "pilot_02")
        CONFIRMATION_CASES = ("confirm_01", "confirm_02", "confirm_03", "confirm_04", "confirm_05", "confirm_06", "confirm_07", "confirm_08")
        PHASE_ORDER = (
            "generate", "decode", "framewise", "baseline-embed-videoseal",
            "baseline-embed-rivagan", "codec", "quality", "baseline-extract-videoseal",
            "baseline-extract-rivagan", "receiver-sync", "receiver-read",
        )
        PORTABLE_ZIP_SHA256 = "{payload_sha}"
        PORTABLE_B64 = "{payload_b64}"
        PORTABLE_BASE_COMMIT_CONTEXT = {manifest['built_from_git_head_context_only']!r}
        FIXED_FULL_DENOMINATOR = {{
            "cases": 10, "artifacts": 690, "receiver_slots": 1600,
            "baseline_slots": 180, "comparison_slots": 180,
            "quality_rows": 70, "cost_rows": 110,
        }}
        FIXED_ATTEMPT_SCOPE = {{
            "cases": 2, "case_ids": list(PILOT_CASES), "receiver_slots": 320,
            "baseline_slots": 36, "comparison_slots": 36,
            "quality_rows": 14, "cost_rows": 22,
            "trajectories": 4, "scheduler_steps": 200,
            "conditional_unconditional_transformer_forwards": 400,
            "framewise_writer_encodes": 2, "framewise_decodes": 4,
            "full_video_codec_roundtrips": 12,
            "receiver_framewise_sync_encodes": 72, "logical_wan_reads": 320,
            "wan_physical_encode_upper_bound": 240,
            "pre_post_rgb8_raster_bytes": 2135162880,
        }}
        STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        RUN_ID = STAMP + "-" + uuid.uuid4().hex[:8]
        OUTPUT_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Two-Pilot") / RUN_ID
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        RUN_OUTPUT = OUTPUT_ROOT / "run_state"
        PORTABLE_ROOT = Path("/content") / ("paper-results-v1-portable-" + RUN_ID)
        CACHE_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_PATH = OUTPUT_ROOT / "execution.log"
        STAGE_RECEIPTS = OUTPUT_ROOT / "stage_receipts.json"
        PYTHON = sys.executable

        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\\n", encoding="utf-8")
            os.replace(temporary, path)

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        stage_receipts = []
        def record_stage(stage, status, **fields):
            row = {{"stage": stage, "status": status, "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}}
            stage_receipts.append(row); atomic_json(STAGE_RECEIPTS, stage_receipts); return row

        def logged(command, stage, *, cwd=None, env=None, check=True):
            started = time.monotonic(); child = None; returncode = None; primary = None; primary_tb = None; cleanup_errors = []
            print(f"[{{stage}}] start; full output: {{LOG_PATH}}", flush=True)
            try:
                with LOG_PATH.open("a", encoding="utf-8") as log:
                    log.write("COMMAND " + repr(command) + "\\n"); log.flush()
                    child = subprocess.Popen(
                        command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                        text=True, start_new_session=(os.name == "posix"),
                    )
                    try:
                        returncode = child.wait()
                    except BaseException as exc:
                        primary = exc; primary_tb = exc.__traceback__
                    finally:
                        if child is not None and os.name == "posix":
                            try: os.killpg(child.pid, signal.SIGTERM)
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_errors.append("group SIGTERM: " + repr(exc))
                            deadline = time.monotonic() + 5.0
                            while time.monotonic() < deadline:
                                try: os.killpg(child.pid, 0)
                                except ProcessLookupError: break
                                except BaseException as exc:
                                    cleanup_errors.append("group probe: " + repr(exc)); break
                                time.sleep(0.05)
                            try: os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_errors.append("group SIGKILL: " + repr(exc))
                        elif child is not None and child.poll() is None:
                            try: child.terminate()
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_errors.append("terminate: " + repr(exc))
                        if child is not None:
                            try: returncode = child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                try: child.kill()
                                except ProcessLookupError: pass
                                except BaseException as exc: cleanup_errors.append("kill: " + repr(exc))
                                try: returncode = child.wait(timeout=10)
                                except BaseException as exc: cleanup_errors.append("wait after kill: " + repr(exc))
                            except BaseException as exc: cleanup_errors.append("wait: " + repr(exc))
            except BaseException as exc:
                if primary is None:
                    primary = exc; primary_tb = exc.__traceback__
                else:
                    cleanup_errors.append("log wrapper: " + repr(exc))
            if primary is None and cleanup_errors:
                primary = RuntimeError("process cleanup failed: " + "; ".join(cleanup_errors)); primary_tb = primary.__traceback__
            if primary is None and check and returncode:
                primary = subprocess.CalledProcessError(returncode, command); primary_tb = primary.__traceback__
            if primary is not None:
                record_stage(stage, "FAILED", returncode=returncode, seconds=time.monotonic()-started, command=command, reason=f"{{type(primary).__name__}}: {{primary}}", cleanup_errors=cleanup_errors)
                print(f"[{{stage}}] failed; full output: {{LOG_PATH}}", flush=True)
                raise primary.with_traceback(primary_tb)
            record_stage(stage, "SUCCEEDED" if returncode == 0 else "FAILED", returncode=returncode, seconds=time.monotonic()-started, command=command, cleanup_errors=cleanup_errors)
            print(f"[{{stage}}] returncode={{returncode}}; full output: {{LOG_PATH}}", flush=True)
            return returncode

        def extract_portable_source():
            raw = base64.b64decode(PORTABLE_B64.encode())
            if hashlib.sha256(raw).hexdigest() != PORTABLE_ZIP_SHA256:
                raise RuntimeError("embedded portable source ZIP digest mismatch")
            archive_path = OUTPUT_ROOT / "portable_source.zip"
            archive_path.write_bytes(raw)
            PORTABLE_ROOT.mkdir(parents=True, exist_ok=False)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (PORTABLE_ROOT / member.filename).resolve()
                    if PORTABLE_ROOT.resolve() not in target.parents and target != PORTABLE_ROOT.resolve():
                        raise RuntimeError("unsafe portable archive member")
                    archive.extract(member, PORTABLE_ROOT)
            manifest_path = PORTABLE_ROOT / "portable_manifest.json"
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, receipt in source_manifest["files"].items():
                path = PORTABLE_ROOT / name
                if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                    raise RuntimeError("portable source file mismatch: " + name)
            if any(".git" in path.parts for path in PORTABLE_ROOT.rglob("*")):
                raise RuntimeError("portable source unexpectedly contains .git")
            atomic_json(OUTPUT_ROOT / "portable_source_receipt.json", {{
                "zip_sha256": PORTABLE_ZIP_SHA256,
                "manifest_sha256": sha256_file(manifest_path),
                "built_from_git_head_context_only": PORTABLE_BASE_COMMIT_CONTEXT,
                "execution_identity": "EMBEDDED_ZIP_AND_PER_FILE_CONTENT_DIGESTS",
                "file_count": len(source_manifest["files"]),
            }})
            return source_manifest

        try:
            SOURCE_MANIFEST = extract_portable_source()
            BASE_CONFIG = PORTABLE_ROOT / "experiments/paper_results_v1/real_eval.adopted.json"
            BOOTSTRAP_PLAN = OUTPUT_ROOT / "bootstrap_plan"
            portable_env = {{**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}}
            logged([PYTHON, "-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(BASE_CONFIG), "--output", str(BOOTSTRAP_PLAN), "--phase", "plan"], "BOOTSTRAP_FIXED_PLAN", env=portable_env)
            atomic_json(OUTPUT_ROOT / "execution_scope.json", {{
                "status": "FIXED_TWO_PILOT_ATTEMPT_SCOPE",
                "attempted_case_ids": list(PILOT_CASES),
                "confirmation_case_ids": list(CONFIRMATION_CASES),
                "confirmation_status": "NOT_EXECUTED_BY_NOTEBOOK",
                "reason": "Run all is fixed to the two adopted pilot cases. Full-denominator evaluation later projects absent confirmation receiver/baseline/comparison evidence to failure/unevaluable rows while quality retains its recorded state; none are attempted runs.",
                "full_denominator": FIXED_FULL_DENOMINATOR,
                "attempt_scope": FIXED_ATTEMPT_SCOPE,
            }})
            record_stage("PORTABLE_SOURCE_AND_BOOTSTRAP_PLAN", "SUCCEEDED")
        except Exception as exc:
            atomic_json(OUTPUT_ROOT / "bootstrap_failure.json", {{"status": "FAILED", "reason": f"{{type(exc).__name__}}: {{exc}}", "traceback": traceback.format_exc(), "full_denominator": FIXED_FULL_DENOMINATOR}})
            raise
        print("Run:", RUN_ID, "portable source:", PORTABLE_ZIP_SHA256)
        print("Fixed execution scope:", PILOT_CASES, "; confirmation is not executed by this notebook.")
    ''')

    prepare_identity = textwrap.dedent('''\
        ## Prepare fixed source and checkpoint identities, then initialize the immutable run
        # Baseline preparation failures are retained and do not prevent the main-method attempt.
        VIDEOSEAL_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"
        VIDEOSEAL_REPO = "https://github.com/facebookresearch/videoseal.git"
        VIDEOSEAL_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth"
        RIVAGAN_COMMIT = "efffa72a4ca46d4d5051f6970c96424c2cdab441"
        RIVAGAN_REPO = "https://github.com/DAI-Lab/RivaGAN.git"
        RIVAGAN_WEIGHT_COMMIT = "4d9928350509f808b6b57d48d2f958aef811d332"
        RIVAGAN_CHECKPOINT_URL = "https://raw.githubusercontent.com/Peachypie98/RivaGAN/4d9928350509f808b6b57d48d2f958aef811d332/model_weight/rivagan_32bit_model.pt"

        def verified_checkout(label, url, commit, destination):
            destination = Path(destination)
            if not destination.exists():
                logged(["git", "clone", "--filter=blob:none", url, str(destination)], label + "_CLONE")
            has_commit = subprocess.run(
                ["git", "-C", str(destination), "cat-file", "-e", commit + "^{commit}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            ).returncode == 0
            if not has_commit:
                logged(["git", "-C", str(destination), "fetch", "origin", commit], label + "_FETCH")
            logged(["git", "-C", str(destination), "checkout", "--detach", commit], label + "_CHECKOUT")
            actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True)
            if actual != commit or dirty:
                raise RuntimeError(label + " source commit/clean check failed")
            return destination

        def cached_download(label, url, destination):
            destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                temporary = destination.with_suffix(destination.suffix + ".download")
                if temporary.exists(): temporary.unlink()
                urllib.request.urlretrieve(url, temporary)
                os.replace(temporary, destination)
            if destination.stat().st_size == 0:
                raise RuntimeError(label + " downloaded an empty file")
            return destination, sha256_file(destination)

        baseline_setup = {}
        VS_ROOT = CACHE_ROOT / "sources" / ("videoseal-" + VIDEOSEAL_COMMIT)
        VS_CARD = VS_ROOT / "videoseal/cards/videoseal_1.0.yaml"
        VS_WEIGHT = CACHE_ROOT / "checkpoints/videoseal/y_256b_img.pth"
        try:
            verified_checkout("VIDEOSEAL_SOURCE", VIDEOSEAL_REPO, VIDEOSEAL_COMMIT, VS_ROOT)
            VS_WEIGHT, vs_weight_sha = cached_download("VIDEOSEAL_CHECKPOINT", VIDEOSEAL_CHECKPOINT_URL, VS_WEIGHT)
            vs_card_sha = sha256_file(VS_CARD)
            baseline_setup["videoseal"] = {"status": "READY", "source_commit": VIDEOSEAL_COMMIT, "card_sha256": vs_card_sha, "checkpoint_sha256": vs_weight_sha}
        except Exception as exc:
            vs_card_sha = "0" * 64; vs_weight_sha = "0" * 64
            baseline_setup["videoseal"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "source_commit": VIDEOSEAL_COMMIT}

        RIVA_ROOT = CACHE_ROOT / "sources" / ("rivagan-" + RIVAGAN_COMMIT)
        RIVA_WEIGHT = CACHE_ROOT / "checkpoints/rivagan/rivagan_32bit_model.pt"
        try:
            verified_checkout("RIVAGAN_SOURCE", RIVAGAN_REPO, RIVAGAN_COMMIT, RIVA_ROOT)
            RIVA_WEIGHT, riva_weight_sha = cached_download("RIVAGAN_CHECKPOINT", RIVAGAN_CHECKPOINT_URL, RIVA_WEIGHT)
            baseline_setup["rivagan"] = {
                "status": "READY", "source_commit": RIVAGAN_COMMIT,
                "checkpoint_source_commit": RIVAGAN_WEIGHT_COMMIT,
                "checkpoint_sha256": riva_weight_sha,
                "checkpoint_provenance": "Peachypie98 community 32-bit checkpoint; not DAI-Lab official weights",
            }
        except Exception as exc:
            riva_weight_sha = "0" * 64
            baseline_setup["rivagan"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "source_commit": RIVAGAN_COMMIT, "checkpoint_source_commit": RIVAGAN_WEIGHT_COMMIT}
        atomic_json(OUTPUT_ROOT / "baseline_setup_receipts.json", baseline_setup)

        WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
        FRAMEWISE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
        WAN_SNAPSHOT = CACHE_ROOT / "models/Wan2.1-T2V-1.3B-Diffusers" / WAN_REVISION
        FRAMEWISE_SNAPSHOT = CACHE_ROOT / "models/sd-vae-ft-mse" / FRAMEWISE_REVISION
        effective = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        effective["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        effective["models"]["framewise"]["local_snapshot_path"] = str(FRAMEWISE_SNAPSHOT)
        effective["models"]["videoseal"].update({
            "source_root": str(VS_ROOT), "source_commit": VIDEOSEAL_COMMIT,
            "card_path": str(VS_CARD), "card_sha256": vs_card_sha,
            "checkpoint_path": str(VS_WEIGHT), "checkpoint_sha256": vs_weight_sha,
            "checkpoint_download_status": "DOWNLOADED_OR_REUSED_AND_SHA256_BOUND_BY_NOTEBOOK" if baseline_setup["videoseal"]["status"] == "READY" else "FAILED_NOT_AVAILABLE",
        })
        effective["models"]["rivagan"].update({
            "source_root": str(RIVA_ROOT), "source_commit": RIVAGAN_COMMIT,
            "checkpoint_path": str(RIVA_WEIGHT), "checkpoint_sha256": riva_weight_sha,
            "source_url": RIVAGAN_REPO,
            "checkpoint_source_url": RIVAGAN_CHECKPOINT_URL,
            "checkpoint_source_commit": RIVAGAN_WEIGHT_COMMIT,
            "checkpoint_download_status": "DOWNLOADED_OR_REUSED_AND_SHA256_BOUND_BY_NOTEBOOK" if baseline_setup["rivagan"]["status"] == "READY" else "FAILED_NOT_AVAILABLE",
        })
        EFFECTIVE_CONFIG = OUTPUT_ROOT / "effective_config.json"
        atomic_json(EFFECTIVE_CONFIG, effective)
        EFFECTIVE_CONFIG_SHA256 = sha256_file(EFFECTIVE_CONFIG)
        portable_env = {**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}
        logged([PYTHON, "-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(OUTPUT_ROOT / "fixed_plan"), "--phase", "plan"], "EFFECTIVE_FIXED_PLAN", env=portable_env)
        logged([PYTHON, "-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(RUN_OUTPUT), "--phase", "init"], "RUN_STATE_INIT", env=portable_env)
        record_stage("FIXED_IDENTITIES_AND_RUN_INIT", "SUCCEEDED", effective_config_sha256=EFFECTIVE_CONFIG_SHA256)
        print("Initialized full 10-case denominator. Run all will attempt only:", PILOT_CASES)
    ''')

    prepare_environment = textwrap.dedent('''\
        ## Prepare the recorded main environment and isolated baseline dependencies
        # The historical non-zero pip check is retained as a diagnostic and is not a blanket hard failure.
        MAIN_PINS = {
            "torch": "2.11.0", "torchvision": "0.26.0", "diffusers": "0.39.0", "transformers": "4.57.6",
            "numpy": "2.1.3", "accelerate": "1.15.0", "safetensors": "0.8.0",
            "huggingface-hub": "0.36.2", "tokenizers": "0.22.2",
            "sentencepiece": "0.2.2", "ftfy": "6.3.1",
        }
        probe = "import importlib.metadata as m; pins=" + repr(MAIN_PINS) + "; actual={k:m.version(k) for k in pins}; assert all(actual[k].split('+')[0]==v for k,v in pins.items()),actual; from diffusers import AutoencoderKL,AutoencoderKLWan,WanPipeline; import torch,sentencepiece,ftfy; print(actual)"
        main_environment_status = "READY"
        try:
            if logged([PYTHON, "-c", probe], "MAIN_DEPENDENCY_PROBE", check=False):
                logged([PYTHON, "-m", "pip", "install", *[f"{name}=={version}" for name, version in MAIN_PINS.items()]], "MAIN_PINNED_REPAIR")
                logged([PYTHON, "-c", probe], "MAIN_DEPENDENCY_REPROBE")
            dependency_check_returncode = logged([PYTHON, "-m", "pip", "check"], "MAIN_PIP_CHECK_DIAGNOSTIC", check=False)
            (OUTPUT_ROOT / "environment_freeze.txt").write_text(subprocess.check_output([PYTHON, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        except Exception as exc:
            main_environment_status = "FAILED"
            dependency_check_returncode = None
            record_stage("MAIN_ENVIRONMENT_PREPARATION", "FAILED", reason=f"{type(exc).__name__}: {exc}")

        baseline_pythons = {"videoseal": PYTHON, "rivagan": PYTHON}
        baseline_dependencies = {
            "videoseal": [
                "PyWavelets", "av", "calflops", "decord", "einops==0.8.2", "lpips",
                "omegaconf==2.3.0", "opencv-python", "pandas", "pycocotools",
                "pytorch-msssim", "scikit-image", "scipy", "tensorboard", "timm==0.9.16",
            ],
            "rivagan": ["torch-dct==0.1.5", "opencv-python", "pandas==2.2.3"],
        }
        for method in ("videoseal", "rivagan"):
            venv = Path("/content") / ("paper-results-v1-" + method + "-venv")
            vpython = venv / "bin/python"
            try:
                if baseline_setup[method]["status"] != "READY":
                    raise RuntimeError(method + " source/checkpoint preparation failed")
                if not vpython.exists():
                    logged([PYTHON, "-m", "venv", "--system-site-packages", str(venv)], method.upper() + "_VENV_CREATE")
                # No official baseline requirements file is installed: it would downgrade the main stack.
                logged([str(vpython), "-m", "pip", "install", "--no-deps", *baseline_dependencies[method]], method.upper() + "_ISOLATED_DEPENDENCIES")
                baseline_pythons[method] = str(vpython)
                baseline_setup[method]["environment_status"] = "READY_UNEXECUTED_MODEL_COMPATIBILITY"
                baseline_setup[method]["environment_freeze_path"] = str(OUTPUT_ROOT / (method + "_environment_freeze.txt"))
                (OUTPUT_ROOT / (method + "_environment_freeze.txt")).write_text(
                    subprocess.check_output([str(vpython), "-m", "pip", "freeze"], text=True), encoding="utf-8",
                )
            except Exception as exc:
                baseline_setup[method]["environment_status"] = "FAILED"
                baseline_setup[method]["environment_reason"] = f"{type(exc).__name__}: {exc}"
                baseline_pythons[method] = PYTHON
        atomic_json(OUTPUT_ROOT / "baseline_setup_receipts.json", baseline_setup)

        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
            if hf_token: os.environ["HF_TOKEN"] = hf_token
        except Exception:
            hf_token = None
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", revision=WAN_REVISION,
                local_dir=str(WAN_SNAPSHOT),
                allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"],
            )
            snapshot_download(
                repo_id="stabilityai/sd-vae-ft-mse", revision=FRAMEWISE_REVISION,
                local_dir=str(FRAMEWISE_SNAPSHOT),
                allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
            )
            model_snapshot_status = "READY"
        except Exception as exc:
            model_snapshot_status = "FAILED"
            record_stage("MODEL_SNAPSHOT_PREPARATION", "FAILED", reason=f"{type(exc).__name__}: {exc}")
        finally:
            hf_token = None
        atomic_json(OUTPUT_ROOT / "environment_setup_receipt.json", {
            "main_environment_status": main_environment_status,
            "main_pip_check_returncode_nonfatal_diagnostic": dependency_check_returncode,
            "model_snapshot_status": model_snapshot_status,
            "wan_revision": WAN_REVISION, "framewise_revision": FRAMEWISE_REVISION,
            "baseline_environments": baseline_setup,
            "note": "Model/backend compatibility is established only by later phase receipts, not by preparation status.",
        })
        print("Environment:", main_environment_status, "models:", model_snapshot_status, "pip check diagnostic:", dependency_check_returncode)
    ''')

    execute = textwrap.dedent('''\
        ## Execute the fixed two pilots once, one model family per process
        from experiments.paper_results_v1.colab_orchestration import execute_fixed_sequence

        if sha256_file(EFFECTIVE_CONFIG) != EFFECTIVE_CONFIG_SHA256:
            raise RuntimeError("effective configuration changed after RunStore initialization")
        portable_env = {**os.environ, "PYTHONPATH": str(PORTABLE_ROOT)}
        cli_prefix = ["-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(RUN_OUTPUT)]
        phase_interpreters = {
            "baseline-embed-videoseal": baseline_pythons["videoseal"],
            "baseline-extract-videoseal": baseline_pythons["videoseal"],
            "baseline-embed-rivagan": baseline_pythons["rivagan"],
            "baseline-extract-rivagan": baseline_pythons["rivagan"],
        }

        def persist_execution(progress):
            atomic_json(OUTPUT_ROOT / "pilot_phase_attempts.json", progress)
            if progress.get("interrupted"):
                atomic_json(OUTPUT_ROOT / "interruption_handoff.json", {
                    "status": "INTERRUPTED_AFTER_RETAINING_PROGRESS",
                    "fixed_denominator": FIXED_FULL_DENOMINATOR,
                    "attempt_scope": FIXED_ATTEMPT_SCOPE,
                    "confirmation_status": "NOT_EXECUTED_BY_NOTEBOOK",
                    "progress": progress,
                    "run_state": str(RUN_OUTPUT / "run_state.json"),
                    "evaluation_report": str(RUN_OUTPUT / "evaluation_report.json"),
                })

        def preflight_once():
            # BLOCKED is retained but does not erase the plan or suppress independent attempts.
            return logged([PYTHON, *cli_prefix, "--phase", "preflight"], "STATIC_PREFLIGHT", env=portable_env, check=False)

        def invoke_once(case_id, phase):
            interpreter = phase_interpreters.get(phase, PYTHON)
            return logged([interpreter, *cli_prefix, "--phase", phase, "--case-id", case_id], f"{case_id}:{phase}", env=portable_env, check=False)

        def evaluate_once():
            return logged([PYTHON, *cli_prefix, "--phase", "evaluate"], "FINAL_FIXED_DENOMINATOR_EVALUATE", env=portable_env, check=False)

        try:
            execution_progress = execute_fixed_sequence(
                pilot_ids=PILOT_CASES, phases=PHASE_ORDER,
                preflight=preflight_once, invoke=invoke_once,
                evaluate=evaluate_once, persist=persist_execution,
            )
        except KeyboardInterrupt:
            record_stage("FIXED_TWO_PILOT_SEQUENCE", "INTERRUPTED_AFTER_REPORT_ONLY_CLEANUP_ATTEMPT")
            raise
        failed_attempts = sum(row["status"] != "COMPLETE" for row in execution_progress["phase_attempts"])
        record_stage(
            "FIXED_TWO_PILOT_SEQUENCE",
            "COMPLETE_WITH_RETAINED_FAILURES" if failed_attempts or execution_progress["evaluate"]["status"] != "COMPLETE" else "COMPLETE",
            evaluate=execution_progress["evaluate"],
        )
        print("Pilot phase attempts:", len(execution_progress["phase_attempts"]), "failed:", failed_attempts, "evaluate:", execution_progress["evaluate"])
    ''')

    summarize = textwrap.dedent('''\
        ## Write the compact handoff summary without changing any result row
        from experiments.paper_results_v1.colab_orchestration import build_scope_summary

        state_path = RUN_OUTPUT / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report_path = RUN_OUTPUT / "evaluation_report.json"
        report = None; report_load = {"status": "MISSING", "reason": "evaluation_report.json not found"}
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report_load = {"status": "LOADED", "reason": None}
            except Exception as exc:
                report_load = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        scope_summary = build_scope_summary(
            state, report, pilot_ids=PILOT_CASES, confirmation_ids=CONFIRMATION_CASES,
        )
        handoff = {
            "schema_version": "paper-results-v1-two-pilot-colab-handoff-v1",
            "run_id": RUN_ID,
            "status": "REPORT_WRITTEN" if report is not None else "REPORT_MISSING_EVALUATE_FAILED",
            "evaluation_report_load": report_load,
            "execution_scope": {
                "attempted": list(PILOT_CASES),
                "confirmation": list(CONFIRMATION_CASES),
                "confirmation_status": "NOT_EXECUTED_BY_NOTEBOOK",
                "pilot_excluded_from_confirmation": True,
            },
            "identity": {
                "portable_zip_sha256": PORTABLE_ZIP_SHA256,
                "effective_config_sha256": EFFECTIVE_CONFIG_SHA256,
                "historical_real_evidence_source_sha": "ac111d0fed253767651929d115c343fe1636c525",
                "historical_evidence_is_not_this_run": True,
            },
            "fixed_denominator": FIXED_FULL_DENOMINATOR,
            "attempt_scope": FIXED_ATTEMPT_SCOPE,
            "pilot_outputs": {
                "pilot_A": str(RUN_OUTPUT / "artifacts/pilot_01"),
                "pilot_B": str(RUN_OUTPUT / "artifacts/pilot_02"),
            },
            **scope_summary,
            "files": {
                "run_state": str(state_path), "evaluation_report": str(report_path),
                "receiver_csv": str(RUN_OUTPUT / "receiver_rows.csv"),
                "baseline_csv": str(RUN_OUTPUT / "baseline_rows.csv"),
                "comparison_csv": str(RUN_OUTPUT / "comparison_rows.csv"),
                "comparison_source_summaries_csv": str(RUN_OUTPUT / "comparison_source_summaries.csv"),
                "quality_csv": str(RUN_OUTPUT / "quality_rows.csv"),
                "execution_log": str(LOG_PATH), "stage_receipts": str(STAGE_RECEIPTS),
            },
            "evidence_ceiling": "This summary reports this run's retained rows only. It does not inherit historical scientific success and pilots do not enter confirmation.",
        }
        atomic_json(OUTPUT_ROOT / "handoff_summary.json", handoff)
        print("Handoff:", OUTPUT_ROOT / "handoff_summary.json")
        print("Pilot status counts:", json.dumps(handoff["pilot_status_counts"], sort_keys=True))
        print("Pilot status source:", handoff["pilot_status_source"])
        print("Confirmation:", handoff["confirmation"]["execution_scope_status"], "; report projection:", handoff["confirmation"]["report_projection_status_counts"])
    ''')

    cells = [
        _cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
        _cell("markdown", textwrap.dedent('''\
            # Paper Results V1 — fixed two-pilot Run-all

            This handoff attempts **pilot_01** and **pilot_02** once, in the frozen order below. The eight confirmation cases remain in the immutable ten-case manifest but are `NOT_EXECUTED_BY_NOTEBOOK`; full-denominator evaluation projects absent receiver/baseline/comparison evidence to disclosed FAILED/UNEVALUABLE rows while quality retains its recorded state. These are not attempted model failures, and pilots are never pooled into confirmation.

            The notebook embeds the current evaluator and its runtime/main source closure, so the notebook is the only project file to upload. It downloads named model snapshots, exact baseline source commits, and named checkpoint objects when the user runs it. The notebook itself has not been run against real models in this delivery; local validation is static plus CPU/fake only.

            Historical real evidence remains bound to source `ac111d0fed253767651929d115c343fe1636c525` and run `20261008T003135066923Z/fixed_reference`. That evidence checks the main chain, not this notebook or either external baseline.
        '''), "goal"),
        _cell("code", setup, "portable-setup"),
        _cell("code", prepare_identity, "identity-and-init"),
        _cell("code", prepare_environment, "environment-and-models"),
        _cell("code", execute, "fixed-execution"),
        _cell("code", summarize, "handoff-summary"),
        _cell("markdown", textwrap.dedent('''\
            ## Checks and handoff

            Fixed two-pilot work comprises 4 trajectories, 200 scheduler steps, 400 conditional/unconditional transformer forwards, 2 framewise writer encodes, 4 framewise decodes, 12 complete MP4 roundtrips, 72 receiver synchronization encodes, 320 logical Wan reads, and at most 240 physical Wan receiver encodes. Aliasing can reduce the physical count; the upper bound is not a measurement.

            Six arms' PRE/POST RGB8 rasters require 2,135,162,880 bytes (about 1.99 GiB) for two pilots before MP4, terminal/shared latents, native-soft sidecars, logs, or filesystem overhead. The fixed Wan snapshot objects total about 28.93 GB and the framewise safetensors object is about 334.64 MB. Peak GPU memory and wall time have not been measured, and no GPU model threshold is claimed. A CUDA runtime is required by the adopted execution configuration.

            Return the entire unique Drive run directory, especially `handoff_summary.json`, `effective_config.json`, `portable_source_receipt.json`, `execution.log`, `stage_receipts.json`, `run_state/run_state.json`, all CSV/JSON evaluation reports, both `run_state/artifacts/pilot_01` and `pilot_02` trees, and every native `.npz` sidecar. Do not rerun a failed phase in the same run directory; another attempt requires a new Run-all directory.

            `pip check` is retained as a diagnostic. The historical successful main environment had a non-zero result from unrelated Gradio/Hub and Jedi conflicts, so a non-zero value alone is not used to erase model-phase evidence. Baseline dependencies are installed into separate system-site-package virtual environments and the old RivaGAN requirements are not installed because they pin obsolete Torch/OpenCV/Pandas/NumPy versions.
        '''), "handoff"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main():
    notebook = build_notebook()
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
