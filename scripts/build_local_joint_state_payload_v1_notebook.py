"""Build the single-file Colab handoff for Local Joint State+Payload V1.

The notebook embeds the exact portable runtime closure.  It does not require a
published Git commit and therefore never invents or inherits a source SHA.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_FILES = (
    "experiments/__init__.py",
    "experiments/wan_state_clock/__init__.py",
    "experiments/wan_state_clock/local_joint_state_payload_v1_run.py",
    "main/__init__.py",
    "main/tube_state/__init__.py",
    "main/tube_state/local_joint_state_payload_carrier_v1.py",
    "main/tube_state/local_joint_state_payload_v1.py",
    "main/tube_state/grow_video_reference.py",
    "runtime/__init__.py",
    "runtime/wan/__init__.py",
    "runtime/wan/generation.py",
    "runtime/wan/fixed_rgb_media.py",
    "runtime/wan/grow_video_reference.py",
    "runtime/wan/io.py",
    "runtime/wan/local_joint_state_payload_experiment_v1.py",
    "runtime/wan/local_joint_state_payload_provider_v1.py",
    "runtime/wan/local_joint_state_payload_v1.py",
    "runtime/wan/provenance.py",
    "runtime/wan/trajectory.py",
    "runtime/wan/vae.py",
    "main/tube_state/local_joint_state_payload_posthoc_v1.py",
    "experiments/wan_state_clock/local_joint_state_payload_posthoc_v1_run.py",
    "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json",
)
RUNNER_FILES = PORTABLE_FILES[:20]
CONFIG_PATH = "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json"


def _content_id(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def portable_archive() -> tuple[bytes, dict[str, object]]:
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in PORTABLE_FILES}
    manifest: dict[str, object] = dict(
        schema="local-joint-portable-source-v1",
        files=hashes,
        content_sha256=_content_id(hashes),
        git_commit=None,
        source_kind="embedded_unversioned_directory",
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in PORTABLE_FILES:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / name).read_bytes())
        info = zipfile.ZipInfo("portable_source_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return stream.getvalue(), manifest


def _code(source: str) -> dict[str, object]:
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=textwrap.dedent(source).strip().splitlines(keepends=True))


def _markdown(source: str) -> dict[str, object]:
    return dict(cell_type="markdown", metadata={},
                source=textwrap.dedent(source).strip().splitlines(keepends=True))


def build_notebook() -> dict[str, object]:
    package, manifest = portable_archive()
    package_sha = hashlib.sha256(package).hexdigest()
    runner_content_sha = _content_id(
        {name: manifest["files"][name] for name in RUNNER_FILES}
    )
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    config_sha = hashlib.sha256((ROOT / CONFIG_PATH).read_bytes()).hexdigest()
    build_id = hashlib.sha256(
        (package_sha + config_sha + "local-joint-colab-v1").encode()
    ).hexdigest()
    encoded = base64.b64encode(package).decode("ascii")
    chunks = "\n".join(f"    {chunk!r}" for chunk in
                       (encoded[index:index + 100] for index in range(0, len(encoded), 100)))

    setup = f'''\
    from pathlib import Path
    import datetime, hashlib, json, os, signal, subprocess, sys, time, traceback

    NOTEBOOK_BUILD_ID = {build_id!r}
    PACKAGE_SHA256 = {package_sha!r}
    SOURCE_CONTENT_SHA256 = {manifest["content_sha256"]!r}
    RUNNER_CONTENT_SHA256 = {runner_content_sha!r}
    CONFIG_SHA256 = {config_sha!r}
    FIXED_CONFIG = {config!r}
    EXPECTED_COST = {{"arms": 2, "native_steps": 100, "transformer_forwards": 200,
                     "joint_decode": 25, "joint_encode": 50, "terminal_decode": 2,
                     "vae_load_attempts_if_complete": 27, "joint_residency_roundtrips": 25}}
    OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1')
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    OUTPUT = OUTPUT_PARENT / STAMP
    OUTPUT.mkdir(exist_ok=False)
    RUN_OUTPUT = OUTPUT / 'run'
    POSTHOC_OUTPUT = OUTPUT / 'posthoc'
    WORKSPACE = Path('/content/Video-WM-Local-Joint-State-Payload-V1-' + STAMP)
    PYTHON = sys.executable

    def write_json(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\\n', encoding='utf-8')
        os.replace(temporary, path)

    def record_failure(stage, exc):
        row = dict(stage=stage, reason=f'{{type(exc).__name__}}: {{exc}}',
                   traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                   retry=False, output=str(OUTPUT), expected_cost=EXPECTED_COST)
        write_json(OUTPUT / 'notebook_failure.json', row)

    def logged(command, stage, *, cwd=None, env=None, check=True):
        child = None; returncode = None; primary = None; primary_tb = None; cleanup_errors = []
        def cleanup_error(label, exc):
            cleanup_errors.append(label + ': ' + repr(exc))
        try:
            with (OUTPUT / 'execution.log').open('a', encoding='utf-8') as log:
                line = 'STAGE ' + stage + ' COMMAND ' + repr(command) + '\\n'
                print(line, end='', flush=True); log.write(line); log.flush()
                group = {{'start_new_session': True}} if os.name == 'posix' else {{}}
                child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, **group)
                try:
                    for line in child.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    returncode = child.wait()
                except BaseException as exc:
                    primary = exc; primary_tb = exc.__traceback__
                finally:
                    if child is not None:
                        if os.name == 'posix':
                            try: os.killpg(child.pid, signal.SIGTERM)
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_error('group SIGTERM', exc)
                            deadline = time.monotonic() + 5.0
                            while time.monotonic() < deadline:
                                try: os.killpg(child.pid, 0)
                                except ProcessLookupError: break
                                except BaseException as exc:
                                    cleanup_error('group probe', exc); break
                                time.sleep(0.05)
                            try: os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_error('group SIGKILL', exc)
                        elif child.poll() is None:
                            try: child.terminate()
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_error('terminate', exc)
                        try: returncode = child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try: child.kill()
                            except ProcessLookupError: pass
                            except BaseException as exc: cleanup_error('kill', exc)
                            try: returncode = child.wait(timeout=10)
                            except BaseException as exc: cleanup_error('wait after kill', exc)
                        except BaseException as exc: cleanup_error('wait', exc)
                        if child.stdout is not None:
                            try: child.stdout.close()
                            except BaseException as exc: cleanup_error('stdout close', exc)
                if primary is None and not cleanup_errors:
                    log.write('EXIT ' + str(returncode) + '\\n'); log.flush()
        except BaseException as exc:
            if primary is None: primary = exc; primary_tb = exc.__traceback__
            else: cleanup_error('log wrapper', exc)
        if primary is None and cleanup_errors:
            primary = RuntimeError('process cleanup failed: ' + '; '.join(cleanup_errors))
            primary_tb = primary.__traceback__
        if primary is None and check and returncode:
            primary = subprocess.CalledProcessError(returncode, command); primary_tb = primary.__traceback__
        if primary is not None:
            if cleanup_errors and hasattr(primary, 'add_note'):
                primary.add_note('secondary cleanup errors: ' + '; '.join(cleanup_errors))
            try: record_failure(stage, primary)
            except BaseException as record_error:
                if hasattr(primary, 'add_note'): primary.add_note('failure record error: ' + repr(record_error))
            raise primary.with_traceback(primary_tb)
        return returncode

    fixed_slots = dict(
        schema='local-joint-colab-fixed-slots-v1', status='SETUP_NOT_COMPLETED',
        arms={{arm: dict(status='NOT_RUN', steps=[dict(index=i, status='NOT_RUN') for i in range(50)],
                        layers={{name: dict(status='MISSING_DEPENDENCY') for name in
                                ('terminal_latent', 'float_rgb', 'rgb8', 'mp4')}},
                        observations={{f'{{layer}}/{{label}}': dict(status='MISSING_DEPENDENCY')
                                      for layer in ('float_rgb', 'rgb8', 'mp4')
                                      for label in ('CORRECT', 'WRONG')}})
              for arm in ('OFF', 'JOINT')}})
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    write_json(OUTPUT / 'setup_receipt.json', dict(
        status='SETUP_STARTED', notebook_build_id=NOTEBOOK_BUILD_ID,
        package_sha256=PACKAGE_SHA256, source_content_sha256=SOURCE_CONTENT_SHA256,
        config_sha256=CONFIG_SHA256, expected_cost=EXPECTED_COST,
        output=str(OUTPUT), python=sys.version, executable=sys.executable,
        execution_authorized_by_notebook_run=True, scientific_pass=False))
    print('fresh B-line output:', OUTPUT, flush=True)
    '''

    environment = '''\
    import importlib.metadata, shutil
    try:
        if any(shutil.which(name) is None for name in ('ffmpeg', 'ffprobe')):
            logged(['apt-get', 'update', '-qq'], 'MEDIA_TOOL_UPDATE')
            logged(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], 'MEDIA_TOOL_INSTALL')
        probe_path = OUTPUT / 'dependency_probe.json'
        probe_lines = [
            "import importlib.metadata as metadata, json, pathlib, shutil, sys, torch, diffusers",
            "from diffusers import WanPipeline, AutoencoderKLWan",
            "pins = {'torch':'2.11.0','diffusers':'0.39.0','transformers':'4.57.6','numpy':'2.1.3','accelerate':'1.15.0','safetensors':'0.8.0','huggingface-hub':'0.36.2','tokenizers':'0.22.2','sentencepiece':'0.2.2','ftfy':'6.3.1'}",
            "actual = {name: metadata.version(name) for name in pins}",
            "assert all(actual[name].split('+',1)[0] == version for name, version in pins.items()), actual",
            "assert torch.cuda.is_available(), 'CUDA torch required'",
            "row = dict(python=sys.version, versions=actual, torch_cuda_runtime=torch.version.cuda, cuda_available=True, device=torch.cuda.get_device_name(0), total_device_memory=int(torch.cuda.get_device_properties(0).total_memory), ffmpeg=shutil.which('ffmpeg'), ffprobe=shutil.which('ffprobe'))",
            "pathlib.Path(" + repr(str(probe_path)) + ").write_text(json.dumps(row, indent=2)+'\\n', encoding='utf-8')",
        ]
        probe_code = '\\n'.join(probe_lines)
        probe_returncode = logged([PYTHON, '-u', '-c', probe_code], 'DEPENDENCY_PROBE', check=False)
        if probe_returncode:
            logged([PYTHON, '-m', 'pip', 'install', 'torch==2.11.0', 'diffusers==0.39.0',
                    'transformers==4.57.6', 'numpy==2.1.3', 'accelerate==1.15.0',
                    'safetensors==0.8.0', 'huggingface-hub==0.36.2', 'tokenizers==0.22.2',
                    'sentencepiece==0.2.2', 'ftfy==6.3.1'], 'DEPENDENCY_REPAIR')
            logged([PYTHON, '-u', '-c', probe_code], 'DEPENDENCY_REPROBE')
        dependency_check_returncode = logged([PYTHON, '-m', 'pip', 'check'], 'DEPENDENCY_REPORT', check=False)
        (OUTPUT / 'environment_freeze.txt').write_text(
            subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
        environment_receipt = json.loads(probe_path.read_text(encoding='utf-8'))
        environment_receipt.update(
            dependency_check_returncode=dependency_check_returncode,
            gpu_name_is_not_a_gate=True,
            current_joint_compatibility_reference='ac111d0 runtime: torch2.11.0+cu130 diffusers0.39.0',
            historical_residency_reference='N2 multistep: Python3.13.15 torch2.11.0+cu128 diffusers0.40.0 L4',
            resource_claim='current 27D/50E/27-load memory and latency remain unverified')
        write_json(OUTPUT / 'environment_receipt.json', environment_receipt)
    except BaseException as exc:
        try: record_failure('ENVIRONMENT_SETUP', exc)
        except BaseException as record_error:
            if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
        raise
    '''

    source = f'''\
    import base64, io, zipfile
    SOURCE_PACKAGE_B64 = (\n{chunks}\n    )
    try:
        package_bytes = base64.b64decode(''.join(SOURCE_PACKAGE_B64), validate=True)
        if hashlib.sha256(package_bytes).hexdigest() != PACKAGE_SHA256:
            raise RuntimeError('embedded source package SHA mismatch')
        WORKSPACE.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
            names = set(archive.namelist())
            expected = set({list(PORTABLE_FILES)!r}) | {{'portable_source_manifest.json'}}
            if names != expected:
                raise RuntimeError('embedded source package file set mismatch')
            for name in sorted(names):
                target = (WORKSPACE / name).resolve()
                if not target.is_relative_to(WORKSPACE.resolve()):
                    raise RuntimeError('unsafe embedded archive path')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        manifest = json.loads((WORKSPACE / 'portable_source_manifest.json').read_text(encoding='utf-8'))
        actual_hashes = {{name: hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest()
                         for name in manifest['files']}}
        actual_content = hashlib.sha256(json.dumps(actual_hashes, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
        if (actual_hashes != manifest['files'] or actual_content != SOURCE_CONTENT_SHA256
                or manifest['content_sha256'] != SOURCE_CONTENT_SHA256
                or manifest['git_commit'] is not None or (WORKSPACE / '.git').exists()):
            raise RuntimeError('portable source identity mismatch')
        config_path = WORKSPACE / {CONFIG_PATH!r}
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != CONFIG_SHA256:
            raise RuntimeError('fixed config byte identity mismatch')
        if json.loads(config_path.read_text(encoding='utf-8')) != FIXED_CONFIG:
            raise RuntimeError('fixed config semantic mismatch')
        write_json(OUTPUT / 'portable_source_receipt.json', dict(
            status='VERIFIED', kind='embedded_unversioned_directory', git_commit=None,
            package_sha256=PACKAGE_SHA256, content_sha256=actual_content,
            files=actual_hashes, workspace=str(WORKSPACE), config_sha256=CONFIG_SHA256))
    except BaseException as exc:
        try: record_failure('PORTABLE_SOURCE', exc)
        except BaseException as record_error:
            if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
        raise
    '''

    run = '''\
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        if hf_token: os.environ['HF_TOKEN'] = hf_token
    except Exception as token_error:
        write_json(OUTPUT / 'optional_hf_token_receipt.json',
                   dict(status='UNAVAILABLE_OR_NOT_SET', reason=f'{type(token_error).__name__}: {token_error}'))
    finally:
        hf_token = None

    env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['PYTHONPATH'] = str(WORKSPACE)
    run_command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_state_payload_v1_run',
                   '--config', str(WORKSPACE / 'experiments/wan_state_clock/configs/local_joint_state_payload_v1.json'),
                   '--output', str(RUN_OUTPUT)]
    posthoc_command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_state_payload_posthoc_v1_run',
                       '--run-result', str(RUN_OUTPUT / 'result.json'),
                       '--config', str(WORKSPACE / 'experiments/wan_state_clock/configs/local_joint_state_payload_v1.json'),
                       '--output', str(POSTHOC_OUTPUT)]
    primary = None; primary_tb = None; secondary = []
    run_returncode = posthoc_returncode = None
    try:
        run_returncode = logged(run_command, 'REAL_FIXED_RUN', cwd=WORKSPACE, env=env, check=False)
        if run_returncode:
            primary = RuntimeError(f'real runner exited {run_returncode}'); primary_tb = primary.__traceback__
        required = (RUN_OUTPUT / 'result.json', RUN_OUTPUT / 'raw_observation_manifest.json')
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            error = FileNotFoundError('runner retained artifacts missing: ' + repr(missing))
            if primary is None: primary = error; primary_tb = error.__traceback__
            else: secondary.append(repr(error))
        else:
            posthoc_returncode = logged(posthoc_command, 'SEALED_POSTHOC', cwd=WORKSPACE, env=env, check=False)
            if posthoc_returncode:
                error = RuntimeError(f'posthoc exited {posthoc_returncode}')
                if primary is None: primary = error; primary_tb = error.__traceback__
                else: secondary.append(repr(error))
    except BaseException as exc:
        if primary is None: primary = exc; primary_tb = exc.__traceback__
        elif exc is not primary: secondary.append(repr(exc))

    audit = dict(
        schema='local-joint-colab-audit-v1', notebook_build_id=NOTEBOOK_BUILD_ID,
        source_content_sha256=SOURCE_CONTENT_SHA256,
        runner_content_sha256=RUNNER_CONTENT_SHA256, config_sha256=CONFIG_SHA256,
        run_returncode=run_returncode, posthoc_returncode=posthoc_returncode,
        expected_cost=EXPECTED_COST, automatic_retry=False, scientific_pass=False,
        resource_claim='27D/50E/27-load and 25 roundtrips are planned costs, not verified capacity')
    try:
        if (RUN_OUTPUT / 'result.json').is_file():
            run_result = json.loads((RUN_OUTPUT / 'result.json').read_text(encoding='utf-8'))
            audit['run'] = dict(status=run_result.get('status'), stage=run_result.get('stage'),
                                execution=run_result.get('execution'), actual_model_calls=run_result.get('actual_model_calls'),
                                source_identity=run_result.get('source_identity'),
                                arms={name: row.get('status') for name, row in run_result.get('arms', {}).items()})
            identity = run_result.get('source_identity', {})
            if (identity.get('kind') != 'unversioned_directory' or identity.get('git_commit') is not None
                or identity.get('content_sha256') != RUNNER_CONTENT_SHA256):
                raise RuntimeError('runner portable source identity mismatch')
        if (POSTHOC_OUTPUT / 'raw_observation_seal.json').is_file():
            raw_seal = json.loads((POSTHOC_OUTPUT / 'raw_observation_seal.json').read_text(encoding='utf-8'))
            audit['raw_seal'] = dict(entries=len(raw_seal.get('entries', {})), truth_loaded=raw_seal.get('truth_loaded'),
                                     statuses={name: row.get('status') for name, row in raw_seal.get('entries', {}).items()})
            if len(raw_seal.get('entries', {})) != 12 or raw_seal.get('truth_loaded') is not False:
                raise RuntimeError('posthoc raw seal fixed directory mismatch')
        if (POSTHOC_OUTPUT / 'posthoc_result.json').is_file():
            posthoc_result = json.loads((POSTHOC_OUTPUT / 'posthoc_result.json').read_text(encoding='utf-8'))
            audit['posthoc'] = dict(status=posthoc_result.get('status'),
                                    outcome_classification=posthoc_result.get('outcome_classification'),
                                    mp4_attribution=posthoc_result.get('mp4_attribution'),
                                    scientific_pass=posthoc_result.get('scientific_pass'))
        write_json(OUTPUT / 'notebook_audit.json', audit)
    except BaseException as exc:
        if primary is None: primary = exc; primary_tb = exc.__traceback__
        else: secondary.append('audit: ' + repr(exc))

    try:
        write_json(OUTPUT / 'execution_receipt.json', dict(
            run_command=run_command, posthoc_command=posthoc_command,
            run_returncode=run_returncode, posthoc_returncode=posthoc_returncode,
            result_path=str(RUN_OUTPUT / 'result.json'), posthoc_path=str(POSTHOC_OUTPUT / 'posthoc_result.json'),
            secondary_errors=secondary, automatic_retry=False))
    except BaseException as exc:
        if primary is None: primary = exc; primary_tb = exc.__traceback__
        else: secondary.append('execution receipt: ' + repr(exc))
    if primary is not None:
        if secondary and hasattr(primary, 'add_note'): primary.add_note('secondary errors: ' + '; '.join(secondary))
        try: record_failure('REAL_RUN_OR_POSTHOC', primary)
        except BaseException as record_error:
            if hasattr(primary, 'add_note'): primary.add_note('failure record error: ' + repr(record_error))
        raise primary.with_traceback(primary_tb)
    fixed_slots = json.loads((OUTPUT / 'fixed_slots.json').read_text(encoding='utf-8'))
    fixed_slots.update(status='SUPERSEDED_BY_RUN_AND_POSTHOC', result_path=str(RUN_OUTPUT / 'result.json'),
                       posthoc_path=str(POSTHOC_OUTPUT / 'posthoc_result.json'))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    '''

    summary = '''\
    audit = json.loads((OUTPUT / 'notebook_audit.json').read_text(encoding='utf-8'))
    print('B-line output:', OUTPUT, flush=True)
    print('run:', audit.get('run'), flush=True)
    print('raw seal entries:', audit.get('raw_seal', {}).get('entries'), flush=True)
    print('posthoc:', audit.get('posthoc'), flush=True)
    print('This is one development-source mechanism result, not blind recovery, FPR, generalization, or scientific PASS.', flush=True)
    '''

    return dict(
        cells=[
            _code("from google.colab import drive\ndrive.mount('/content/drive')"),
            _markdown('''
            # Local Joint State+Payload V1 — fixed Colab run

            1. Upload and open this notebook in Colab.
            2. Select a CUDA GPU runtime; no GPU model is required by the notebook.
            3. Choose **Run all** once and authorize the Drive mount when prompted.

            This notebook mounts Drive, prepares the historically grounded Wan environment,
            verifies an embedded no-`.git` B-line source closure, then runs exactly the adopted
            `yellow_sailboat_dev_s2026100701` OFF/JOINT experiment and its seal-first posthoc.

            The fixed configuration uses seed 2026100701, rho 0.5, cap 1, message `8001a55a`, the adopted
            correct/wrong keys, 181×320×512, 50 steps, CFG 5, 8 fps and libx264 CRF18/yuv420p. There are no
            mode switches, parameter scans or automatic retries. A complete run plans 200 transformer forwards,
            100 native steps, 27 VAE decodes, 50 encodes, 27 VAE load attempts and 25 residency roundtrips.
            Historical N2 proved a smaller no-gradient serial path on L4; it does not establish this run's memory
            or latency. The known two-arm float-RGB plus RGB8 rasters total about 0.89 GB. Additional storage for
            model weights/cache, temporary values, 12 raw-observation JSON files, MP4 and logs has not been measured.

            After completion or failure, return/share the entire new UTC directory under
            `MyDrive/Video-WM/Local-Joint-State-Payload-V1/`, including raw observations and failure records.
            When present it contains `notebook_failure.json`, `execution.log`, `fixed_slots.json`,
            `setup_receipt.json`, `environment_receipt.json`, `portable_source_receipt.json`,
            `execution_receipt.json`, `notebook_audit.json`, `run/result.json`,
            `run/raw_observation_manifest.json`, `posthoc/raw_observation_seal.json`, and
            `posthoc/posthoc_result.json`. Return every file that exists. The final summary cell may not run after a
            failure, so the UTC directory is the handoff artifact rather than copied notebook output.

            Results describe one seen development source. The posthoc is known-grid and preserves all missing
            metric slots; it is not blind recovery, message reconstruction, an FPR estimate, or scientific PASS.
            ''') ,
            _code(setup), _code(environment), _code(source), _code(run), _code(summary),
        ],
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3"),
            colab=dict(name="local_joint_state_payload_v1_colab.ipynb", provenance=[]),
        ),
        nbformat=4,
        nbformat_minor=5,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notebooks/local_joint_state_payload_v1_colab.ipynb")
    parser.add_argument(
        "--package-output",
        type=Path,
        default=ROOT / "notebooks/local_joint_state_payload_v1_portable_source.zip",
    )
    args = parser.parse_args(argv)
    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    package, _manifest = portable_archive()
    args.package_output.parent.mkdir(parents=True, exist_ok=True)
    args.package_output.write_bytes(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
