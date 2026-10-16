"""Build the fixed P0 notebook and ordinary editable source ZIP; no publishing."""
from pathlib import Path
import io
import json
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "local_joint_p0_input_vjp_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/" + NAME + ".json"
PORTABLE_FILES = (
    "main/__init__.py", "main/tube_state/__init__.py", "main/tube_state/"+NAME+".py",
    "runtime/__init__.py", "runtime/wan/__init__.py", "runtime/wan/"+NAME+".py",
    "runtime/wan/local_joint_p0_checkpoint_v1.py", "runtime/wan/local_joint_readout_checkpoint_v1.py",
    "experiments/__init__.py", "experiments/wan_state_clock/__init__.py",
    "experiments/wan_state_clock/"+NAME+"_run.py", CONFIG_PATH, "docs/"+NAME+".md",
    "scripts/build_"+NAME+"_notebook.py",
)


def code(source):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=textwrap.dedent(source).strip().splitlines(keepends=True))


def portable_archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PORTABLE_FILES:
            archive.writestr(name, (ROOT/name).read_bytes())
    return buffer.getvalue()


def build_notebook():
    config = json.loads((ROOT/CONFIG_PATH).read_text())
    cells = [code("from google.colab import drive\ndrive.mount('/content/drive')")]
    cells.append(dict(cell_type="markdown", metadata={}, source=[
        "# B P0 — saved full-size DiT input VJP\n\n",
        "Run all downloads the ordinary source ZIP from the existing development branch below. No manual ZIP upload is needed. GPU execution belongs to the user.\n\n",
        "Only P0 is adopted: fixed CONTROL_TAIL_after48, saved prompt/negative/BF16 arithmetic and exact units/gradient.pt direction d as test output cotangent. No new writer, P48/P49, quality caps, fork regeneration or baseline. Native Wan keep-in-FP32 exceptions and attention processors remain unchanged.\n\n",
        "Budget: 1 Transformer preparation; conditional/unconditional each one forward and input VJP; 30 native blocks per branch, at most one checkpoint replay per block (60 total). Serial branch graphs with disk checkpoint boundaries. 0 native/VAE/decode/writer/new video/encode/codec. Record GPU allocated/reserved peaks, sampled host RSS, spool storage/traffic and every attempted call/block. Actual full-size backward resources remain unknown until user execution.\n\n",
        "Save raw outputs/cotangents before reports, including zero/nonfinite/failed units. Physical calls are never retried. A source-precheck failure with confirmed zero preparation/forward/VJP/block attempts may repeat the corrected precheck on same-Drive Run all and enter its first physical attempt. Keep old failures and both rows. Any physical call or uncertain history stays report-only; lost graphs and consumed preparations are not recreated.\n\n",
        "The original saved direction is FP64: keep its values and dtype unchanged, without scaling or renormalizing. Pass it directly as grad_outputs to the FP32 output; native autograd performs the backward dtype conversions along the FP32/BF16 path.\n\n",
        "An ordinary branch-local error ends that unit, then the other independently planned branch may run within the fixed budget. OOM, nonfinite forward/VJP or process interruption stops later work, retaining missing rows. Actual embedding/model dtypes are checked; saved dtype labels are only notes.\n\n",
        "Differences against old saved predictions span preparations and cannot isolate gradmode effects. BF16 autograd is an approximate path cotangent, not a verified derivative of the discrete finite map or writer success. No scientific PASS. Return the whole task directory, including failures.\n",
    ]))
    cells.append(code("""
        from pathlib import Path
        import datetime, importlib, json, os, signal, subprocess, sys, time, traceback
        SOURCE_REF = 'dev/local-joint-state-payload-v1'
        SOURCE_URL = 'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/' + SOURCE_REF + '/notebooks/local_joint_p0_input_vjp_v1_portable_source.zip'
        FIXED_CONFIG = __CONFIG__
        OUTPUT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-P0-Input-VJP/resumable-p0-input-vjp-v1')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        ATTEMPT = OUTPUT / 'notebook_attempts' / STAMP
        ATTEMPT.mkdir(parents=True, exist_ok=True)
        RUN_OUTPUT = OUTPUT / 'run'
        WORKSPACE = Path('/content/Video-WM-P0-' + STAMP)
        PYTHON = sys.executable
        def write_json(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\\n')
            os.replace(tmp, path)
        if (OUTPUT / 'fixed_config.json').exists():
            FIXED_CONFIG = json.loads((OUTPUT / 'fixed_config.json').read_text())
        else:
            write_json(OUTPUT / 'fixed_config.json', FIXED_CONFIG)
        fixed_slots = dict(status='SETUP_STARTED', branches={b: dict(status='MISSING', reason='not_run') for b in ('conditional', 'unconditional')}, scientific_pass=False)
        write_json(ATTEMPT / 'fixed_slots.json', fixed_slots)
        def fail(stage, exc):
            write_json(ATTEMPT / 'failure.json', dict(stage=stage, reason=repr(exc), retry=False, traceback=traceback.format_exc()))
            fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
            write_json(ATTEMPT / 'fixed_slots.json', fixed_slots)
        def logged(command, stage, *, cwd=None):
            child = None
            try:
                with (ATTEMPT / 'execution.log').open('a') as log:
                    print(stage, flush=True)
                    child = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1, start_new_session=True)
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    return child.wait()
            except BaseException as exc:
                fail(stage, exc)
                raise
            finally:
                if child is not None:
                    if child.poll() is None:
                        try: os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try: os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            child.wait(timeout=10)
                    if child.stdout is not None: child.stdout.close()
        print('Fixed P0 task:', OUTPUT)
        """.replace("__CONFIG__", repr(config))))
    cells.append(code("""
        import io, urllib.request, zipfile
        try:
            with urllib.request.urlopen(SOURCE_URL) as response:
                package = response.read()
            WORKSPACE.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(package)) as archive:
                for member in archive.infolist():
                    target = (WORKSPACE / member.filename).resolve()
                    if not target.is_relative_to(WORKSPACE.resolve()):
                        raise ValueError('unsafe source ZIP path: ' + member.filename)
                    if member.is_dir(): target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            for module_name in list(sys.modules):
                if module_name.split('.', 1)[0] in ('main', 'runtime', 'experiments'):
                    sys.modules.pop(module_name, None)
            importlib.invalidate_caches()
            sys.path.insert(0, str(WORKSPACE))
            print('Source location:', SOURCE_URL, WORKSPACE)
        except BaseException as exc:
            fail('SOURCE_DOWNLOAD', exc)
            raise
        """))
    cells.append(code("""
        try:
            from runtime.wan.local_joint_p0_input_vjp_v1 import needs_model_work
            NEEDS_MODEL = needs_model_work(RUN_OUTPUT)
            probe = 'import torch, numpy' + ('; from diffusers import WanTransformer3DModel' if NEEDS_MODEL else '')
            rc = logged([PYTHON, '-u', '-c', probe], 'DEPENDENCY_IMPORT')
            if rc:
                packages = ['torch', 'numpy'] + (['diffusers', 'transformers', 'accelerate', 'safetensors', 'huggingface-hub'] if NEEDS_MODEL else [])
                rc = logged([PYTHON, '-m', 'pip', 'install', *packages], 'DEPENDENCY_INSTALL')
                if rc: raise RuntimeError('dependency installation failed; see execution.log')
                if logged([PYTHON, '-u', '-c', probe], 'DEPENDENCY_IMPORT_AFTER_INSTALL'):
                    raise RuntimeError('required imports failed; see execution.log')
            if NEEDS_MODEL and str(FIXED_CONFIG.get('device', 'cuda')).startswith('cuda'):
                if logged([PYTHON, '-c', "import torch; assert torch.cuda.is_available(), 'CUDA unavailable for the adopted full-size P0'"], 'CUDA_AVAILABILITY'):
                    raise RuntimeError('CUDA unavailable; no model calls attempted')
            try:
                import torch
                write_json(ATTEMPT / 'environment_note.json', dict(python=sys.version, torch=str(torch.__version__), new_model_work=NEEDS_MODEL))
            except Exception as exc: print('Optional environment note unavailable:', repr(exc))
        except BaseException as exc:
            fail('DEPENDENCIES', exc)
            raise
        """))
    cells.append(code("""
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token: os.environ['HF_TOKEN'] = token
        except Exception as exc: print('Optional token unavailable:', type(exc).__name__)
        from runtime.wan.local_joint_p0_input_vjp_v1 import finalize_interrupted
        rc = None
        try:
            rc = logged([PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_p0_input_vjp_v1_run',
                         '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)], 'P0_FIXED_TWO_BRANCHES', cwd=WORKSPACE)
        finally:
            # Child cleanup precedes report-only takeover; never restart a model unit.
            result = finalize_interrupted(RUN_OUTPUT, 'notebook child finished/interrupted; return code ' + str(rc))
            write_json(ATTEMPT / 'execution_receipt.json', dict(returncode=rc, result_status=None if result is None else result['status'], automatic_retry=False))
        if result is None: raise RuntimeError('No recoverable P0 result; retain notebook attempt logs')
        fixed_slots.update(status=result['status'], branches=result['branches'], counts=result['counts'])
        write_json(ATTEMPT / 'fixed_slots.json', fixed_slots)
        print('P0 status:', result['status'], 'return code:', rc)
        print('Failure/unknown rows remain in the task; no retry.')
        """))
    cells.append(code("""
        result = json.loads((RUN_OUTPUT / 'result.json').read_text())
        print('Status:', result['status'], '; scientific PASS: False')
        print('Expected:', result['expected_cost'])
        print('Physical calls:', result['counts'], 'blocks:', result['block_counts'])
        for branch, row in result['branches'].items():
            print(branch, row)
        print('Resources:', result['resources'])
        print('Spool/replay:', result['replay'])
        print('Current attempt:', result.get('current_attempt'))
        print('Earlier zero-call preflight snapshots:', len(result.get('preflight_history', [])), '; retained for history; current-attempt status is separate')
        print('Failures (includes earlier preflight/report history):', result['failures'])
        print(result['ceiling'])
        print('Return whole task directory:', OUTPUT)
        """))
    for i, cell in enumerate(cells):
        cell["id"] = "p0-input-vjp-" + str(i)
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(colab={"name": NAME+"_colab.ipynb"},
                              kernelspec={"name": "python3", "display_name": "Python 3"},
                              language_info={"name": "python"}))


def main():
    (ROOT/"notebooks").mkdir(exist_ok=True)
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
