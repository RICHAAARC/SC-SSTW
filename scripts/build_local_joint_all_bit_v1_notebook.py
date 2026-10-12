"""Direct Run-all adopted all-bit two-half-step handoff; ordinary editable GitHub companion ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_terminal_bridge_v1_notebook as bridge
from scripts import build_local_joint_readout_m0_v1_notebook as m0
from scripts import build_local_joint_state_payload_v1_notebook as previous

ROOT = previous.ROOT
NAME = "local_joint_all_bit_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/local_joint_all_bit_v1.json"
PORTABLE_FILES = m0.PORTABLE_FILES + (
    "main/tube_state/local_joint_all_bit_v1.py",
    "runtime/wan/local_joint_all_bit_v1.py",
    "runtime/wan/local_joint_all_bit_resume_v1.py",
    "runtime/wan/local_joint_all_bit_quality_v1.py",
    "experiments/wan_state_clock/local_joint_all_bit_v1_run.py", CONFIG_PATH,
)


def portable_archive():
    return previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    config = json.loads((ROOT/CONFIG_PATH).read_text())
    c, m = previous._code, previous._markdown
    setup = f'''\
from pathlib import Path
import datetime, json, os, signal, subprocess, sys, time, traceback
SOURCE_REF = 'dev/local-joint-state-payload-v1'  # Editable GitHub branch/tag/ref.
FIXED_CONFIG = {config!r}
EXPECTED_COST = dict(vae_load=1, decode=6, decoder_vjp=2, encode=0, dit=0, native=0, scheduler=0, codec=0,
    ordinary_chunks=184, gradient_forward_chunks=92, nominal_replay_chunks=92,
    windows=352, chips=5632, metrics=220)
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / 'resumable-task-v1'  # Same logical task on every Run all.
OUTPUT.mkdir(exist_ok=True)
ATTEMPT_OUTPUT = OUTPUT / 'notebook_attempts' / STAMP
ATTEMPT_OUTPUT.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT = OUTPUT / 'run'
WORKSPACE = Path('/content/Video-WM-Local-Joint-All-Bit-' + STAMP)
PYTHON = sys.executable
'''
    setup += bridge.reused_setup_functions().replace("OUTPUT /", "ATTEMPT_OUTPUT /") + '''

def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_all_bit_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))

fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,
    views={name: dict(status='NOT_RUN', expected_windows=88, expected_chips=1408, expected_metrics=55)
           for name in ('BASE','TWO_MEAN_SAVED','ALLBIT_HALF','ALLBIT_TWO')})
write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
if (OUTPUT / 'fixed_config.json').exists():
    FIXED_CONFIG = json.loads((OUTPUT / 'fixed_config.json').read_text())
else:
    write_json(OUTPUT / 'fixed_config.json', FIXED_CONFIG)
write_json(ATTEMPT_OUTPUT / 'setup_receipt.json', dict(config=FIXED_CONFIG, expected_cost=EXPECTED_COST,
    output=str(OUTPUT), python=sys.version, scientific_pass=False, automatic_retry=False))
print('Fixed ALLBIT-2H output:', OUTPUT, flush=True)
'''
    source = previous.download_source_cell(NAME+"_portable_source.zip").replace(
        "    sys.path.insert(0, str(WORKSPACE))",
        """    # A repeated Run all may retain packages from an earlier extracted workspace.
    import importlib
    for module_name in list(sys.modules):
        if module_name.split('.', 1)[0] in ('main', 'runtime', 'experiments'):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    sys.path.insert(0, str(WORKSPACE))""").replace("OUTPUT / 'source_receipt.json'", "ATTEMPT_OUTPUT / 'source_receipt.json'")
    environment = '''\
from runtime.wan.local_joint_all_bit_v1 import model_work_remaining
try:
    NEEDS_MODEL = model_work_remaining(RUN_OUTPUT)
    probe_path = OUTPUT / 'dependency_probe.json'
    probe_lines = [
        "import json, pathlib, sys, torch, numpy, PIL",
        "row = dict(python=sys.version, torch=str(torch.__version__), model_work_remaining=" + repr(NEEDS_MODEL) + ")",
    ]
    if NEEDS_MODEL:
        probe_lines += ["from diffusers import AutoencoderKLWan"]
        if str(FIXED_CONFIG.get('device', 'cuda')).startswith('cuda'):
            probe_lines += ["assert torch.cuda.is_available(), 'CUDA is needed for unfinished decode/VJP'",
                            "row.update(cuda_available=True, device=torch.cuda.get_device_name(0))"]
    probe_lines += ["pathlib.Path(" + repr(str(probe_path)) + ").write_text(json.dumps(row))"]
    probe_code = '\\n'.join(probe_lines)
    probe_returncode = logged([PYTHON, '-u', '-c', probe_code], 'DEPENDENCY_PROBE', check=False)
    if probe_returncode:
        packages = ['torch', 'numpy', 'pillow']
        if NEEDS_MODEL:
            packages += ['diffusers', 'transformers', 'accelerate', 'safetensors', 'huggingface-hub', 'sentencepiece', 'ftfy']
        logged([PYTHON, '-m', 'pip', 'install', *packages], 'DEPENDENCY_REPAIR')
        logged([PYTHON, '-u', '-c', probe_code], 'DEPENDENCY_REPROBE')
    dependency_check_returncode = logged([PYTHON, '-m', 'pip', 'check'], 'DEPENDENCY_REPORT', check=False)
    environment_receipt = json.loads(probe_path.read_text())
    environment_receipt.update(dependency_check_returncode=dependency_check_returncode,
        source_and_versions_are_location_notes=True, checkpoint_interval_seconds=60)
    write_json(ATTEMPT_OUTPUT / 'environment_receipt.json', environment_receipt)
    print('Remaining model work:', NEEDS_MODEL, '; task:', OUTPUT, flush=True)
except BaseException as exc:
    record_failure('ENVIRONMENT_SETUP', exc)
    raise
'''
    run = '''\
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token: os.environ['HF_TOKEN'] = hf_token
except Exception as exc:
    print('Optional HF token unavailable:', type(exc).__name__, flush=True)

run_returncode = None
try:
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_all_bit_v1_run',
               '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_ALL_BIT', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_all_bit_v1 import finalize_interrupted
    # Takeover is report-only and follows child/process-group reap in logged().
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    write_json(ATTEMPT_OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('ALLBIT-2H engineering interruption; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status='SUPERSEDED_BY_RUN_RESULT', result='run/result.json')
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_ALL_BIT', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
'''
    summary = '''\
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary decode chunks:', result['ordinary_chunks'])
for name, replay in result['replay'].items():
    print('Gradient ledger:', name, replay['counts'], replay.get('boundary_storage', {}))
for name, view in result['views'].items():
    print(name, view['status'], 'G=', view.get('state_gap'),
          'positive bits=', sum(x is not None and x>0 for x in view.get('margins', [])), '/32',
          'reason=', view.get('reason'))
print(result['ceiling'])
print('Return the entire output directory, including failures:', OUTPUT)
'''
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(accelerator="GPU",
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3"), colab=dict(name=NAME+"_colab.ipynb", provenance=[])),
        cells=[c("from google.colab import drive\ndrive.mount('/content/drive')"), m('''
        # B: adopted all-32-bit margin deficit, two fresh half steps

        **Run all resumes the same logical task automatically.** Use a CUDA runtime
        while decode/VJP work remains. CPU-only scoring, quality and report recovery
        does not load a VAE or require CUDA. A finished task only displays its report.
        A new uninterrupted full task loads one frozen FP32 Wan VAE.
        The editable input paths locate the original saved JOINT BASE and the completed
        Mean Refresh TWO_MEAN delta. The saved comparator is used verbatim; its absence
        is retained as MISSING and does not stop the new path. No old experiment is rerun.
        No encode, DiT, native writer, scheduler, generation or codec is run.
        SOURCE_REF and GPU names are informational; actual file/import/model/numeric
        failures are recorded. The source is an ordinary editable companion ZIP.

        Adopted loss: L=0.8 mean_32 H_0.02(margin) + 0.2 H_0.01(G), where H_a(x) is
        zero for x>=a, (a-x)^2/(2a) for 0<x<a, and a/2-x for x<=0. All 32 terms enter;
        near-zero positives are softly protected, and terms reactivate when below target.
        State effort becomes zero at G>=0.01. These are optimization preferences,
        not detection or quality thresholds. Lower L does not guarantee bit coverage,
        G maintenance or finite-step improvement; a final-step drop is not auto-repaired.

        Starting at original BASE, compute **two fresh scalar decoder VJPs**, at BASE
        and the new half-step respectively. For each, mask once, descend along the
        normalized scalar gradient by 0.5, and project the cumulative delta only when
        needed into the BASE-centered L2<=1 ball. Never fill unused radius. Internal
        VAE/RGB clamps and causal cache gradients remain in the already-run checkpoint
        path. Each VJP has its own disk/replay ledger and finally cleanup.

        Four fixed views: **BASE, TWO_MEAN_SAVED, ALLBIT_HALF, ALLBIT_TWO**.
        **ALLBIT_TWO versus TWO_MEAN_SAVED is primary.** All finite candidates remain
        even when worse. No method rollback, scan, extra step, cap increase,
        COMMON refresh or M1/M2. Zero masked gradient or a wrong-offset max tie with
        G<0.01 leaves dependent new views undefined. Payload-min ties and inactive
        state ties do not block this C1 loss. Two valid opposing directions yielding
        cumulative zero are valid and decoded normally.

        Full nondegenerate cost: **1 load + 6 decode + 2 decoder VJP**. Ordinary chunks
        184, gradient forward92 and nominal replay92 are counted separately. Fixed
        denominator: **352 windows, 5632 chips, 220 original metrics**. Missing and
        failed slots remain; repeat gradient-forward reads are diagnostics, not samples.
        Six decodes with two VJPs cost more than the old six-decode/one-VJP run. The
        reused single-VJP path had about64.53GiB disk peak; two sequential VJPs reuse
        temporary capacity but add two read/write streams. Resume overhead remains unmeasured.

        Stable output: MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/resumable-task-v1/.
        Each Run all uses this same folder and appends attempt logs. Older UTC runs
        are untouched; no migration or rerun is needed. Use a different task folder
        only when deliberately starting a separate task.

        Inputs, complete gradients with their actual input terminal, step delta/terminal,
        float RGB, raw observations, scores and quality are committed as they finish.
        Every 60 seconds a recoverable state snapshot and progress are saved on Drive,
        retaining a previous readable snapshot. After expiry, open the notebook and
        Run all again. Saved complete units are skipped, even if /content was erased.
        In-flight decode/VJP cannot resume inside autograd: only that unfinished unit
        is recomputed from its saved input. Every attempt, call and failure remains;
        physical cost may exceed the ordinary 1 load/6 decode/2 VJP after interruption.
        Step 2 never applies a cached gradient at a different input.

        Return the entire folder. It includes raw-before-truth observations, all32
        margins, gained/lost bits and deficits,22 correlations/G, loss and coefficients,
        intended delta and actual saved-terminal displacements, per-step scalar predicted
        and actual changes, gradients, tensors, actual calls and resource records.
        Two gradient-forward55-value readouts are compared with ordinary BASE/HALF;
        finite differences are diagnostic and never identity gates.

        All181 frame and180 temporal quality records are saved. Every nonBASE view's
        actual frame-RMSE/max_abs and temporal-RMSE/max_abs peak neighborhoods join the
        original fixed frames, producing the same frame indices across available views.
        Peak pixel coordinates and matching crops are saved. RGB uses [0,1], residuals
        use [-0.25,0.25] with clipped-pixel counts; numerical metrics stay unclipped.
        No quality PASS is inferred. Same radius ceiling does not mean equal spent norm.

        Validation for this delivery is CPU/static/stub only. Actual GPU execution is
        user-run. The float known-message descriptive condition stays G>0 and32/32
        positive margins; this terminal-local diagnostic establishes no blind decoding,
        codec, FPR, trajectory/CFG/native causality or scientific PASS.
        '''), c(setup), c(source), c(environment), c(run), c(summary)])


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
