"""Direct Run-all M0 handoff; ordinary editable GitHub companion ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_terminal_bridge_v1_notebook as bridge
from scripts import build_local_joint_state_payload_v1_notebook as previous

ROOT = previous.ROOT
NAME = "local_joint_readout_m0_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/local_joint_readout_m0_v1.json"
PORTABLE_FILES = bridge.PORTABLE_FILES + (
    "main/tube_state/local_joint_readout_m0_v1.py",
    "runtime/wan/local_joint_readout_checkpoint_v1.py",
    "runtime/wan/local_joint_readout_m0_v1.py",
    "experiments/wan_state_clock/local_joint_readout_m0_v1_run.py", CONFIG_PATH,
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
EXPECTED_COST = dict(vae_load=1, decode=11, decoder_vjp=5, encode=0, dit=0, native=0, codec=0,
    ordinary_chunks=276, gradient_forward_chunks=230, nominal_replay_chunks=230,
    windows=528, chips=8448, metrics=330)
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / STAMP
OUTPUT.mkdir(exist_ok=False)
RUN_OUTPUT = OUTPUT / 'run'
WORKSPACE = Path('/content/Video-WM-Local-Joint-Readout-M0-' + STAMP)
PYTHON = sys.executable
'''
    setup += bridge.reused_setup_functions() + '''

def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_readout_m0_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))

fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,
    views={name: dict(status='NOT_RUN', expected_windows=88, expected_chips=1408, expected_metrics=55)
           for name in ('BASE','BRIDGE','MEAN','COMMON','FD_MINUS','FD_PLUS')})
write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
write_json(OUTPUT / 'fixed_config.json', FIXED_CONFIG)
write_json(OUTPUT / 'setup_receipt.json', dict(config=FIXED_CONFIG, expected_cost=EXPECTED_COST,
    output=str(OUTPUT), python=sys.version, scientific_pass=False, automatic_retry=False))
print('Fixed M0 output:', OUTPUT, flush=True)
'''
    source = previous.download_source_cell(NAME+"_portable_source.zip")
    environment = bridge.reused_environment().replace(
        "saved-terminal 2E/4D/one-VAE-load memory and latency remain unverified",
        "M0 11D/5-decoder-VJP/one-VAE-load memory, replay disk and latency remain unverified")
    run = '''\
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token: os.environ['HF_TOKEN'] = hf_token
except Exception as exc:
    print('Optional HF token unavailable:', type(exc).__name__, flush=True)

run_returncode = None
try:
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_readout_m0_v1_run',
               '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_READOUT_M0', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_readout_m0_v1 import finalize_interrupted
    # Takeover is report-only and follows child/process-group reap in logged().
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    write_json(OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('M0 engineering interruption; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status='SUPERSEDED_BY_RUN_RESULT', result='run/result.json')
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_READOUT_M0', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
'''
    summary = '''\
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Actual calls:', result['counts'])
print('Ordinary decode chunks:', result['ordinary_chunks'])
print('Gradient forward/replay records:', result['replay'])
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
        # B: adopted readout-driven M0, one saved JOINT

        Select a CUDA runtime and **Run all**. This reads the saved normalized JOINT terminal
        and saved capped bridge delta at the paths in the next cell. It loads one frozen FP32
        Wan VAE. No generation, encode, DiT, native writer, VAE re-encoding or codec is run.
        SOURCE_REF and input paths are ordinary editable settings; GPU names are informational.

        Fixed objectives: state gap G and the weakest signed bit margin in each of four
        fragments. Their five masked gradients are used at original magnitude. MEAN is their
        arithmetic mean; COMMON is the minimum-norm convex combination. Each defined direction
        is normalized once to L2=1. The causal cache stays in the gradient graph; exact checkpoint
        boundaries are temporarily spooled to local disk until each sequential VJP finishes.
        Real memory/disk/time for this path remain unverified. Actual I/O failures are retained.

        Six fixed views: BASE, saved BRIDGE, MEAN, COMMON, FD_MINUS and FD_PLUS. FD uses
        ±1/64 COMMON. There is no retry, direction scan, extra VJP, or fallback direction.
        An active min/max tie makes all four new views undefined. Zero mean only removes MEAN;
        zero/numerically indistinguishable COMMON removes COMMON and both FD views.
        Exact common zero excludes only all-five strictly positive first-order gains.
        Weak coordination and finite-step gains are not ruled out.

        Full nondegenerate plan: **11 decode + 5 decoder VJP + 1 load**. Ordinary chunks276,
        gradient original chunks230 and nominal replay230 are accounted separately. Degenerate
        or failed paths retain real attempted/completed counts; unused calls are never filled.
        All six slots remain: 528 windows, 8448 chips and 330 metrics, with missing evidence explicit.

        Fresh UTC output goes to `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0/`.
        Return the entire folder: raw evidence and 22 correlations/32 margins per view,
        gradients/directions, delta/terminal/float-RGB tensors, predicted/actual changes,
        gained/lost bits, float quality and fixed-frame PNGs at common display scales.
        These images are float-output diagnostics; no codec observations are implied.
        Process completion and CPU tests are engineering evidence, not scientific PASS.
        M0 cannot establish blind decoding, FPR or original trajectory/CFG/native causality.
        '''), c(setup), c(source), c(environment), c(run), c(summary)])


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
