"""Direct Run-all adopted R-MEAN-2H handoff; ordinary editable GitHub companion ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_terminal_bridge_v1_notebook as bridge
from scripts import build_local_joint_readout_m0_v1_notebook as m0
from scripts import build_local_joint_state_payload_v1_notebook as previous

ROOT = previous.ROOT
NAME = "local_joint_mean_refresh_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/local_joint_mean_refresh_v1.json"
PORTABLE_FILES = m0.PORTABLE_FILES + (
    "main/tube_state/local_joint_mean_refresh_v1.py",
    "runtime/wan/local_joint_mean_refresh_v1.py",
    "experiments/wan_state_clock/local_joint_mean_refresh_v1_run.py", CONFIG_PATH,
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
EXPECTED_COST = dict(vae_load=1, decode=6, decoder_vjp=1, encode=0, dit=0, native=0, codec=0,
    ordinary_chunks=230, gradient_forward_chunks=46, nominal_replay_chunks=46,
    windows=440, chips=7040, metrics=275)
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Mean-Refresh')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / STAMP
OUTPUT.mkdir(exist_ok=False)
RUN_OUTPUT = OUTPUT / 'run'
WORKSPACE = Path('/content/Video-WM-Local-Joint-Mean-Refresh-' + STAMP)
PYTHON = sys.executable
'''
    setup += bridge.reused_setup_functions() + '''

def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_mean_refresh_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))

fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,
    views={name: dict(status='NOT_RUN', expected_windows=88, expected_chips=1408, expected_metrics=55)
           for name in ('BASE','ONE_MEAN','ONE_COMMON','MID_MEAN','TWO_MEAN')})
write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
write_json(OUTPUT / 'fixed_config.json', FIXED_CONFIG)
write_json(OUTPUT / 'setup_receipt.json', dict(config=FIXED_CONFIG, expected_cost=EXPECTED_COST,
    output=str(OUTPUT), python=sys.version, scientific_pass=False, automatic_retry=False))
print('Fixed R-MEAN-2H output:', OUTPUT, flush=True)
'''
    source = previous.download_source_cell(NAME+"_portable_source.zip")
    environment = bridge.reused_environment().replace(
        "saved-terminal 2E/4D/one-VAE-load memory and latency remain unverified",
        "R-MEAN-2H 6D/1-decoder-VJP/one-VAE-load new-run memory, disk and latency remain unverified")
    run = '''\
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token: os.environ['HF_TOKEN'] = hf_token
except Exception as exc:
    print('Optional HF token unavailable:', type(exc).__name__, flush=True)

run_returncode = None
try:
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_mean_refresh_v1_run',
               '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_MEAN_REFRESH', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_mean_refresh_v1 import finalize_interrupted
    # Takeover is report-only and follows child/process-group reap in logged().
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    write_json(OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('R-MEAN-2H engineering interruption; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status='SUPERSEDED_BY_RUN_RESULT', result='run/result.json')
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_MEAN_REFRESH', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
'''
    summary = '''\
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Actual calls:', result['counts'])
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
        # B: adopted R-MEAN-2H, one saved JOINT terminal

        Select a CUDA runtime and **Run all**. This is the adopted two-half-step MEAN
        refresh experiment. It reuses the saved M0 MEAN and COMMON deltas and original
        JOINT terminal at the editable paths below. One frozen FP32 Wan VAE is loaded.
        No encode, DiT, native writer, generation or codec is run. SOURCE_REF and GPU
        names are informational; actual file/import/model/numeric failures are recorded.

        MID_MEAN is BASE + 0.5 * saved MEAN. At that midpoint one VJP computes the
        arithmetic mean of the original five objectives: state gap G and four weakest
        fragment bit margins. Gradients retain their original magnitudes. Normalize
        the masked average once, then add a second half step. Project only if needed
        to the L2<=1 ball centered on the original BASE. Never fill unused radius.
        Internal VAE and outer RGB clamps remain in the true gradient path. Exact causal
        checkpoint boundaries use the already-run M0 local disk spool, without detach.

        Five fixed views: BASE, ONE_MEAN, ONE_COMMON, MID_MEAN, TWO_MEAN.
        **TWO_MEAN versus ONE_MEAN is the primary comparison.** COMMON remains an
        independent one-step comparator; refreshing COMMON has not been adopted.
        An active min/max tie in either midpoint read, zero average, or a failed
        gradient leaves TWO_MEAN undefined. Independent controls and every failure
        remain. No extra step, retry, fallback direction, scan or M1/M2 is included.

        Full nondegenerate cost: **6 decode + 1 decoder VJP + 1 load**. Ordinary
        chunks230, gradient forward46 and nominal replay46 are separately counted.
        Fixed denominator: 440 windows, 7040 chips, 275 metrics. Missing calls and
        measurements stay explicit; no unused budget is automatically filled.
        M0 actually ran on L4, but this new experiment's time/memory remain unmeasured.
        One VJP still has substantial replay disk/time cost.

        Fresh UTC output: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Mean-Refresh/.
        Return the entire folder. It contains all32 margins and22 correlations per view,
        gained/lost bits, active minima, cumulative L2 and average predicted/actual
        change, raw evidence, saved tensors, quality and images for zero-based frames
        [1,44,88,112,116,120,132,176], plus all180 adjacent residual changes. No quality
        PASS threshold is invented. Average ascent does not guarantee each objective
        or all bits improve. The terminal-local result cannot establish original
        step25..49, CFG/native causality, blind decoding or FPR.

        Delivery validation is static plus CPU/stub only; real Colab has not been run
        for this delivery. Run completion is engineering status, never scientific PASS.
        '''), c(setup), c(source), c(environment), c(run), c(summary)])


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
