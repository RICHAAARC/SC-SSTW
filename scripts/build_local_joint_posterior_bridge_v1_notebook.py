"""Build the direct Run-all residual-compensated posterior bridge notebook."""
from __future__ import annotations

import json

from scripts import build_local_joint_ball_probe_v1_notebook as previous
from runtime.wan.local_joint_posterior_bridge_v1 import EXPECTED
from main.tube_state.local_joint_posterior_bridge_v1 import VIEWS


ROOT = previous.ROOT
NAME = "local_joint_posterior_bridge_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/" + NAME + ".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/" + NAME + ".py",
    "runtime/wan/" + NAME + ".py",
    "experiments/wan_state_clock/" + NAME + "_run.py",
    CONFIG_PATH,
    "docs/local_joint_posterior_bridge_v1.md",
)


def portable_archive():
    return previous.previous.previous.previous.previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    notebook = previous.build_notebook()
    old_config = json.loads((ROOT / previous.CONFIG_PATH).read_text())
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace("FIXED_CONFIG = " + repr(old_config), "FIXED_CONFIG = " + repr(config))
        if "EXPECTED_COST = " in source:
            start, end = source.index("EXPECTED_COST = "), source.index("OUTPUT_PARENT = ")
            source = source[:start] + "EXPECTED_COST = " + repr(EXPECTED) + "\n" + source[end:]
        for old, new in (
            (previous.NAME, NAME),
            ("Local-Joint-State-Payload-V1-Ball-Probe'", "Local-Joint-State-Payload-V1-Posterior-Bridge'"),
            ("resumable-ball-probe-v1'", "resumable-posterior-bridge-v1'"),
            ("Video-WM-Local-Joint-Ball-Probe-", "Video-WM-Local-Joint-Posterior-Bridge-"),
            (repr(previous.VIEWS), repr(VIEWS)),
            ("BALL-PROBE", "POSTERIOR-BRIDGE"),
            ("FIXED_BALL_PROBE", "FIXED_POSTERIOR_BRIDGE"),
            ("Fixed BALL-PROBE output:", "Fixed POSTERIOR-BRIDGE output:"),
        ):
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)

    markdown = previous.previous.previous.previous.previous._markdown
    code = previous.previous.previous.previous.previous._code
    notebook["metadata"]["colab"]["name"] = NAME + "_colab.ipynb"
    notebook["cells"][1] = markdown('''
    # B: one fixed residual-compensated posterior bridge

    Run all evaluates exactly one adopted candidate, or resumes the same Drive
    task. The start is the original saved JOINT/BASE terminal. The completed
    terminal-bridge `raw.pt` and `reconstruction_residual.pt` fields are reused
    under their saved meanings. Source paths are ordinary locators and do not
    create a source-integrity or dependency-version admission gate.

    The only direction is
    `d = M(raw - reconstruction_residual)` (`lambda=1`). The candidate is
    `FP32(BASE + 0.99998*d/||d||)`. The fixed 0.99998 scale is the adopted
    numerical radius rule. It is applied once and checked once; there is no scan,
    endpoint projection, second candidate, encode, VJP or solver. Missing files,
    invalid tensor shape/dtype, nonfinite values, a zero direction, mask leakage,
    or an actual FP32 BASE radius above 1 are concrete engineering failures.

    BASE and the old terminal-bridge CAPPED readouts are reused as the primary
    controls. Only POSTERIOR_BRIDGE is decoded. The run saves all 32 equal-mean(q)
    margins with all four ROI contributions, the correct and all 21 wrong state
    correlations and G, and clamped float RGB quality for the new candidate:
    aggregate, every frame, temporal residuals, actual peaks and peak
    neighborhoods. BASE and candidate quality records state that their inputs
    are decoder-output clamped float RGB in [0,1], while numeric residuals use
    the full post-clamp difference range; [-0.25,0.25] applies only to display.
    The old CAPPED run did not retain float RGB, so its saved
    aggregate BASE-quality summary is kept explicitly; no replacement decode is
    performed.

    Nominal new GPU work is **1 VAE load + 1 ordinary decode (46 chunks)** and
    **0 encode, VJP, DiT, native/scheduler step or codec**. Candidate latent is
    committed before model loading. Candidate RGB and scoring commit immediately.
    Every 60 seconds the notebook records last durable progress. A completed Run
    all loads no model; only an unfinished candidate decode can be repeated, with
    attempts, physical chunks and failures retained. There is no planned stop.

    Stable task: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Posterior-Bridge/
    resumable-posterior-bridge-v1/. All32 margins >0 and G>=0.01 are descriptive
    observations only. Every complete result ends this one-candidate run; a
    negative result neither closes all bridge mechanisms nor proves global
    radius-1 infeasibility. CPU/stub/static checks are engineering evidence; GPU
    execution is user-run.
    ''')
    notebook["cells"][4] = code('''
from runtime.wan.local_joint_posterior_bridge_v1 import model_work_remaining
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
            probe_lines += ["assert torch.cuda.is_available(), 'CUDA is needed for the unfinished ordinary decode'",
                            "row.update(cuda_available=True, device=torch.cuda.get_device_name(0))"]
    probe_lines += ["pathlib.Path(" + repr(str(probe_path)) + ").write_text(json.dumps(row))"]
    probe_code = '\\n'.join(probe_lines)
    probe_returncode = logged([PYTHON, '-u', '-c', probe_code], 'DEPENDENCY_PROBE', check=False)
    if probe_returncode:
        packages = ['torch', 'numpy', 'pillow']
        if NEEDS_MODEL:
            packages += ['diffusers', 'transformers', 'accelerate', 'safetensors',
                         'huggingface-hub', 'sentencepiece', 'ftfy']
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
''')
    notebook["cells"][5] = code('''
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token: os.environ['HF_TOKEN'] = hf_token
except Exception as exc:
    print('Optional HF token unavailable:', type(exc).__name__, flush=True)

run_returncode = None
try:
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_posterior_bridge_v1_run',
               '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_POSTERIOR_BRIDGE', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_posterior_bridge_v1 import finalize_interrupted
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    write_json(ATTEMPT_OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('POSTERIOR-BRIDGE engineering interruption; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status=result['status'], result='run/result.json',
        views=result['views'], counts=result['counts'], ordinary_chunks=result['ordinary_chunks'])
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_POSTERIOR_BRIDGE', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
''')
    notebook["cells"][6] = code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary chunks:', result['ordinary_chunks'])
print('Fixed construction:', result['steps']['construct'])
for name,row in result['views'].items():
    print(name,row['status'],'G=',row.get('state_gap'),'positive bits=',
          sum(x is not None and x>0 for x in row.get('margins',[])),'/32','reason=',row.get('reason'))
print('Descriptive all32/G target:', comparison['descriptive_target'])
print('All 21 wrong-state correlations are in comparison.json states[].')
print('All 32 margins and four ROI contributions are in comparison.json bits_32[].')
print('Quality versus BASE:', {name: None if row is None else {
    'status': row.get('status','FULL'), 'rmse': row.get('rmse'), 'max_abs': row.get('max_abs'),
    'temporal_residual_rmse': row.get('temporal_residual_rmse'),
    'temporal_residual_max_abs': row.get('temporal_residual_max_abs')}
    for name,row in comparison['quality_vs_BASE'].items()})
print(result['ceiling'])
print('Return the entire output folder, including failures:', OUTPUT)
''')
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = "posterior-bridge-" + str(index)
    return notebook


def main():
    (ROOT / "notebooks" / (NAME + "_colab.ipynb")).write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"
    )
    (ROOT / "notebooks" / (NAME + "_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
