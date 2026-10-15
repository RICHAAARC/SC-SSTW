"""Build the direct Run-all fixed saved-gradient sum notebook."""
from __future__ import annotations

import json

from scripts import build_local_joint_posterior_bridge_v1_notebook as previous
from runtime.wan.local_joint_gradient_sum_v1 import EXPECTED
from main.tube_state.local_joint_gradient_sum_v1 import VIEWS

ROOT = previous.ROOT
NAME = "local_joint_gradient_sum_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/" + NAME + ".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/" + NAME + ".py",
    "runtime/wan/" + NAME + ".py",
    "experiments/wan_state_clock/" + NAME + "_run.py",
    CONFIG_PATH,
    "docs/" + NAME + ".md",
)


def portable_archive():
    return previous.previous.previous.previous.previous.previous.portable_archive(PORTABLE_FILES)


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
            ("Local-Joint-State-Payload-V1-Posterior-Bridge'", "Local-Joint-State-Payload-V1-Gradient-Sum'"),
            ("resumable-posterior-bridge-v1'", "resumable-gradient-sum-v1'"),
            ("Video-WM-Local-Joint-Posterior-Bridge-", "Video-WM-Local-Joint-Gradient-Sum-"),
            (repr(previous.VIEWS), repr(VIEWS)),
            ("POSTERIOR-BRIDGE", "GRADIENT-SUM"),
            ("FIXED_POSTERIOR_BRIDGE", "FIXED_GRADIENT_SUM"),
        ):
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)

    markdown = previous.previous.previous.previous.previous.previous._markdown
    code = previous.previous.previous.previous.previous.previous._code
    notebook["metadata"]["colab"]["name"] = NAME + "_colab.ipynb"
    notebook["cells"][1] = markdown('''
    # B: one fixed raw-gradient 1:1 sum step

    Run all evaluates exactly one adopted candidate, or resumes the same Drive
    task. It reads `base`, `units/aggregate_payload.pt` and
    `units/state_G16.pt` from the completed BASE-Alignment run. The BASE clamped
    float RGB is reused from its already decoded source path. These paths and the
    editable source reference are ordinary locators; there is no source-identity
    or exact dependency-version admission gate.

    The fixed construction preserves each saved gradient in its original scalar
    units: `d=(g_A+g_G)/||g_A+g_G||`, then `SUM_STEP=BASE+0.10*d`. The weights
    are exactly 1:1. There is no separate gradient normalization, endpoint
    projection, scan, second step, rollback, alternate direction, new VJP or
    replacement BASE decode. The support mask and original BASE radius cap 1
    remain. A missing input, invalid/nonfinite tensor or zero gradient sum is an
    explicit saved engineering failure.

    BASE q, all32 margins and all22 correlations are reused from the saved
    gradient unit. Only SUM_STEP is decoded and read. Both views report every bit
    delta, every state branch, actual max-wrong offset/G, fixed G16, fixed-roster
    mean A, positive/nonpositive counts, and distances from mu=.02 and gamma=.01.
    Mu/gamma are descriptive only. Lost bits, a wrong-state branch change, G
    decrease or poor quality are retained complete observations.

    Quality uses decoder-output clamped float RGB in [0,1]. Aggregate, every
    frame, adjacent temporal residual, actual peak pixels and their neighborhoods
    are saved. Numeric residuals use the full post-clamp difference range;
    [-0.25,0.25] applies only to display.

    Nominal uninterrupted new GPU work is **1 VAE load + 1 ordinary decode
    (46 chunks)**. VJP, gradient decode, encode, DiT, native/scheduler and codec
    work are zero. Candidate latent commits before model loading, ordinary RGB
    commits before readout, and scoring commits immediately. Every 60 seconds the
    Journal records durable progress. Repeated Run all skips completed units and
    a complete run loads no model. An interrupted unfinished decode may repeat;
    physical calls, chunks, attempts and failures remain cumulative.

    Stable task: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Gradient-Sum/
    resumable-gradient-sum-v1/. This one source point and one finite step do not
    add an independent source video or independent sample. They do not establish
    blind recovery, codec survival, FPR, independent-sample behavior,
    generalization or scientific PASS. GPU/Colab/Drive execution is user-run.
    Return the entire stable task directory.
    ''')
    notebook["cells"][4] = code('''
from runtime.wan.local_joint_gradient_sum_v1 import model_work_remaining
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
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_gradient_sum_v1_run',
               '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_GRADIENT_SUM', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_gradient_sum_v1 import finalize_interrupted
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    write_json(ATTEMPT_OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('GRADIENT-SUM engineering interruption; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status=result['status'], result='run/result.json',
        views=result['views'], counts=result['counts'], ordinary_chunks=result['ordinary_chunks'])
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_GRADIENT_SUM', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
''')
    notebook["cells"][6] = code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
print('Engineering status:', result['status'])
print('Nominal uninterrupted budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary chunks:', result['ordinary_chunks'])
print('Fixed construction:', result['steps']['construct'])
for name, row in result['views'].items():
    print(name, row['status'], 'G=', row.get('state_gap'),
          'G16=', row.get('fixed_G16'), 'meanA=', row.get('fixed_mean_A'),
          'positive/nonpositive=', row.get('positive_count'), row.get('nonpositive_count'),
          'active wrong=', row.get('actual_max_wrong_offsets'), 'reason=', row.get('reason'))
print('All 32 bit deltas:', comparison['deltas']['margins'])
print('All 22 state deltas:', comparison['deltas']['correlations'])
print('Descriptive mu/gamma distances:', comparison['descriptive_distances'])
print('Quality versus BASE:', {name: None if row is None else {
    'rmse': row['rmse'], 'max_abs': row['max_abs'],
    'temporal_residual_rmse': row['temporal_residual_rmse'],
    'temporal_residual_max_abs': row['temporal_residual_max_abs'],
    'peak_events': row['peak_events']} for name, row in comparison['quality_vs_BASE'].items()})
print(result['ceiling'])
print('Return the entire output folder, including failures:', OUTPUT)
''')
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = "gradient-sum-" + str(index)
    return notebook


def main():
    (ROOT / "notebooks" / (NAME + "_colab.ipynb")).write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"
    )
    (ROOT / "notebooks" / (NAME + "_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
