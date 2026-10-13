"""Build the direct Run-all fixed guarded full-BASE-ball probe notebook."""
from __future__ import annotations

import json

from scripts import build_local_joint_protected_common_v1_notebook as previous
from runtime.wan.local_joint_ball_probe_v1 import EXPECTED
from main.tube_state.local_joint_ball_probe_v1 import VIEWS

ROOT = previous.ROOT
NAME = "local_joint_ball_probe_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/" + NAME + ".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/" + NAME + ".py",
    "runtime/wan/" + NAME + ".py",
    "experiments/wan_state_clock/" + NAME + "_run.py",
    CONFIG_PATH,
)


def portable_archive():
    return previous.previous.previous.previous.portable_archive(PORTABLE_FILES)


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
            ("local_joint_protected_common_v1", NAME),
            ("Local-Joint-State-Payload-V1-Protected-Common'", "Local-Joint-State-Payload-V1-Ball-Probe'"),
            ("resumable-protected-common-v1'", "resumable-ball-probe-v1'"),
            ("Video-WM-Local-Joint-Protected-Common-", "Video-WM-Local-Joint-Ball-Probe-"),
            (repr(previous.VIEWS), repr(VIEWS)),
            ("PROTECTED-COMMON", "BALL-PROBE"),
            ("FIXED_PROTECTED_COMMON", "FIXED_BALL_PROBE"),
        ):
            source = source.replace(old, new)
        source = source.replace("'source_run': 'resumable-ball-probe-v1'", "'source_run': 'resumable-protected-common-v1'")
        source = source.replace("CUDA is needed for unfinished decode/VJP", "CUDA is needed for the unfinished ordinary decode")
        if "def record_failure(stage, exc):" in source:
            old = """def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_ball_probe_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))
"""
            new = """def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_ball_probe_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))
        actual = json.loads((RUN_OUTPUT / 'result.json').read_text())
        fixed_slots.update(run_status=actual.get('status'),
            views=actual.get('views', fixed_slots['views']), counts=actual.get('counts'),
            ordinary_chunks=actual.get('ordinary_chunks'))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
"""
            if old not in source:
                raise RuntimeError("fixed_slots failure block changed upstream")
            source = source.replace(old, new)
        if "NEEDS_MODEL = " in source:
            source = source.replace(
                "    NEEDS_SOLVER = not (RUN_OUTPUT / 'direction_unit.pt').is_file()",
                "    NEEDS_SOLVER = not (RUN_OUTPUT / 'candidate_unit.pt').is_file()",
            )
            source = source.replace("if NEEDS_MODEL:\n        probe_lines", "if NEEDS_MODEL:\n        probe_lines")
            source = source.replace("CUDA is required for the fixed gradient/decode work", "CUDA is required for the fixed ordinary decode")
        if "if run_returncode != 0" in source:
            source = source.replace(
                "result['status'] not in ('COMPLETE', 'COMPLETE_NO_CANDIDATE', 'PAUSED_FOR_SESSION_RESTART')",
                "result['status'] != 'COMPLETE'",
            )
            source = source.replace(
                "    if result['status'] == 'PAUSED_FOR_SESSION_RESTART':\n        print('The first new gradient is saved. Delete/disconnect this Colab runtime, reconnect, and Run all again. Completed units will be skipped.', flush=True)\n",
                "",
            )
            source = source.replace(
                "fixed_slots.update(status=result['status'], result='run/result.json')",
                "fixed_slots.update(status=result['status'], result='run/result.json', "
                "views=result['views'], counts=result['counts'], ordinary_chunks=result['ordinary_chunks'])",
            )
        cell["source"] = source.splitlines(keepends=True)

    markdown = previous.previous.previous.previous._markdown
    code = previous.previous.previous.previous._code
    notebook["metadata"]["colab"]["name"] = NAME + "_colab.ipynb"
    notebook["cells"][1] = markdown('''
    # B: one fixed guarded full-BASE-ball endpoint probe

    Run all solves and evaluates exactly one adopted endpoint, or resumes the same
    Drive task. It reuses the original BASE, numerical START_FINAL, support mask,
    key/message, six saved target gradients and four saved protection gradients from
    the completed Protected-Common run. Source paths are editable locators. There is
    no source digest, manifest, Git or dependency-version gate.

    The CPU SOCP maximizes the worst predicted endpoint margin among bits
    [4,12,18,20,27,30] inside the original BASE-relative radius-1 ball. It requires
    saved START linear changes bit29>=0, B<=0, G>=0 and Q<=0. This is the ambient
    masked ball problem: the displacement from START can cancel its component outside
    the ten-gradient span. It is not the prior common-gain direction or a fixed tangent
    step. Six target rows remain in raw margin units.

    The solver uses one predeclared radius interior margin 2e-5 and one normalized
    guard interior margin 2e-5 solely to survive FP32 candidate storage. It solves once,
    performs no margin search, and directly measures the stored candidate's radius and
    linear residuals. It never projects the endpoint, increases the cap, relaxes a
    guard, adds another candidate or refreshes a gradient. A numerical failure is kept
    as an engineering failure.

    BASE, START_FINAL and PROTECTED RGB/readouts are copied from the completed source
    run. Only BALL_PROBE is decoded. All four views retain all32 bit margins, all22
    state correlations and G, the original protection quantities, full frame and
    temporal quality against BASE and START_FINAL, peak pixels and neighborhoods.
    Positive-bit loss, wrong-state behavior, quality degradation and B leaving its log
    domain are saved and reported; they do not censor the fixed result.

    Nominal new GPU work is **1 VAE load + 1 ordinary decode (46 chunks)** and
    **0 VJP, encode, DiT, native/scheduler step or codec**. Candidate latent commits
    before model loading. Candidate RGB and scoring units commit immediately. Every
    60 seconds the notebook records last durable progress. Completed units are skipped;
    only an unfinished candidate decode may be replayed, and failures/attempts remain.
    There is no planned stop between units.

    Stable task: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Ball-Probe/
    resumable-ball-probe-v1/. One saved START and its local linear model do not prove
    finite-step protection, full32 recovery, all wrong-offset protection, per-bit
    protection of the other25, blind decoding, codec survival, FPR, generalization or
    scientific PASS. CPU/stub/static checks are engineering evidence; GPU is user-run.
    Return the complete task directory, including failures, attempts and checkpoints.
    ''')
    notebook["cells"][6] = code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary chunks:', result['ordinary_chunks'])
print('Fixed solve:', result['steps']['solve'])
for name,row in result['views'].items():
    print(name,row['status'],'G=',row.get('state_gap'),'positive bits=',
          sum(x is not None and x>0 for x in row.get('margins',[])),'/32','reason=',row.get('reason'))
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
for bit in (4,12,18,20,27,30):
    row=comparison['bits_32'][bit]
    print('bit',bit,'margins=',row['margins'],'predicted endpoint=',row['predicted_endpoint_margin'],
          'BALL-START=',row['ball_minus_START'])
print('Protection observations:',comparison['protections'])
print('Predicted guards:',comparison['predicted'].get('normalized_signed_guard_changes'))
print('Quality versus BASE:',comparison['quality_vs_BASE'])
print(result['ceiling'])
print('Return the entire output folder, including failures:', OUTPUT)
''')
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = "ball-probe-" + str(index)
    return notebook


def main():
    (ROOT / "notebooks" / (NAME + "_colab.ipynb")).write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"
    )
    (ROOT / "notebooks" / (NAME + "_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
