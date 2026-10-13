"""Direct Run-all saved pure tangent common direction; ordinary source ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_all_bit_v1_notebook as previous
from runtime.wan.local_joint_tangent_common_v1 import EXPECTED
from main.tube_state.local_joint_tangent_common_v1 import VIEWS

ROOT = previous.ROOT
NAME = "local_joint_tangent_common_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/"+NAME+".py", "runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py", CONFIG_PATH)


def portable_archive(): return previous.previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    notebook = previous.build_notebook()
    old_config = json.loads((ROOT/previous.CONFIG_PATH).read_text())
    config = json.loads((ROOT/CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        source = "".join(cell["source"])
        source = source.replace("FIXED_CONFIG = "+repr(old_config),"FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in source:
            start,end = source.index("EXPECTED_COST = "),source.index("OUTPUT_PARENT = ")
            source = source[:start]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+source[end:]
        for old,new in (("local_joint_all_bit_v1",NAME),
            ("Local-Joint-State-Payload-V1-All-Bit'","Local-Joint-State-Payload-V1-Tangent-Common'"),
            ("OUTPUT = OUTPUT_PARENT / 'resumable-task-v1'","OUTPUT = OUTPUT_PARENT / 'resumable-tangent-common-v1'"),
            ("Video-WM-Local-Joint-All-Bit-","Video-WM-Local-Joint-Tangent-Common-"),
            ("('BASE','TWO_MEAN_SAVED','ALLBIT_HALF','ALLBIT_TWO')",repr(VIEWS)),
            ("ALLBIT-2H","TANGENT-COMMON"),("FIXED_ALL_BIT","FIXED_TANGENT_COMMON")):
            source = source.replace(old,new)
        cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["colab"]["name"] = NAME+"_colab.ipynb"
    notebook["cells"][1] = previous.previous._markdown('''
    # B: one saved pure tangent common direction, original BASE radius 1

    Run all executes one fixed step or resumes the same task. Read the step_size
    in the fixed config below. The source is Six-Bit-Final/resumable-six-bit-final-v1:
    inputs/fixed_inputs.pt supplies original BASE and actual CONT_FINAL;
    diagnostic_directions.pt supplies the saved FP64 tangent_common verbatim.
    Input paths and SOURCE_REF are editable locators, not integrity requirements.

    Candidate = FP32(FP64(START_FINAL) + step_size * saved_direction).
    The direction is tangent at START only. A straight step generally increases the
    BASE-relative radius; it is not motion along a constant-radius sphere.
    Validate finite tensors, original support, unit tangency and actual radius <=1.
    An invalid candidate is reported, without projection, renormalization, smaller
    step, search, rollback, radius filling or additional step.

    Three fixed views: BASE, START_FINAL, TANGENT. Nominal uninterrupted cost:
    **1 VAE load + 3 ordinary decodes (138 chunks), 0 VJP**. No generation,
    encode, DiT/native/scheduler, gradient refresh or codec. Each view retains all
    88 windows/1408 chips/55 metrics: 32 margins,22 state correlations,G.
    Total fixed denominator264 windows/4224 chips/165 metrics. Failed slots remain.

    Primary comparison is TANGENT minus START_FINAL: all six negative-bit continuous
    margin changes, all32 gained/lost bits and margins,22 correlations/G, and float
    quality. Unchanged positive-bit count alone does not establish no progress.
    When the six saved paired gradients are readable, CPU computes g_i dot both the
    planned increment and actual FP64 terminal difference. Unavailable predictions
    remain missing and never trigger a replacement VJP. Other26 margin derivatives,
    state and quality derivatives are unmeasured. Slopes are not finite-step guarantees.

    Quality is measured relative to BOTH original JOINT BASE and START_FINAL: all181
    frame and180 temporal records, actual RMSE/max_abs peak frames and neighborhoods,
    matching peak crops and fixed frames. BASE-relative quality is cumulative change
    from this already-watermarked JOINT BASE, not total distortion versus OFF.
    Numeric metrics are unclipped; residual displays use [-.25,.25] with clipped counts.
    No new quality threshold or scientific PASS. This is one seen source terminal,
    not blind recovery, FPR, codec survival or trajectory/CFG/native causality.

    Stable Drive task: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Tangent-Common/
    resumable-tangent-common-v1/. Keep this folder and Run all after runtime expiry.
    BASE/START/direction/candidate and step predictions commit before model loading.
    Each completed float RGB, raw q, scores and quality artifact commits on Drive.
    Every60 seconds saves the last committed JSON/progress, with previous fallback;
    it does NOT save an in-flight decoder and does not bound lost work to60 seconds.
    Only an unfinished decode is replayed as a whole; all physical attempts/failures
    remain and can exceed the nominal budget. Completed RGB enables CPU report/quality
    recovery. A complete rerun makes zero model calls. No activation spool is needed.

    Reuses the successful FP32 Wan ordinary decoder and reader/quality paths.
    CPU/stub/static validation is engineering evidence only. GPU is user-run.
    Return the entire task directory, including attempts, failures and checkpoints.
    ''')
    notebook["cells"][6] = previous.previous._code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary chunks:', result['ordinary_chunks'])
print('Step geometry:', result['steps']['step'])
for name, row in result['views'].items():
    print(name, row['status'], 'G=', row.get('state_gap'), 'positive bits=',
          sum(x is not None and x>0 for x in row.get('margins',[])), '/32', 'reason=', row.get('reason'))
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
for bit in (4,12,18,20,27,30):
    row = comparison['bits_32'][bit]
    print('bit',bit,'margins=',row['margins'],'predicted=',row['predicted_change'],'actual=',row['actual_change'])
for ref in ('BASE','START_FINAL'):
    quality = json.loads((RUN_OUTPUT / 'TANGENT' / ('quality_vs_' + ref + '.json')).read_text())
    print('TANGENT quality versus',ref,{key:quality.get(key) for key in
        ('rmse','max_abs','temporal_residual_rmse','temporal_residual_max_abs','peak_pixel')})
print(result['ceiling'])
print('Return entire output folder:', OUTPUT)
''')
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__": main()
