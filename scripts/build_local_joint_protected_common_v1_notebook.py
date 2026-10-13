"""Run-all protected common direction with one planned session restart."""
from __future__ import annotations

import json

from scripts import build_local_joint_tangent_common_v1_notebook as previous
from runtime.wan.local_joint_protected_common_v1 import EXPECTED
from main.tube_state.local_joint_protected_common_v1 import VIEWS

ROOT = previous.ROOT
NAME = "local_joint_protected_common_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/local_joint_six_bit_final_v1.py",
    "main/tube_state/"+NAME+".py", "runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py", CONFIG_PATH)


def portable_archive(): return previous.previous.previous.portable_archive(PORTABLE_FILES)


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
        for old,new in (("local_joint_tangent_common_v1",NAME),
            ("Local-Joint-State-Payload-V1-Tangent-Common'","Local-Joint-State-Payload-V1-Protected-Common'"),
            ("resumable-tangent-common-v1'","resumable-protected-common-v1'"),
            ("Video-WM-Local-Joint-Tangent-Common-","Video-WM-Local-Joint-Protected-Common-"),
            (repr(previous.VIEWS),repr(VIEWS)),
            ("TANGENT-COMMON","PROTECTED-COMMON"),("FIXED_TANGENT_COMMON","FIXED_PROTECTED_COMMON")):
            source = source.replace(old,new)
        source = source.replace('"import json, pathlib, sys, torch, numpy, PIL",',
                                '"import json, pathlib, sys, torch, numpy, PIL",')
        if "NEEDS_MODEL = " in source:
            # Solver dependencies are needed only until direction construction.
            source = source.replace("    if NEEDS_MODEL:\n        probe_lines", "    NEEDS_SOLVER = not (RUN_OUTPUT / 'direction_unit.pt').is_file()\n    if NEEDS_SOLVER:\n        probe_lines += ['import cvxpy, clarabel']\n    if NEEDS_MODEL:\n        probe_lines",1)
            source = source.replace("        packages = ['torch', 'numpy', 'pillow']", "        packages = ['torch', 'numpy', 'pillow']\n        if NEEDS_SOLVER: packages += ['cvxpy', 'clarabel']")
        if "if run_returncode != 0" in source:
            source = source.replace("result['status'] != 'COMPLETE'", "result['status'] not in ('COMPLETE', 'COMPLETE_NO_CANDIDATE', 'PAUSED_FOR_SESSION_RESTART')")
            source = source.replace("    fixed_slots.update(status='SUPERSEDED_BY_RUN_RESULT', result='run/result.json')", "    if result['status'] == 'PAUSED_FOR_SESSION_RESTART':\n        print('The first new gradient is saved. Delete/disconnect this Colab runtime, reconnect, and Run all again. Completed units will be skipped.', flush=True)\n    fixed_slots.update(status=result['status'], result='run/result.json')")
        cell["source"] = source.splitlines(keepends=True)
    markdown = previous.previous.previous._markdown
    code = previous.previous.previous._code
    notebook["metadata"]["colab"]["name"] = NAME+"_colab.ipynb"
    notebook["cells"][1] = markdown('''
    # B: one protected common direction, same START_FINAL and .10 step

    **Run all first saves the bit29 protection gradient and stops normally.**
    When PAUSED_FOR_SESSION_RESTART appears, delete/disconnect the Colab runtime,
    reconnect to a new GPU runtime, and Run all again. Keep the same Drive folder.
    The one-time stop is recorded; a later Run all skips the completed gradient.
    Simply starting another child process is not evidence of a new Colab runtime.

    Original BASE/mask/key/message/readout and radius cap 1 are retained.
    START_FINAL is the saved CONT_FINAL used by Six-Bit-Final. Reuse its six gradients
    for bits [4,12,18,20,27,30]. Reuse BASE, START_FINAL and TANGENT RGB/readouts from
    the completed Tangent-Common task. There are no replacement decodes of old views.
    Input locations and source branch are editable. No source/identity checks are required.

    Four serial independent VJPs at START, in order:
    * positive margin29 directly;
    * B derivative: -mean25 log((m_j-.02)/(m_j(START)-.02)), excluding bit29;
    * G directly: c0-max(other correlations), requiring a unique active wrong offset;
    * Q: full RGB MSE relative to saved original BASE.
    B uses fixed START coefficients -1/[25*(m_j(START)-.02)] to differentiate once.
    This is aggregate protection, not individual protection for all25 bits. A log-domain
    failure at the endpoint is reported without epsilon, clipping or dropping the bit.
    Q uses the existing VAE float RGB mapping clamp[0,1], with FP64 residual arithmetic;
    residuals are not clipped, quantized or encoded. Q relative to START is not used.

    All gradients use the fixed mask and START-minus-BASE tangent plane. A CPU SOCP in
    their at-most-ten-dimensional span maximizes the minimum raw six-bit margin slope,
    with margin29/G nondecreasing and B/Q nonincreasing predictions, and norm<=1.
    Six target gradients are never individually normalized. Protection rows can be
    positively scaled for numerical conditioning. Zero is always feasible; unresolved
    solver status is not method infeasibility. No automatic constraint relaxation.
    Only a numerically clear positive result defines a unit direction and one .10 step.
    Candidate=FP32(FP64(START_FINAL)+.10*direction), no endpoint projection, second step,
    step scan, rollback, radius fill, cap increase or new source point. Candidate radius
    is approximately .959667. Actual FP32 increment predictions and radius are saved.

    Fixed report views BASE / START_FINAL / TANGENT / PROTECTED retain all32 margins,
    22 correlations and G: 352 windows/5632 chips/220 metrics including three reused
    views. The four gradient probe reads have a separate denominator; they are not
    independent samples. Failed, undefined and unexecuted slots remain in the report.
    If no clear positive direction exists, PROTECTED remains NOT_RUN and the result
    package is COMPLETE_NO_CANDIDATE. No extra model call tries to improve that result.

    Nominal new work without a pause: **1 load + 4 gradient decodes / 4 VJPs +
    1 candidate ordinary decode**, 184 gradient-forward and nominal replay chunks,
    46 ordinary chunks. The planned restart adds one load: expected cumulative totals
    after completion are 2 loads / 5 decodes / 4 VJPs when nothing else is interrupted.
    If no candidate is defined, ordinary decode is skipped. Physical attempts can
    exceed these budgets only when unfinished units must be recovered; all are counted.

    Uses the already-run checkpoint_decode and activation-spool path. Every60 seconds
    stores last committed progress, not the live autograd graph. Inputs, each complete
    gradient, direction, candidate latent and decoded RGB commit immediately on Drive.
    An interrupted VJP/decode restarts only that unfinished unit; complete units are
    skipped. The activation spool is temporary local storage, not a durable graph.
    Complete report recovery needs no VAE/CUDA. Call receipts and failures accumulate.

    Stable task: MyDrive/Video-WM/Local-Joint-State-Payload-V1-Protected-Common/
    resumable-protected-common-v1/. Do not change this output for the restart.
    Four-view report includes six-bit gain retention versus old TANGENT, all32 bit
    changes, bit29 versus mu=.02, other25 losses/domain failures, G versus gamma=.01,
    full frame/temporal quality versus BASE and START, peak pixels and neighborhoods.
    BASE is the already-watermarked JOINT BASE, not OFF. MSE alone is not quality PASS.
    First-order constraints can fail after real decoding; retain and report those outcomes.
    This one-point known-message diagnostic is not blind decoding, codec survival,
    FPR, trajectory/CFG/native causality, generalization or scientific PASS.

    Delivery checks are CPU/fake/static only; GPU execution is yours. Return the entire
    stable task directory including failures, attempts and checkpoint sidecars.
    ''')
    notebook["cells"][6] = code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Ordinary chunks:', result['ordinary_chunks'])
for name,row in result['gradient_units'].items():
    print('Protection unit:', name, row['status'])
for name,replay in result['replay'].items():
    print('Gradient ledger:', name, replay['counts'], replay.get('boundary_storage', {}))
print('Direction:', result['steps']['direction'])
print('Step:', result['steps']['step'])
for name,row in result['views'].items():
    print(name,row['status'],'G=',row.get('state_gap'),'positive bits=',
          sum(x is not None and x>0 for x in row.get('margins',[])),'/32','reason=',row.get('reason'))
if result['status'] == 'PAUSED_FOR_SESSION_RESTART':
    print('First new gradient is saved. Delete/disconnect runtime, reconnect, then Run all again with this same task folder.')
else:
    comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
    for bit in (4,12,18,20,27,30):
        row=comparison['bits_32'][bit]
        print('bit',bit,'margins=',row['margins'],'gain=',row['protected_gain'],'retained=',row['retained_tangent_gain_ratio'])
    print('Protection observations:',comparison['protections'])
    print('Quality versus BASE:',comparison['quality_vs_BASE'])
    print('Resume observation:',result.get('resume_observation'))
print(result['ceiling'])
print('Return the entire output folder, including failures:', OUTPUT)
''')
    for index,cell in enumerate(notebook["cells"]):
        cell["id"] = "protected-common-"+str(index)
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__": main()
