"""Direct Run-all six-bit FINAL diagnostic with ordinary source ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_all_bit_v1_notebook as previous
from runtime.wan.local_joint_six_bit_final_v1 import EXPECTED
from main.tube_state.local_joint_six_bit_final_v1 import VIEWS

ROOT = previous.ROOT
NAME = "local_joint_six_bit_final_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/"+NAME+".py", "runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py", CONFIG_PATH)


def portable_archive():
    return previous.previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    notebook = previous.build_notebook()
    old_config = json.loads((ROOT/previous.CONFIG_PATH).read_text())
    config = json.loads((ROOT/CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace("FIXED_CONFIG = "+repr(old_config),"FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in source:
            start,end = source.index("EXPECTED_COST = "),source.index("OUTPUT_PARENT = ")
            source = source[:start]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+source[end:]
        for old,new in (
            ("local_joint_all_bit_v1",NAME),
            ("Local-Joint-State-Payload-V1-All-Bit'","Local-Joint-State-Payload-V1-Six-Bit-Final'"),
            ("OUTPUT = OUTPUT_PARENT / 'resumable-task-v1'","OUTPUT = OUTPUT_PARENT / 'resumable-six-bit-final-v1'"),
            ("Video-WM-Local-Joint-All-Bit-","Video-WM-Local-Joint-Six-Bit-Final-"),
            ("('BASE','TWO_MEAN_SAVED','ALLBIT_HALF','ALLBIT_TWO')",repr(VIEWS)),
            ("ALLBIT-2H","SIX-BIT-FINAL"),("FIXED_ALL_BIT","FIXED_SIX_BIT_FINAL")):
            source = source.replace(old,new)
        cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["colab"]["name"] = NAME+"_colab.ipynb"
    notebook["cells"][1] = previous.previous._markdown('''
    # B: fixed CONT_FINAL six-bit derivative diagnostic

    Run all measures only fixed bits **[4,12,18,20,27,30]** or resumes unfinished units.
    The input is the actual `terminal` field of Continue/resumable-task-v1-2x010/run/step_2_unit.pt.
    The old gradient checkpoint's input_terminal is MID and must not be used.
    Original BASE comes from that run's inputs/terminal_latent.pt; saved FINAL readouts
    come from result.json. Input paths and SOURCE_REF are editable locators.

    Stable independent output:
    MyDrive/Video-WM/Local-Joint-State-Payload-V1-Six-Bit-Final/resumable-six-bit-final-v1/.
    Keep this folder when reopening. Each Run all appends attempt logs; complete paired
    gradient units are skipped. All six saved means CPU report-only with zero VAE calls.
    Missing/corrupt state or invalid committed input/bit pairs is reported explicitly,
    never silently treated as permission to repeat the full experiment.

    Uninterrupted nominal budget: **1 load + 6 gradient decode + 6 VJP**, six independent
    fresh graphs in serial; **276 forward + 276 nominal recompute chunks**.
    Ordinary decode/encode/DiT/native/scheduler/codec: **0**. No shared-graph multi-backward.
    Each forward retains **55 quantities: 32 margins,22 correlations,G; 88 windows/1408 chips**,
    payload denominators [24,24,20,20]. Across six rows:528 windows/8448 chips/330 metrics;
    repeated same-point measurements are not six independent samples. Failed/missing rows remain.

    Differentiate the positive margin exactly at flat index i. Mask once and retain the
    unnormalized h_i. Compare each 55-value readout to saved FINAL; differences are diagnostics.
    Original key/message/mask/rho/cap/readout remain. The full scalar loss gradient equals
    -0.025 sum(h_i) only if payload_active is exactly the six bits, all six remain negative,
    state_active is false, and coefficients apply on every read. Fragment-min and wrong-offset
    branches are not this scalar-loss active set. Otherwise retain derivatives without that claim.

    CPU reports raw Gram/cosines/norms, aggregate slopes, radial/tangential decomposition,
    and the unnormalized simplex minimum-norm common directions, with primal/dual/KKT
    residuals and FP64 error bounds. Strict common direction is numerically supported only
    if all six slope lower bounds remain positive. Otherwise NUMERICALLY_UNRESOLVED;
    a near-zero solution or negative cosine does not establish infeasibility.
    Pure tangent uses u=(FINAL-BASE)/r, t_i=h_i-(h_i dot u)u. Ball distances are geometry,
    not gain predictions or steps to cross zero. Numerical bounds concern stored gradients,
    not the unknown error of the FP32 model derivative.

    No update/candidate/scan/rollback/cap increase/COMMON refresh/M1/M2. Other26 positive-bit,
    state and quality derivatives are unmeasured; no protection or finite-budget reachability
    claim. This known-message local diagnostic is not scientific PASS or route infeasibility.

    Inputs are backed up before model load. Each complete gradient/readout/input/bit unit
    commits immediately; every60 seconds saves the last committed state/progress and retains
    the previous readable fallback. In-flight graphs restart only as whole unfinished units.
    Activation spool is temporary, not a cross-session autograd checkpoint. Cumulative physical
    calls, failures and current-attempt counts are separate; interruption can exceed nominal cost.

    Reuses the successful Wan FP32 causal decoder/checkpoint/spool path. Historical single-unit
    spool peak ~64.53GiB; serial six units reuse capacity, while total per-direction I/O is ~387.2GiB.
    Historical GPU peak allocated19.01GiB/reserved21.72GiB; approximate six-unit time44 minutes
    plus load/readout/save. These are references, not guarantees or GPU model gates.
    Validation is CPU/fake/static; real GPU/Colab execution is user-run. Return the entire folder.
    ''')
    notebook["cells"][6] = previous.previous._code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Fixed totals:', result.get('totals'))
for name, row in result['views'].items():
    print(name, row['status'], row.get('gradient_norms'), row.get('reason'))
for name, replay in result['replay'].items():
    print(name, replay['counts'], replay.get('boundary_storage', {}))
diagnostic_path = RUN_OUTPUT / 'diagnostic.json'
if diagnostic_path.exists():
    diagnostic = json.loads(diagnostic_path.read_text())
    print('Full current scalar-loss gradient formula valid:', diagnostic['loss_gradient_formula_valid_on_all_six_reads'])
    print('Common:', diagnostic['common']['status'], 'Tangent:', diagnostic['tangent']['status'])
    print('Numerical scope:', diagnostic['numerical_error_scope'])
print(result['ceiling'])
print('Return entire output folder:', OUTPUT)
''')
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
