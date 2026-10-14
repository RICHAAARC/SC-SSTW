"""Build the direct Run-all BASE two-scalar alignment notebook and source ZIP."""
from __future__ import annotations

import json

from scripts import build_local_joint_six_bit_final_v1_notebook as previous
from runtime.wan.local_joint_base_alignment_v1 import EXPECTED
from main.tube_state.local_joint_base_alignment_v1 import UNITS

ROOT = previous.ROOT
NAME = "local_joint_base_alignment_v1"
CONFIG_PATH = "experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/"+NAME+".py",
    "runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py",
    CONFIG_PATH,
    "docs/"+NAME+".md",
)


def portable_archive():
    return previous.previous.previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    notebook = previous.build_notebook()
    old_config = json.loads((ROOT/previous.CONFIG_PATH).read_text())
    config = json.loads((ROOT/CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace("FIXED_CONFIG = "+repr(old_config), "FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in source:
            start, end = source.index("EXPECTED_COST = "), source.index("OUTPUT_PARENT = ")
            source = source[:start]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+source[end:]
        for old, new in (
            (previous.NAME, NAME),
            ("Local-Joint-State-Payload-V1-Six-Bit-Final'", "Local-Joint-State-Payload-V1-Base-Alignment'"),
            ("resumable-six-bit-final-v1'", "resumable-base-alignment-v1'"),
            ("Video-WM-Local-Joint-Six-Bit-Final-", "Video-WM-Local-Joint-Base-Alignment-"),
            (repr(previous.VIEWS), repr(UNITS)),
            ("SIX-BIT-FINAL", "BASE-ALIGNMENT"),
            ("FIXED_SIX_BIT_FINAL", "FIXED_BASE_ALIGNMENT"),
        ):
            source = source.replace(old, new)
        source = source.replace("fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,\n    views={name:",
                                "fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,\n    units={name:")
        cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["colab"]["name"] = NAME+"_colab.ipynb"
    markdown = previous.previous.previous._markdown
    code = previous.previous.previous._code
    notebook["cells"][1] = markdown('''
    # B: fixed BASE aggregate-payload / state alignment diagnostic

    Run all measures exactly two independent VJPs at the same saved JOINT BASE,
    or resumes unfinished units in the same Drive task. The fixed payload scalar
    is the mean of message-aligned margins at historical indices
    **[0,1,4,5,7,9,12,13,16,17,18,20,27,29,30,31]**. The fixed state scalar is
    **correlation[0]-correlation[16]**. A fresh readout may have a different
    nonpositive roster or active wrong offset; the notebook records that difference
    without redefining a scalar or rejecting the observation.

    The BASE and historical CONT_FINAL tensors come from the existing Six-Bit-Final
    `run/inputs/fixed_inputs.pt` fields `base` and `final`. Old and residual-compensated
    bridge directions use the completed Terminal-Bridge `tensors/raw.pt` and
    `tensors/reconstruction_residual.pt`. These are the actual saved large-tensor
    locations used by the earlier runtimes. Paths and SOURCE_REF are ordinary editable
    locators; source/version identity is not an execution gate.

    Each scalar uses a fresh decoder graph through the proven checkpoint/replay disk
    spool path. Every forward saves q, all 32 margins, all 22 correlations, fresh G,
    fixed-scalar value, active offsets, and the fresh nonpositive roster. Each complete
    unit then atomically saves the raw gradient, masked gradient, both norms, outside-mask
    norm, replay ledger and call receipts. Zero, adverse, or conflicting gradients are
    complete results; there is no sign, margin, protection or scientific-success gate.

    CPU analysis projects the two unnormalized masked gradients onto three fixed unit
    directions: `M(raw)`, `M(raw-reconstruction_residual)`, and `CONT_FINAL-BASE`.
    It saves masked direction norms, pairwise unit-direction cosines, and the 2x3 raw
    scalar slopes, gradient norms and dimensionless gradient/direction cosines, plus the
    two-gradient Gram and cosine for payload/state competition. Direction
    normalization defines unit displacement only; gradients are never normalized or
    relabeled as another scalar unit. A zero gradient has zero slopes and null cosines;
    a zero saved direction has no unit direction, so its slope and cosine are null.

    Nominal work is **1 VAE load + 2 gradient decodes + 2 VJPs**, 92 forward and
    92 nominal recompute chunks, 176 windows, 2,816 chips and 110 metrics at one
    source point. Ordinary decode, encode, DiT, native/scheduler, codec and candidate
    generation are all zero. No scan, update, 17-VJP extension or stop/restart step.
    Units run serially and independently.

    Stable task: `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Base-Alignment/`
    `resumable-base-alignment-v1/`. Every 60 seconds saves last committed progress.
    Each gradient commits immediately. Repeated Run all skips complete units and a
    fully complete task needs no VAE/CUDA. An interrupted unfinished unit restarts as
    one unit; attempts, failure history and observed physical replay counts accumulate.

    Completion means the fixed observations and projection report were saved. This
    known-message one-point diagnostic cannot establish per-bit common improvement,
    which individual bit is sacrificed, protection of the other 16 bits, ROI-specific
    gradient conflict, finite-step branch switching, blind recovery, codec survival,
    generalization or scientific PASS. GPU/Colab execution is user-run. Return the
    entire stable task directory.
    ''')
    notebook["cells"][6] = code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
print('Engineering status:', result['status'])
print('Nominal budget:', result['expected_cost'])
print('Cumulative physical calls:', result['counts'])
print('This Run all:', result.get('current_attempt_counts', {}))
print('Fixed totals:', result.get('totals'))
for name, row in result['units'].items():
    readout = row.get('readout', {})
    print(name, row['status'], row.get('gradient_norms'),
          'fresh nonpositive=', readout.get('fresh_nonpositive_bits'),
          'fresh active wrong=', readout.get('fresh_active_wrong_offsets'),
          'reason=', row.get('reason'))
for name, replay in result['replay'].items():
    print('Replay:', name, replay['counts'], replay.get('boundary_storage', {}))
diagnostic_path = RUN_OUTPUT / 'diagnostic.json'
if diagnostic_path.exists():
    diagnostic = json.loads(diagnostic_path.read_text())
    print('Direction geometry:', diagnostic['direction_geometry'])
    print('Unit-direction cosines:', diagnostic['unit_direction_cosines'])
    print('Two-gradient geometry:', diagnostic['gradient_geometry'])
    print('2x3 projections:', diagnostic['projections'])
print(result['ceiling'])
print('Return the entire output folder, including failures:', OUTPUT)
''')
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = "base-alignment-"+str(index)
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__ == "__main__":
    main()
