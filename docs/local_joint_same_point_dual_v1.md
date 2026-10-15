# Adopted same-point dual comparison

Both arms start from the saved Gradient-Sum `candidate_unit.pt[terminal]` S.
Original `BASE/terminal_latent.pt` remains the radius center. Each arm has a
fresh independent graph, one scalar VJP, one fixed .10 step and one ordinary decode.

NEXT_MEAN maximizes `mean(m[i] for i in R16)+c0-max(c1..c21)`, with original R16
`[0,1,4,5,7,9,12,13,16,17,18,20,27,29,30,31]`. Raw scalar weights are 1:1.
NEXT_ALLBIT minimizes the exact historical `local_joint_all_bit_v1.scalar_loss`:
`.8*mean(H_.02(m[0:32]))+.2*H_.01(trueG)`. H is zero at/above its target,
quadratic between zero and target, and `target/2-x` for nonpositive x.
Its derivative is continuous at zero and the target. All32 terms remain present.
The ALLBIT direction is the negative masked loss gradient; MEAN is the positive
masked scalar gradient. Normalize the single combined gradient only.
The old All-Bit .5 updater and projection are never called. No selection,
scan, rollback or additional step. Actual saved displacement from BASE stays ≤1.

MEAN requires a unique active wrong-state maximum. ALLBIT reuses historical
tie semantics: an exact tie stops only when G<gamma; inactive state ties do not
block payload descent. No new tie tolerance. Unique branch changes are recorded
and trueG follows the actual maximum of all21 wrong states. Zero gradient and
undefined active ties retain forward q/readout and fixed missing endpoint rows;
they do not invent another direction. Nonfinite tensors are execution/data failures.
Mu/gamma are adopted ALLBIT loss parameters and descriptive result fields; no
new result PASS threshold exists. Total loss/mean are auxiliary to all32 outcomes.

Budget without interruptions: 1 load, 2 gradient decodes/2 VJPs, 2 ordinary decodes;
92 gradient-forward +92 ordinary-forward chunks and 92 nominal replay chunks.
Encode/DiT/native/scheduler/codec are zero. Two new endpoints, zero independent
source videos. Replay and failures are recorded per physical attempt and cumulatively.

BASE/S saved readouts and historical ALLBIT_TWO (26/32) are reused references.
Historical 26/32 remains valid at its own point/budget and is not a same-S control.
Both new endpoints report every bit, gained/lost positives, deepest negatives,
historical hard bits [4,12,18,20,27,30], all22 correlations and trueG.
All failed/missing rows remain. Four quality pairs use each new RGB against saved
BASE/S RGB. A missing old RGB leaves that pair MISSING; no replacement decode.
Quality retains clamped float RGB [0,1], 181 frames, 180 transitions, actual peaks
and neighborhoods. Numeric differences are unclipped; display uses [-.25,.25].

Run `python -m experiments.wan_state_clock.local_joint_same_point_dual_v1_run
--config experiments/wan_state_clock/configs/local_joint_same_point_dual_v1.json
--output <new-task-run>`. The editable no-.git source copy works as well.
The matching `local_joint_same_point_dual_v1_colab.ipynb` is a fixed Run-all task.
Its builder and ordinary portable source ZIP are delivered together. The current
local package is not yet published; its GitHub companion URL needs publication
before that default remote source locator can load it.

The existing Journal saves last committed state every60 seconds. Inputs, complete
gradient units, candidate latents, ordinary RGB, raw/q/scoring, and quality artifacts
commit separately. Same Drive task Run all skips complete units; report-only repair
requires no model. Unfinished autograd graphs cannot survive Colab loss; unfinished
physical work may repeat on a user resume, counted separately and cumulatively.
Committed missing units are explicit data failures, not silent replacement calls.
There is no automatic retry loop. Normal persistence is not a tested cross-Colab recovery.

Two recipes differ in state weight as well as payload aggregation. Compare their
finite-step tradeoffs, not pure payload causality or global superiority. Neither
CPU/fake tests nor a successful run establish blind recovery, codec, FPR,
generalization, native causality or combined A/B scientific PASS. GPU is user-run.
No B64/hash/manifest/identity/version/Git-clean admission or override modes.
