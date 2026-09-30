# Video-Temporal-Sync-Multi-V1 implementation freeze

A1 local implementation is ready for same-version independent review. A2/A3
formal review, A4 synthesis and A5 milestone audit are pending for this snapshot.
All task files are new; prior GROW/bridge sources and historical evidence remain
unchanged. Base: `b255029ec508631f38a124ac49712a5b9299d2a1`, branch
`dev/video-trajectory-blind-validation-v1`.

## Fixed implementation

OFF/PAYLOAD_MULTI/PILOT_MULTI use three independent full native histories. The
marked arms perform 25 fresh conditional-clean masked-mean gradients at indices
25..49, followed by the unchanged Flow mapping, FP32 CFG5 and native step. The
pre25 state/history/conditional/unconditional identity is checked. Per-update
N/eta are44160/5520 and47104/5888; there is no division by25 or total-budget
matching. Same-input operator scaling does not imply equal later cross-arm
deltas or repeated0.875 contraction through the actual trajectory.

The five fixed received views are full181 and crops4/5/6/7 of length129. The
unchanged length-driven global search has30 search slots and1278 candidates.
102 primary phase/key observations retain full R45 and crop R31 support. Truth
joins use the corresponding view start only after blind evidence is persisted.
Normal rejection is valid completion; it supplies no accepted payload/BER.
Five pilot-positive and25 related negative controls are not independent FPR
samples. Global exact localization is not a prerequisite for a future state path.

The new interface adds102 atomic hashed window files/3270 rows and retains51
normalized tensors. Each window exposes64 pilot coefficients and32 payload
FP64 sum/sumsq/count/first-coefficient records. Crop g0 j32 is exported only for
future observations and never changes primary scores or payload votes. Window
contents contain no source clock, arm, attack start, condition label or path.
Ordered aggregation supports reordering/repeat/skip with original first-vote
tie semantics; it neither estimates hidden tau_i nor implements DP. Watermark
state s_n and correspondence tau_i remain future path semantics. Time-dependent
payload is disabled/unverified.

Window export failure preserves usable primary evidence but prevents interface
completion. All original and window denominators remain fixed. Canonical result
commits before the blind projection; candidates batch at each search boundary.
Terminal6 feature/6 score diagnostics and30 posthoc saved references stay separate.
Quality has15 extra reads/15 within-view comparisons, no threshold.

## Validation and limits

The explicit CPU command and package/source receipts are in
`local_validation.json`; `pytest_v1.txt` records15 passed in20.86s, zero skips.
Tests exercise actual lightweight native histories, independent window numeric
checks, the saved-tensor golden fixture, renamed input blindness, combined
view/phase/export failure, both canonical commit boundaries for phase and window
artifacts, normal rejection semantics and static notebook setup. Borrowed CPU
environment files were not modified; local CUDA disabling is test-only.

The external golden receipt is the main session's independent NumPy FP64
calculation on a previously saved LAST tensor. Its successful CPU comparison
is neither a new model/VAE run nor new real-video evidence. The full artifact
paths and hashes are recorded in local_validation.json.

No new model, GPU, VAE, Colab or scientific experiment was executed. No
synchronization, FPR, image/video quality or generalization PASS is claimed.
The notebook remains SOURCE_SHA=None until source publication and verified
binding by the main session. No commit or push is part of this freeze.
