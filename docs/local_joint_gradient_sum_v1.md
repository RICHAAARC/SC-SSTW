# Local Joint Gradient Sum V1

This fixed family reuses the two saved masked BASE gradients from the completed
Base-Alignment task. `g_A` is the derivative of the fixed historical 16-bit
nonpositive-roster mean margin. `g_G` is the derivative of
`correlation[0]-correlation[16]`. Their raw scalar weights are exactly 1:1:

`d = (g_A + g_G) / ||g_A + g_G||`, then `SUM_STEP = BASE + 0.10*d`.

Neither input gradient is normalized separately. The summed direction is formed
once, normalized once and stored once. The original BASE-relative radius cap 1
and adopted support mask remain. There is no endpoint projection, scan, second
step, rollback, alternate direction or replacement VJP. A zero or nonfinite sum
is an explicit engineering failure and remains in the failure ledger.

The BASE q/readout and clamped float RGB are reused. Only SUM_STEP receives one
ordinary VAE decode (46 chunks). Its ordinary RGB is committed before readout.
Both views report all 32 bit margins, all 22 state correlations, actual
max-wrong offset and G, fixed G16, fixed-roster mean A, positive/nonpositive
counts, and descriptive distances from mu=.02 and gamma=.01. These constants do
not gate completion. Full, per-frame and temporal clamped-float RGB differences,
actual peak pixels and their neighborhoods are saved without a quality gate.

Nominal uninterrupted new GPU work is one VAE load and one ordinary decode.
VJP, gradient decode, encode, DiT, native/scheduler and codec work are zero.
Candidate latent, BASE reuse, candidate RGB and scoring are durable units.
Completed work is skipped on Run all; only an unfinished decode may be repeated,
while physical calls, chunks, attempts and failures accumulate. Every 60 seconds
the Journal records the last durable state.

Completion is engineering evidence for this one saved source point and fixed
finite step: one source point and one new endpoint, with zero independent source
videos and zero independent samples. It does not establish blind recovery, media survival, FPR,
independent-sample behavior, generalization or scientific PASS. GPU, Colab and
Drive execution remain user-run.
