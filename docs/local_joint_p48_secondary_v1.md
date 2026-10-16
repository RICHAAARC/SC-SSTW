# P48 fixed-target secondary continuation

The single authorized CPU validation on the saved P48 anchor and S/P/W/Q gradients
remains **NUMERICALLY_UNRESOLVED**: CLARABEL returned `optimal_inaccurate` in 11
iterations. Its stationarity residual is 2.6186873980371672e-6, also exceeding the
unchanged 1e-6 certificate criterion. No FP32 candidate was formed. No new GPU
execution is useful under this recorded decision.

The implementation eliminates secondary t and minimizes ||u||² subject to
h@u <= -0.3510973330917523, ||u|| <= 1 and ||ac+u|| <= 1. It uses the original
saved secondary_target, original normalization/two-pass span, CLARABEL 1e-10
tolerances and 200-iteration budget. It did not re-solve the primary problem or
change a tolerance, amplitude, solver or target. The new secondary primal/dual
and diagnostics are preserved; original primary primal/dual were never saved.

**Run all is a report-only continuation for this failed fixed CPU attempt.** It
actually reads anchor.pt, all four saved control_cotangent.pt files, the original
solution.json and ZERO/ANCHOR values from the existing Drive run. It rechecks the
single recorded secondary witness, preserves missing C and writes a separate
`secondary-direct-v1/run` directory beside the original `run`. It does not solve
again, load a model, decode, recompute any gradient or overwrite original records.
Repeated Run all reuses the stage's saved CPU decision and only rereads evidence.
The original 3-endpoint and 4-calibration denominators remain present.

Actual new-stage physical cost is **0**. If a future separately qualified single
C were available, a fresh physical stage would require 1 Transformer preparation,
1 VAE load, 2 ordinary DiT forwards, 2 native updates and 1 ordinary decode: 7
top-level calls, 60 original DiT blocks and 46 original decoder chunks, with zero
gradient calls or replay. The two loads are additional restart cost, not unused
old budget. That physical branch is not included/enabled in this unresolved
delivery; no hidden model retry exists.

Use the supplied ordinary editable source ZIP through Run all, or the standalone
CLI `python -m experiments.wan_state_clock.local_joint_p48_secondary_v1_run
--config <config.json> --decision <cpu_solution.json> --output <separate-stage>`.
No Git checkout, hashes, source manifests, exact version or GPU model is required.
File readability, numerical validity, fixed support and actual method quantities
are checked. Missing saved inputs are reported as saved-evidence errors; they do
not trigger regeneration. Return the entire new stage including failures.

This fixed-target formulation preserves the original secondary feasible u set.
Its observed numerical failure does not prove common descent infeasible. The
earlier primary positive local evidence remains; physical C and finite effects
remain unmeasured, scientific_pass remains false. Further method/numerical choices
are not adopted by this delivery.
