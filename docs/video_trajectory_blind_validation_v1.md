# Video Trajectory Blind Validation V1: continuous solver candidate

This candidate replaces only the three-coefficient lattice optimizer. The
historical protocol/configuration and its negative result remain unchanged.
`rgb_dct_continuous_solver.select_coefficients` minimizes the identical weighted
hinge plus ridge objective on a>=0, sum(a)<=remaining, with unavailable
coordinates fixed to zero. Margin=1e-4, ridge=1e-4, positive weight=4,
R*=0.042943312697648145, direction bins, steps T46/47/48/49, fallback scales
1,1/2,1/4 and every real acceptance condition remain unchanged.

The deterministic FP64 projected-gradient solver uses the maximum eigenvalue of
the full quadratic Hessian bound as its Lipschitz constant. Its stopping test is
the Frank-Wolfe gap over the feasible simplex, padded by a documented FP64
roundoff cushion, <= max(1e-18, f(0)*1e-11), within 200000 iterations. This is a
numerical convex global-objective certificate, not interval arithmetic or a
true-model error bound. Invalid input, nonfinite arithmetic, uncertified
stagnation, exhausted iterations, or feasibility/descent failure raises
SolverFailure. The runner persists SOLVER_FAILURE and stops that arm; it cannot
rename failure ZERO_OPTIMUM or commit its baseline and continue a sequential
arm. A zero candidate is returned only when its feasible zero-gradient KKT
condition and numerical certificate hold. No minimum coefficient is imposed.

Stage 1 is a fixed six-point saved-state diagnostic: two existing sources times
T46/47/48, each with spent=0 and an independent copy of its original complete
UniPC scheduler object, z and v. All six rows exist before work, and every
failure/missing row remains. It is NOT a sequential trajectory, MP4 comparison,
blind detection or payload result. The real shared feedback loop recomputes
baseline/probes, forms q/J, uses the new solver, computes original scales and
applies the original real RMS/peak/positive-group/hinge guards. The first
accepted scale is retained; the real tail must produce that acceptance.

Input is the original run directory containing result.json, source_receipt.json
and both source state directories. First verify the exact saved result/source
and all 26 tensor/scheduler file SHA256 values. Restore complete saved objects
on CPU, verify their tensor/history fingerprints and cursors. Never regenerate
a prefix. Recreate only the same model revision and prompt embeddings using
prepare_generation, discard its unused initial noise, and verify its initial
noise fingerprint; the newly created scheduler is not a replacement history.
Rebuild the original basis from saved off_terminal through FP32 VAE and the
same key/carrier/encode pair. Require old full raw_sha256 and masked_sha256;
legacy NumPy repr fingerprints are not accepted as basis byte identity. Save
three newly computed full-byte basis hashes and arrays. A mismatch fails the
source as IDENTITY_MISMATCH, with all three rows retained. Each restored point
must reproduce old state/history/baseline-terminal fingerprints. Record q/J
identity/drift separately; solve the newly measured same-state q/J, never feed
saved offline coefficients into the runner.

The new config names and hashes this protocol, the historical config/protocol,
and the immutable saved result. The run records working source file hashes,
Git HEAD and dirty state, config/protocol hashes, seed, calls, resources and
output paths. No new published SHA or notebook binding is claimed.

Sequential follow-up uses feedback_arm with the continuous solver in the same
shared loop: after acceptance it commits the first-step native history, then
recomputes velocity, baseline, probes and q/J at the next point. The six saved
independent coefficients cannot be concatenated into MULTI. That later real
execution and MP4 OFF/SINGLE49/ONLY49/MULTI/FREE49 comparison is evidence-gated
and is not executed by this stage-1 entrypoint.

No GPU/model run, source publication, later method adoption or complete blind
video-watermark claim follows from local CPU checks or saved-proxy solutions.
