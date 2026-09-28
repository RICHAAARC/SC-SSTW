# Final independent review report — continuous solver candidate v3

A2, A3 and A5: engineering PASS on the frozen v3 candidate. A4 synthesis: PASS.
The candidate is ready for the next publication decision. No real model/GPU
execution, early true ACCEPT, sequential media contribution, blind video
detection, payload recovery or robustness is established by these checks.

## Frozen versions and evidence inheritance

| Version | Snapshot | Patch SHA256 | Review disposition |
| --- | --- | --- | --- |
| v1 | candidate_snapshot.json | 60da06ffa5a6858147214cd23f399d76a5fe232427e8c1c4949e4cd957038f26 | Numerical review passed; ordinary-exception accounting P2 required repair. |
| v2 | candidate_snapshot_v2.json | 70afc7b9d6dbb80423492eae6c2a512953e3d9c2332c028b07d85c410a368025 | A2/A3 passed after first repair; A5 reproduced a hard-exit/timeout P2, so milestone approval was still pending. |
| v3 | candidate_snapshot_v3.json | 027986e51e0f5113aa0610ab0d1dbebc35694cdfb7f6d7f0eedabea819f0a998 | A2/A3/A5 final engineering PASS after second repair. |

Reviewers verified the 11-file snapshots. Each repair changed only the
candidate runner and test file. Solver, shared real loop, configuration,
protocol and original method acceptance conditions remain byte-identical.
This report is outside the frozen 11 files; original snapshots, patches,
numerical replay and implementation report remain preserved.

The six saved surrogate solutions remain v1 numerical evidence, inherited
because the solver and mathematical inputs are unchanged. They were not
rerun for v2 or v3. A2 explicitly distinguished source-tree digest prefixes
v1 `5b3297da…` and v2 `e05978e2…`; these are different working-source receipts,
not interchangeable run identities. Likewise, the 26-artifact restoration
review is inherited with its stated CPU environment and unchanged loader.

## A2 — numerical solver, constraints and budget

A2 independently ran v1's 18 targeted tests (9.97 s). An independent
one-dimensional brentq root was 0.005116279069767451, with coefficient error
3.23e-14; the three-identical-column analytic case agreed. Independent scaled
SLSQP checks on all six saved proxies differed in objective by at most
3.3881e-21. A 70-digit Decimal recomputation from exact binary inputs found
all six convex gap values at or below the reported numerical bounds.

The actual solver with max_iterations=0, injected into the shared real loop,
produced SOLVER_FAILURE: coefficients=None, zero candidates, spent=0, no
terminal and no sequential advancement. Source state/history stayed unchanged.
The original real acceptance conditions were not relaxed.

A2 verified the restricted v2 delta and ran three focused tests (2.58 s).
On v3 it independently passed five focused recovery/failure tests (3.36 s)
and confirmed the numerical solver/constraints/budget/acceptance semantics
were unchanged. Its final decision is engineering PASS with the explicit
numerical-evidence inheritance above.

## A3 — saved artifacts, native history and first P2 repair

A3 read all 26 original Drive .pt files through Windows OpenRead, streamed the
bytes to CPU BytesIO, and called the frozen restore_backend with only the file
reading boundary replaced. SHA checks, torch.load, fingerprints, tensor shape,
scheduler cursor and finite-history checks followed the original loader logic.
All 26 restores passed; all six early z/history/v identities matched.
Scheduler last_sample was retained; model_outputs deep copies shared no storage.
Tampered z46 hash checks were rejected for both sources. A4 independently
verified the 26 real artifact hashes and the source/patch relationship.

This was read-only CPU loading of real saved artifacts. It loaded no model,
used no GPU and did not write the original data. Local torch 2.5.1+cpu and
diffusers 0.39.0 do not establish model/VAE equivalence in Colab diffusers 0.40.0.

A3's v1 P2: after the shared loop recorded ACCEPT, a committed-history guard
could raise an ordinary Exception while the six-point catch retained ACCEPT.
V2 converts such a subsequently failed normal verdict to ENGINEERING_FAILURE,
retains prior_outcome, candidate decisions and measured spent, and marks its
budget FAILED. SOLVER_FAILURE and IDENTITY_MISMATCH keep their classifications.
A3 replayed the original counterexample and passed the focused regression
(1 test, 2.36 s): expected=6, accepted=0, evaluated=0, failed=1, pending=5.
Prior ACCEPT and measured spent remained visible. This first P2 is CLOSED.

## A5 — hard-exit/timeout P2 and final milestone audit

A5 reproduced the second P2 on v2: a hard exit after ACCEPT/budget RUNNING was
persisted bypassed the ordinary Exception handler. Parent recovery changed
only pending rows, leaving an incomplete ACCEPT counted as successful.

V3's parent worker-failure recovery treats budget COMPLETE as the persisted
point completion marker. It converts a normal verdict with an incomplete
budget to ENGINEERING_FAILURE, preserves prior verdict/candidates/measurements,
marks that budget FAILED, and retains every unrun row. Earlier COMPLETE points
remain unchanged when a later independent point fails.

The hard-exit regression at T46 now reports expected=6, accepted=0,
evaluated=0, failed=3, pending=3. A T47 timeout after T46 completed reports
expected=6, accepted=1, evaluated=1, failed=2, pending=3, with the COMPLETE T46
record unchanged. The retained successful count belongs only to completed work.

A3 independently passed both hard-exit recovery tests on v3 (2 tests, 2.98 s).
A5 passed hard-exit recovery, preservation of the COMPLETE early point and the
ordinary-exception regression (3 tests, 3.45 s), and independently replayed its
original counterexample successfully. The second P2 is CLOSED. A5's final
milestone decision is engineering PASS; A4 synthesis agrees.

## Validation and next decision

A1's complete v2 targeted suite passed 19 tests in 10.29 s. V3 received focused
failure-recovery validation, including A1's five tests in 3.46 s and the
independent reviewer runs above; the numerical oracle was not rerun or relabeled.
At local save, 11/11 frozen hashes and the ordinary git diff --check pass.
The additional staged source/JSON/report check uses command-local
core.whitespace=cr-at-eol because frozen files include CRLF/CR line endings.
The archived .patch evidence contains literal diff context spaces and is
excluded from that source-whitespace check; none of those frozen bytes is
rewritten to silence formatting diagnostics.

No real six-point GPU run has occurred. The new notebook remains an explicit
UNPUBLISHED_DRAFT with SOURCE_SHA=None and no executable published Colab link.
The next decision is to publish this dedicated development-branch candidate,
verify its actual published immutable source SHA, bind that SHA with the
notebook builder, then give the user a Colab link for the fixed six-point run.
Local engineering PASS is not authorization to publish or execute that run.
