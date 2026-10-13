# Protected common direction: user-run delivery

The adopted experiment starts again from saved START_FINAL (CONT_FINAL), reuses
the six saved positive-margin gradients at indices 4, 12, 18, 20, 27 and 30, and
adds four independent VJPs: margin29, the fixed-START logarithmic barrier25
derivative, G directly, and complete RGB MSE versus saved original BASE.

The mask, key, message, readout, BASE radius 1 and single .10 step are unchanged.
The low-dimensional SOCP maximizes the minimum of the six raw margin slopes in
the START tangent plane. It requires margin29/G slopes >=0 and barrier25/Q
slopes <=0. Protection row scaling is positive; six objective rows share one
common scale and are never separately normalized. Solver tolerance is 1e-9;
post-normalization numerical residual and positive-gain resolution are 1e-8
in the documented scaled units. These are numerical resolution parameters,
not scientific criteria. Zero direction remains feasible. Solver failure or
inaccurate status stays unresolved, rather than a method infeasibility claim.

Only a clear positive result creates one FP32 candidate. The saved direction,
planned and actual increment predictions and original-BASE radius are retained.
No scan, fallback direction, second step, endpoint projection, relaxed constraint,
rollback or cap increase is implemented. Actual endpoint protection failures
remain readable results. If there is no candidate, its fixed slot remains NOT_RUN.

The four report views are BASE / START_FINAL / TANGENT / PROTECTED. The first
three reuse saved RGB, q and readouts from Tangent-Common. Report denominators are
352 windows / 5632 chips / 220 metrics. Four gradient probe reads have separate
denominators. Reports preserve all32 margins, six-bit gain retention relative to
TANGENT, margin29 versus .02, other25 changes and log-domain failures, G versus
.01, full frame/temporal errors and actual peak neighborhoods. BASE is the saved
watermarked JOINT BASE. Neither MSE nor optimizer status determines quality PASS.

## Run and restart

Open `notebooks/local_joint_protected_common_v1_colab.ipynb` and Run all on GPU.
Its first code cell only mounts Drive. The source URL/ref is editable; an ordinary
companion ZIP supplies runnable source without Git. There are no B64, expected
digest, source version, manifest or dependency-equality admission checks.

The first new bit29 gradient is committed and read back, then the process exits
normally as `PAUSED_FOR_SESSION_RESTART`. **Delete/disconnect the Colab runtime,
reconnect to a fresh runtime, and Run all again using the same task directory.**
The one-time pause marker prevents another scheduled stop. Runtime identifiers
are observations only; they never block recovery or establish scientific identity.
Same-process continuation is not recorded as real cross-runtime evidence.

Stable output:
`MyDrive/Video-WM/Local-Joint-State-Payload-V1-Protected-Common/resumable-protected-common-v1/`.
Inputs, complete gradients, direction, candidate and RGB commit immediately.
Every60 seconds a Journal snapshot saves last committed progress, with previous
state fallback. Activation spools and live autograd graphs are not persisted.
Only unfinished units can be repeated; all physical attempts and failures remain.
Completed RGB supports report/quality recovery without another model call.
Once RGB is committed, a durable completion record prevents missing or unreadable
core RGB from silently causing a new model decode; the data loss is retained as
an engineering failure. A decode that returned before its RGB could be committed
remains an unfinished unit. Missing or damaged derived q tensors are rebuilt on
CPU from validated raw observations, as are derived report images from saved RGB.
Each of the four new gradients also retains its completed-unit fact. A committed
gradient later lost or unreadable causes an engineering failure before model
loading, never an extra replacement VJP; a VJP returned before atomic unit commit
remains recoverable as unfinished work.

Without interruptions the planned pause makes total physical work **2 model
loads, 5 decodes and 4 VJPs**, including one ordinary candidate decode. The
uninterrupted computational budget is 1 load, 4 gradient decodes/4 VJPs and one
ordinary decode. No positive direction means no candidate decode. Expected
gradient forward/recompute chunks are184 each; ordinary chunks46.

CLI (also works in an extracted source directory without `.git`):

```sh
python -m experiments.wan_state_clock.local_joint_protected_common_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_protected_common_v1.json \
  --output /path/to/stable-task/run
```

CPU tests cover raw max-min semantics, tiny quality-gradient scaling, tangent
geometry, direct scalar derivatives, barrier-domain failure, no-positive/solver
failure retention, planned pause and new model instance resume, full completion
without new model calls, unfinished gradient replay, completed RGB recovery and
no-Git notebook execution with stubbed external calls. These are engineering
tests; no real GPU/Colab/Drive execution was performed for this delivery.

Return the full output directory, including attempts, failures and checkpoints.
The result is one known-message saved-point diagnostic, not codec survival, blind
decoding, FPR, generalization, trajectory/CFG/native causality or scientific PASS.
