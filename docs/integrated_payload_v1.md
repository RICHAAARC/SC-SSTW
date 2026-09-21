# SC-SSTW Core Integration V1

## Dependency selection

The candidate starts from formal main `3f0a5fafa7c2aa56fbc69bed17649accf49ae152`.
It selectively re-expresses the successful live-history mechanics recorded by
Uniform-Tanh source `3dd3d0ec7c69dd67fcd6e7acf94ae0f3434ba084`:

- fresh `generation.prepare_generation` without a VAE in the transformer worker;
- FP32 CFG velocity, native scheduler steps, copied complete histories, unit
  same-history probes, and detached CPU clean-leaf gradients;
- temperature one, `R*=0.042943312697648145`, SINGLE46 full budget,
  MULTI44/46 half budgets, and free steps 47--49.

The historical `flow_tube_uniform_tanh_run.py` dependency chain, failed
experiments, `velocity_coefficients`, old oracle solvers, notebooks, and raw
evidence are not copied. Exact historical saved-state reproduction remains at
branch `dev/flow-tube-uniform-tanh`, source `3dd3d0e...`, fixed notebook tree
`e41afefa723d6b60d08861e03d8a6fcf4e2727c3`, and run
`flow_tube_uniform_tanh_20260921T085713169978Z`.

## Payload and writer

One video carries one arbitrary four-bit value `0..15`. RM(1,3) produces eight
coded data-window bits with `(n,k,d)=(8,4,4)`. Windows 0, 5, and 10 are keyed
common pilots. The writer minimizes

`(mean-rival data tanh loss + 0.25 * common-pilot tanh loss) / 1.25`.

The pilot term is explicit because common positions cancel from a pure
target-minus-rival loss. The native-response budget is unchanged; the 0.25
coefficient allocates direction within that budget and is not added strength.
The hard decoder accepts only a unique word satisfying `2e+s<4`, where `e` is
the number of wrong visible data windows and `s` the erased data windows.

## Blind receiver and decisions

Each payload defines eleven keyed phase states. Data-window phases are negated
by the RM bits; pilot phases are common. The receiver computes public projection
observations, runs the existing fixed-gain directional observer over every
bounded clock path, and ranks all 16 payloads. This is a new 16-code state
receiver. The older A/B state-clock remains unchanged and its exact behavior is
covered by the original regression tests.

The receiver signature is `read(observations, book)`. It receives no prompt,
truth, original state, writer evidence, or known attack. Ranking is always
reported separately. Existence decisions are `DETECTED`, `REJECTED`,
`AMBIGUOUS`, `INVALID`, or `UNCALIBRATED`; every state except `DETECTED` returns
no payload.

Calibration runs both independent OFF sources through the complete fixed search
family before any evaluation process starts. The frozen threshold is the larger
source maximum plus `1e-6`. A source maximum includes message, clock path, all
seven saved views, and the three-crop sequence aggregate. Missing or failed
members invalidate that source and leave the run uncalibrated. Raw rankings from
partial phase sets are retained for diagnosis, but only a `SCORED` view with all
four phase rows `COMPLETE` is eligible for aggregation, calibration, or a
decision. If calibration is unavailable, the fixed evaluation cases still run
to preserve payload/attack denominators; every existence decision is
`UNCALIBRATED` and returns no payload.

After freezing, the two calibration OFF sources receive the same per-view,
crop-aggregate, and source decision fields as evaluation arms. They are marked
`THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR`; their rejection is a mechanical
consequence of the max-plus-guard rule, not held-out negative evidence.

## Fixed GPU candidate

The four fresh prompt/seed cases create eight arms: two calibration OFF arms and
OFF/SINGLE46/MULTI44_46 for each of two evaluation sources. Each arm saves:

1. FULL;
2. CROP0_129;
3. CROP4_129;
4. CROP8_129;
5. DELETE90;
6. SPEED5_4;
7. REENCODE, produced from the decoded FULL MP4 and saved again.

Every saved MP4 is read back and independently VAE-encoded at phases 0, 1, 2,
and 3. The predeclared denominator is 4 fresh cases, 8 arms, 56 views, and 224
receiver encodes. The three crop files are real received assets; aggregation
does not slice an already encoded full latent.

The planned model-side counts are 4 generation preparations, 448 transformer
forwards, 224 main native scheduler steps, 6 zero shadows, 6 unit probes, and 6
CPU clean-leaf backwards. Media counts are 8 VAE decodes, 56 MP4 saves, 64 MP4
reads, and 224 VAE encodes. Each counter records attempted and completed calls.
The parent process tees child output to the live notebook and stage log. Its
progress file keeps the cumulative call snapshot and current case/stage/arm/view;
child launch failures and both stage exit codes remain attached to the fixed
case rows.

## Evidence ceiling

The fixed candidate has not been run on GPU, Wan, Colab, or Drive in this
delivery. The earlier two-content/two-message 4/4 development result supports
only its exact historical method. The new payload, pilot, state receiver,
calibration, attacks, and new roster require the user-run notebook. No baseline,
large-FPR, quality-blind-review, generalization, or paper claim is part of this
candidate.
