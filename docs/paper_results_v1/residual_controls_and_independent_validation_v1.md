# Residual joint controls and independent-validation proposal

## Adopted same-run follow-up

The fixed follow-up uses the four completed cases `confirm_01`, `confirm_02`,
`confirm_07`, and `confirm_03` under `CLEAN_MP4` and the existing fixed
`DELETE_G0` edit.  Its denominator is 24 rows / 768 compared bits:

- 8 completed watermarked K0 rows, reused without new model calls;
- 8 watermarked K1 wrong-key rows; and
- 8 unwatermarked K0 rows from saved `OFF_NATIVE/POST` MP4 readbacks.

K1 recomputes key-dependent clock evidence and blind `CENTERED_U` selection
from the saved framewise latent.  A K0 Wan latent may be reused only when the
received media is shared and the complete selected operation maps are equal.
OFF is an absence/false-match control and has no inserted payload truth.  No
presence threshold, FPR interpretation, quality gate, or `OFF/K1` arm is part
of this denominator.

A targeted audit of the saved checkpoint and recovery records found completed
framewise calls for `OFF_NATIVE/full` and `OFF_NATIVE/delete_g0`, but no saved
raw key-independent `[T,4,40,64]` POST latent.  The available score sidecars
are already key-dependent outputs; PRE, other-attack, and Wan temporal
16-channel tensors have different inputs or semantics.  Consequently the
current uninterrupted budget still plans eight new OFF framewise clips / 176
physical batch-8 calls.  The runner durably reuses those new tensors after
they are produced, so a resumed kernel does not repeat completed clips.

The post-MP4 CPU quality plan has 24 pairs: eight received observations against
`CANDIDATE_PRE`, saved `NATIVE_PRE`, and saved `NATIVE_POST`.  The delete map is
applied to each reference for the delete rows.  Each pair retains full-clip,
per-frame, temporal residual-delta, and four independent peak measurements:
frame RMSE, frame maximum pixel/channel absolute error, transition RMSE, and
transition maximum pixel/channel residual-delta.  Every peak retains its local
neighborhood and actual candidate/reference/absolute-difference frame and crop
triptych PNGs.  Difference panels use the fixed RGB8 range 0..255 without
auto-normalization or display-driven selection.

Initialization writes all 24 receiver rows and all 24 quality slots before it
reads individual source rows or artifact metadata.  Missing source rows,
missing OFF POST metadata, and unreadable source state files therefore become
retained `MISSING`/`FAILED` slots with reasons.  They do not erase the fixed
denominator.  All in-flight unit attempts and call ledgers are closed as
`FAILED` or `INTERRUPTED`; a completed atomic tensor cache is reusable after a
kernel restart even if the prior process ended before closing its attempt.

Known probe bits and post-hoc clock truth are joined into reports only after
the clock and Wan phases reach `COMPLETE` and no model unit remains pending.
Before then the report says `NOT_JOINED` and
keeps comparison fields null.  Final K0 rows report exact payload recovery.
K1 and OFF rows report false-target exact matches.  Post-hoc clock-path equality
is a separate descriptive component and is never a presence decision, FPR
estimate, or gate.  The report appendix mirrors all 24 CPU quality-pair slots.

Every Run-all attempt downloads the ordinary companion ZIP, extracts it into a
new temporary source directory after safe-path and required-member checks, and
probes the CLI in a child process before use.  Existing or partially extracted
source is not accepted as the current CLI.  Unreadable stage-receipt or attempt
history is moved to a timestamped `.unreadable-*` file and a fresh list records
that recovery without blocking the scientific rows.  No digest, version, or
repository-state gate is added.

## Independent new-sample proposal — `PROPOSED_NOT_ADOPTED`

After the same-run controls are complete and audited, use a minimum independent
cohort of four new cases.  Prompts, seeds, and case IDs remain unset; this file
does not freeze a roster or authorize execution.  Once adopted, freeze those
four cases before generation and apply the unchanged writer, codec, K0/K1
receiver, `CLEAN_MP4`, and `DELETE_G0` semantics.

For four cases, the proposed receiver denominator is again 24 rows / 768 bits:
eight WM/K0, eight WM/K1, and eight OFF/K0 rows.  All failed, missing, and
unsupported rows remain in that denominator.  OFF remains an absence control,
and the comparison to the fixed probe bits does not define presence or FPR.

The proposed quality appendix also has 24 fixed pairs: eight actual WM received
observations against `CANDIDATE_PRE`, `NATIVE_PRE`, and `NATIVE_POST`.
`DELETE_G0` applies the unchanged fixed map to each reference before
comparison.  Every pair retains the same four independently selected peaks,
their neighborhoods, and fixed-range 0..255 frame/crop triptychs described
above.  These values have no quality threshold or gate.

The following is a planning upper bound, contingent on reusing the already
implemented writer stages and shared model loads.  It is not a measured budget
and must be revised after the same-run controls are audited:

- up to 8 paired OFF/WM generation trajectories and 8 codec
  publications/readbacks, only if the original writer stages can be reused
  unchanged; the exact generation-stage call plan is not frozen here;
- 8 CPU delete edits;
- 24 CPU post-MP4 quality pairs with the three fixed references and four-peak
  inspection artifacts;
- 16 framewise clips, 352 physical batches at batch size 8;
- 24 blind `CENTERED_U` selections;
- at most 24 Wan clip encodes; exact K1/K0 equality of the complete corrected-
  RGB operation may reduce this count by reusing the same actual encoder input;
  unsupported or failed receiver rows may also reduce actual calls while the
  rows remain in the denominator; and
- 24 payload reads.

Model loads are shared within each uninterrupted phase where the existing
runtime permits it.  No claim is made here about generation-stage peak memory,
elapsed time, or exact writer-call count beyond this provisional upper-bound
accounting, because the new cohort has not been run.

This proposal must be revised to the adopted cohort size by linear scaling.
No new sample is generated, selected, or reserved by the current follow-up.
