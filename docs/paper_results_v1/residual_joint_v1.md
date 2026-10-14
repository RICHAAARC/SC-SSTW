# Saved residual candidate: clean MP4 and delete_g0 joint validation

This fixed user-run handoff tests the four saved `residual_clock_pre.rgb8`
candidates from residual run `20261014T110044944207Z-c569a935`. It does not
reconstruct the writer, change its coefficient, add a source, or scan an
attack. The two fixed report groups are:

- `CLEAN_MP4`: one row for each candidate after one libx264 / 8 fps / CRF18 /
  yuv420p roundtrip;
- `DELETE_G0`: one row for each fixed delete_g0 edit applied to that same
  decoded MP4 RGB. The deletion is not followed by another codec.

Each of the eight rows uses the received RGB, K0, and the public protocol to
run a fresh framewise encode, `CENTERED_U` blind path selection,
`decode_visible_span`, Wan VAE encode, and the unchanged payload reader. The
selector cannot receive the saved attack recipe, payload answer, writer state,
or source truth. The delete map and payload answer join only after the blind
selection and payload read records have been saved. An unresolved clock,
unsupported visible span, unreadable artifact, codec/model failure, or payload
failure remains in the fixed eight-row / 256-bit denominator.

There is no NO_SYNC group. The prior NO_SYNC comparison measured the benefit
of synchronization for old M05 media; it is not required to answer whether the
new saved residual candidate can jointly recover clock and payload through the
fixed clean and delete_g0 paths. Those old M05 results are not imported into
either new-candidate group.

Before codec execution, the notebook also performs a bounded CPU inspection
of the saved candidate and `NATIVE_PRE` RGB8 frames. It rereads the recorded
spatial-peak and temporal-peak neighborhoods plus frames 179/180, reports
per-frame changed-pixel extent, candidate boundary-value counts, and the
fraction of channels newly equal to 0 or 255 relative to `NATIVE_PRE`, and
displays a native/candidate/4x-absolute-difference contact sheet. The latter is
a new boundary-value channel fraction, not an exact clipping measurement. This inspection does
not reconstruct the candidate, add model calls, set a quality threshold, block
execution, or change either four-row group.

## Fixed execution budget

For an uninterrupted complete run:

| operation | fixed budget |
| --- | ---: |
| MP4 encode + readback | 4 + 4 |
| CPU delete_g0 edits | 4 |
| framewise model loads | 1 |
| framewise clip encodes | 8 |
| framewise physical batch8 calls | 176 (92 clean + 84 delete) |
| CENTERED_U selections | 8 |
| Wan model loads | at most 1 |
| Wan clip encodes and payload reads | at most 8 |
| generation / Wan decode / VJP / candidate construction | 0 |

If a clock row is unresolved or unsupported, its Wan call is omitted while the
row remains in the denominator. Interrupted unfinished units may repeat after
Run all resumes the same active Drive task; all actual attempts and failures
remain recorded. Completed units are skipped. `StateJournal` commits each
completed unit and writes a last-committed snapshot at least every 60 seconds.
Wan payload-read attempts have a separate cumulative call ledger and
attempt-numbered vote/read artifacts, so a retry does not overwrite an earlier
returned or failed attempt.
On a fresh-kernel resume, a row with a durable framewise latent but unfinished
clock selection remains a `wan-after-clock` pending environment unit. This
ensures Wan dependencies are prepared before the recovered selector can make
that row payload-readable.

## Run-all handoff

Open `notebooks/paper_results_v1_residual_joint_v1_colab.ipynb`, select a GPU
runtime, and use **Run all**. The first code cell only mounts Drive. The
notebook loads ordinary editable source from
`paper_results_v1_residual_joint_v1_companion.zip`, uses the fixed saved-run
path, and resumes through one active-pointer record in the same Drive task
directory. It does not use B64 source, expected digests, Git state, a source
identity comparison, or an exact dependency-version gate.

The output directory contains `summary.json`, `evaluation_report.json`,
separate `clean_mp4_joint_rows.csv` and `delete_g0_joint_rows.csv`, complete
state/checkpoints, clock score and blind-selection sidecars, payload vote
sidecars, `quality/peak_quality_report.json`,
`quality/peak_quality_contact_sheet.png`, actual-call ledgers, execution logs,
and the handoff summary.

The final report exposes `payload_exact`, `clock_exact`, and `joint_exact` for
every fixed row. Clock exactness means zero posthoc path error against the
fixed single-source clean/delete truth; group `exact_recoveries` and
`joint_exact_recoveries` both count only joint exact rows. No path or payload
truth is joined into an intermediate report or shared state before the Wan
phase reaches its terminal durable state.

The result can support only a fixed four-source observation under these two
adopted media conditions. It introduces no quality threshold, clock-presence
threshold, FPR, generalization statement, or scientific PASS. Notebook build
and CPU/stub checks do not execute the real model, GPU, codec, Colab, or Drive
task.
