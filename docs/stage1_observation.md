# Stage 1 implementation protocol (engineering only)

The current user selected **frame-difference weighted centroid, fixed camera,
single subject**. Sample IDs, development/validation split, numerical thresholds,
template/missing model and final burst-versus-sequence semantics remain pending.
No real video has been run by this change. Readiness remains research_defined;
all tests here are CPU array fixtures with science_denominator=0.

## Observer and decoding

`runtime/stage1/observation.py` uses uint8 grayscale frames, weights
`abs(current-previous)` strictly above explicit `pixel_delta`. q is the weighted
pixel centroid divided by `(width-1,height-1)`. Support is the fraction of pixels
with positive weight; mass is the sum of weights. First frame has no q. No motion,
insufficient support and excessive support produce invalid rows with reasons and
null q, never zero-filled success. Excessive support is a global-change/cut
heuristic; it cannot prove a cut. Occlusion with no motion yields no support;
multiple movers, camera motion and identity switches cannot be resolved by this
observer. Subject identity/localization remain unmeasured without labels.

FFprobe checks rotation metadata and rejects nonzero rotation pending an explicit
rotated-input protocol. FFmpeg uses first video stream, stored orientation (`-noautorotate`), no resizing,
grayscale conversion and `fps=sample_hz:start_time=0`. Output times are regular
sample indices/sample_hz, **not original PTS or source correspondences**. Fixed
pixel, frame and per-process time budgets are mandatory. The decoder reads at
most max_frames+1 frames to detect overbudget input and records a blocking failure
instead of silently using a prefix. Raw output memory is bounded by
`(max_frames+1)*max_pixels`; choose these jointly. FFmpeg version affects grayscale
and resampling numerics; this change makes no cross-version reproducibility claim.
Two independent decodes compare times, validity, reasons, q and all row fields.
Trajectory covariance/validity are reported without an unselected scientific gate.

## Public acquisition and freezing

`main/sc_sstw/public_scan.py:scan_public_q` receives only q (invalid positions are
None), explicit public templates/missing sets and two positive budgets. Enumeration
order is template ID, missing tuple, start. It preserves each evaluated window's
status, residual, scatter and covariance determinant. Invalid positions are never
deleted. It uses the existing affine residual as a diagnostic; constants can score
zero and are **not** accepted AISB detections. No residual/2-D validity threshold
is selected. It never invokes calibration, truth, private scoring, or legacy local
top-k. At an evaluation cap it returns incomplete with unknown full count, records
and retained candidates; it refuses a complete frozen object. This is an algorithm
budget limitation, not automatically an environment failure.

For complete enumeration `public_candidates.py` globally ranks by residual,
template ID, start, observed length, template length, missing indices, then applies
the explicit retained budget. Missing indices must be sorted unique in range.
Duplicate identities and nonfinite scores are rejected. Counts before/after and
drop reason are frozen. The container is a burst set only; disjoint half-open
observed intervals supply a basic compatibility predicate, not a complete sequence
set. Full-path coverage and final set semantics remain unselected. There is no
ordinary-video alignment truth or real coverage claim.

Canonical protocol `stage1-json-v1` is UTF-8, sorted keys, compact JSON, finite
Python binary64 shortest-roundtrip numbers, negative score zero normalized to
positive zero, no quantization. SHA-256 covers this candidate object only. Strict
readback reconstructs candidates, ordering and count metadata, and requires byte
equivalence. This is not RFC8785, a provenance chain or proof of scientific truth.
The pure scanner returns frozen bytes before any caller can calibrate/score.

## Running and statuses

Run `python3 -m unittest discover -s tests -p test_stage1_engineering.py -v` for
minimal engineering validation. Existing pytest tests require pytest separately.
The default suite does not launch video, models, GPU or external services.

After selecting a saved video and explicitly fixing parameters, run from the
checkout: `python3 -m experiments.stage1.run_observation --manifest /absolute/input.json
--output /absolute/fresh-directory-outside-git`. No output may overwrite a prior run.
Manifest shape (numbers below are illustrative engineering values, not approved
scientific thresholds; replace path and select every field before a real run):

```json
{"samples":[{"sample_id":"chosen-id","path":"/absolute/chosen-video.mp4","split":"development","content_category":"fixed-camera-single-subject"}],"observer":{"pixel_delta":12,"min_support_fraction":0.0001,"max_support_fraction":0.5},"decode":{"sample_hz":5,"max_frames":300,"max_pixels":2073600,"timeout_seconds":60}}
```

The input manifest is persisted before decoding. Each sample gets one terminal row,
including missing files, dependency/decode/time/budget failures as
OPERATIONAL_BLOCKED. Partial successful repeats survive later failure. No valid
motion observations is NO_GO_THIS_CONSTRUCTION; otherwise OBSERVED_SCIENTIFIC_GATES_UNDETERMINED.
Raw rows and summaries are written together and reconciled to the fixed ID list.
The runner accepts optional `public_acquisition` with exactly public `templates`
(each has `template_id` and `points`), `missing_sets` (lists of deleted template
indices, including `[]` for no deletion), `max_evaluations` and `max_retained`.
When supplied it sends the first complete observation array, including null invalid
positions, through scanning and freeze/readback. Canonical candidate files and their
object-only digest accompany the per-window raw records. Without this configuration
candidate_freeze is NOT_RUN. Evaluation exhaustion retains records and reports
INCOMPLETE_ALGORITHM_BUDGET. Parameters remain pending for any real run. No generated
video, owner/wrong-key loop or scientific promotion.
