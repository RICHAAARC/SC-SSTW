# Stage 1 implementation protocol (engineering only)

The current user selected **frame-difference weighted centroid, fixed camera,
single subject**. The user confirmed the complete S0 plan: one historical Drive
development sample, numerical engineering parameters, public templates, no deletion
and bounded burst-set semantics are frozen before CPU execution. Scientific AISB
validity and alignment coverage remain undecided. Readiness remains research_defined;
unit tests have science_denominator=0 and the real run is development engineering only.

## S0 user-confirmed freeze before execution

The working branch is `dev/real-video-public-observation`. Historical branch mapping: [branch_mapping.md](branch_mapping.md).
Current branch names use English kebab-case under `dev/`, without a stage number.
The existing worktree directory stays `worktrees/Stage1-Public-Observation`.

`experiments/stage1/s0_frozen.json` records the user-confirmed S0 freeze. The
following choices are approved for this development engineering run, not
scientifically validated thresholds:

| Item | Frozen value and purpose |
| --- | --- |
| Input denominator | N=1 historical UCF101 video, fixed development role and sample ID |
| Sampling | 5 Hz; at most 300 output samples; no silent first-60-second prefix |
| Decode budget | At most 2,073,600 pixels/frame; 60 seconds per ffprobe or ffmpeg process |
| Pixel support | Absolute gray difference strictly above 12; support fraction in [0.0001, 0.5] |
| Templates | Existing public six-point burst_alpha, burst_beta, burst_gamma, coordinates embedded in the frozen protocol |
| Missing model | No deletion only, `missing_sets=[[]]`; no claim of deletion robustness |
| Enumeration | 885 evaluations = 3 × (300 − 6 + 1); invalid windows still consume evaluations |
| Retention | Global first 128 by the documented total ordering; engineering bound, not a validation-tuned choice |
| Set meaning | Burst set plus disjoint half-open observed-interval compatibility; complete sequence set deferred |
| Stability acceptance | On the same machine with identical tools, two independent decodes must have identical structure and all observation fields exactly equal; no cross-environment claim |

These budgets bound the first entry exercise without inheriting a synthetic
residual threshold. At most 300 frames permits 885 window/template evaluations;
actual evaluated count is `3*max(0,n-5)`, before invalid-window exclusions and global
retention. With 300 samples the required initially invalid frame excludes the
first window for each template, so 885 is an enumeration bound, not a promise of
885 scored candidates. A 128-candidate retained set may drop genuine alignments;
without independent positive labels, pre/post-truncation coverage is unassessed.

The user authorized historical Google Drive Datasets as the sample source. Before
running the observer, the lexicographically first JumpingJack member was selected
from `Datasets/UCF101/raw/UCF-101.zip` (Drive file ID
`1UgqetmsGMSttIRt8sbD45GICFF1zX1Nk`). The original AVI is locally available without
transcoding or generation:
`/home/richar/projects/Video-WM/datasets/UCF101/JumpingJack/v_JumpingJack_g01_c01.avi`.
The frozen sample ID is `UCF101-JumpingJack-v_JumpingJack_g01_c01`, role
`development`, N=1. Its neighboring `sample_source.json` records the source.
Preliminary ffprobe inspection reports 320×240, 93 original frames,
30000/1001 fps and 3.1031 seconds. Nine-frame visual inspection suggests one indoor
person performing jumping jacks, an approximately fixed background and no obvious
cut; this is not subject annotation or proof of the content assumption.

The protocol's `manifest` is copied to a standalone Git-external runner input before
execution, after the protocol is committed. The former draft guard is removed
under the user's complete S0 confirmation. The selected sample is not an AISB
positive. A single
development sample does not create an independent validation
split; a validation sample must not be used for tuning. An ordinary saved video
has no AISB alignment truth. Independent positive labels remain a separate
scientific prerequisite and are never inputs to public acquisition.

The one-row sample list and chosen parameters are frozen in the final manifest
before execution; preserve every failure and never replace the sample, adjust
parameters after seeing validation outcomes, or sweep to rescue the result.
The frozen output is
`/home/richar/projects/Video-WM/diagnostics/真实视频公共观测/帧差加权质心-固定镜头单主体/run01`,
outside Git and required to be fresh. Exceeding pixel/frame/time or encountering
file/decode/dependency failure stops that sample as OPERATIONAL_BLOCKED and retains
its partial records. All-invalid observations stop this construction as
NO_GO_THIS_CONSTRUCTION. Enumeration exhaustion retains evidence as an algorithm
budget limitation and does not freeze a complete set. Exact repeat disagreement
fails the frozen stability engineering acceptance; the current runner reports
the comparison and does not convert it into a scientific terminal PASS/FAIL.
Scientific AISB validity, 2-D support/degeneracy, localization and true alignment
coverage thresholds remain unset; no S0 text makes them passed. Authorized local
CPU execution does not increase scientific readiness.

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
Manifest shape (numbers below are approved engineering values, not scientific
thresholds; use the frozen complete manifest for the authorized run):

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
INCOMPLETE_ALGORITHM_BUDGET. This S0 freezes parameters for its one development run. No generated
video, owner/wrong-key loop or scientific promotion.
