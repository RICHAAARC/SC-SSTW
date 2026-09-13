# Independent position-to-observation relation development protocol

This evaluator tests the development hypothesis `q_t = A m_t + b`, where
`m_t=(p_(t-1)+p_t)/2`. p is a manually annotated subject position on the exact
decoded 5 Hz frame; q is the existing frame-difference centroid. A and b are one
constant affine map per video. This is not a carrier, AISB positive or blind
detector. Annotations enter only `experiments/stage1/evaluate_relation.py`, which
does not import or invoke the observer. Main method code and the frozen observer
parameters are unchanged. For this authorized development diagnostic the operational
position definition is `pelvis_center`, the image projection of the midpoint of
the left and right hip, visually estimated by the annotator. This is the simplest
operational choice made within the authorized scope, not a separately user-confirmed
anatomical specification or precision ground truth. Each point retains its own
subjective uncertainty radius.

The first development intake retains three unavailable content slots and one
in-place JumpingJack slot. That existing sample's observer result had already been
seen in the earlier task, so its annotations are not claimed to be historically
q-blind. The current annotation process uses only the exported actual sampled
frames, without looking at q or revising points against q. It does not rerun the
observer. New candidate screening may not use new q results to choose samples.

## Fixed sample and time denominators

Four ordered development slots are retained: `horizontal`,
`vertical_or_diagonal`, `two_dimensional_turn`, `in_place_limb_motion`. Unavailable
content remains an explicit MISSING slot, never a smaller denominator or a new
slot. Content is selected and annotations frozen without inspecting q. A numerical
rank-two result does not turn approximately straight content into a turning sample.

Index t is the actual sampled output index, not a source correspondence. t=0 is
initial. All even t>=1 are fit and all odd t>=1 are heldout, fixed before observing
validity or annotations. Missing p_t or p_(t-1), invalid q, and unannotated rows
remain in their original positions with reasons. An eligible fit pair needs both
positions and valid q. Do not shift, interpolate, search an offset, reindex or
move heldout rows into fit. Report fixed and eligible counts separately. Since
neighboring midpoints share p and video frames are correlated, heldout is an
interleaved within-video prediction diagnostic, not independent validation.
Report `q_valid_fraction` over every sampled row including t=0, and the longest
consecutive invalid-row count on that same unfiltered sequence. Annotation gaps
do not alter the q-validity denominator; fit eligibility is reported separately.

## Fit, baseline, normalization and uncertainty

Fit centered ordinary least squares without ridge to the eligible even-index
pairs only. Baseline is the q mean on exactly those same fit pairs, held constant
on heldout. No heldout q affects the affine map, baseline or normalization.
Require at least three eligible pairs and rank-two centered m covariance;
otherwise retain INPUT_RANK_INSUFFICIENT_FOR_2D_AFFINE, no model or predicted error.
Rank uses eigenvalue `> max(1e-15, lambda_max*1e-10)` solely as a floating-point
linear algebra criterion. It is not a scientific two-dimensional motion gate.
There is no regularization fallback to manufacture an identifiable fit.

Per-point error is Euclidean q prediction error. Heldout RMSE is
`sqrt(mean(||q-pred||^2))`; baseline RMSE uses the same eligible heldout rows.
Normalize by fit m's RMS radial scale `sqrt(mean(||m-mean_fit_m||^2))`.
Zero scale produces null normalized errors. Relative improvement is
`1 - RMSE/baseline_RMSE`, null for zero baseline or absent model. A negative value
is retained. These errors are diagnostics in normalized frame coordinates, not
an invariant cross-video unit or a scientific PASS threshold.

Each manual p records original pixel coordinates, image width/height and a
subjective uncertainty radius in pixels. p is exactly
`(pixel_x/(width-1),pixel_y/(height-1))`; mismatches are rejected. Convert the
uncertainty to a conservative normalized radial bound with
`radius_pixels*max(1/(width-1),1/(height-1))`. A midpoint's bound is half the sum
of the two position bounds; the fit uncertainty summary is their RMS.
These are subjective bounds, not statistical confidence intervals.

Report p, m and q covariance eigenvalues, lambda_min/lambda_max, minor-axis SD and
mechanical rank. Compare fit m minor SD and normalization scale with the fit
annotation uncertainty bound. Annotation noise may create nonzero minor variance
or numerical full rank in straight content. `dimensional_readability` stays
UNDETERMINED: mechanical rank does not prove observable two-dimensional motion.
Likewise in-place limb motion may produce q drift while subject center is static.
Report q valid count/drift separately; never label these rows falsely valid using
an unselected scientific threshold. Nearly static motion makes normalization
unstable relative to annotation uncertainty even when its exact scale is nonzero.

## Input schemas and commands

An annotation file contains the chosen position definition and one row per actual
sampled frame. Missing positions use null p/p_pixel/uncertainty and a reason.
Missing entire annotation rows are retained by the evaluator as missing; duplicate
or out-of-range indices and timestamps differing by more than 1e-9 seconds are
rejected. Example schema (illustrative definition, not an approved choice):

```json
{"sample_id":"chosen-id","position_definition":"USER_SELECTED_DEFINITION",
 "coordinate_system":"normalized_xy_width_minus_1_height_minus_1",
 "frame_width":321,"frame_height":241,
 "rows":[{"sample_index":0,"time_seconds":0.0,"p":[0.5,0.5],
          "p_pixel":[160,120],"uncertainty_pixels":3,"reason":"VISIBLE"}]}
```

The evaluator manifest has `position_definition` and an ordered `slots` list.
Each slot is either:

```json
{"slot":"horizontal","status":"READY","sample_id":"chosen-id",
 "observation_results_path":"/absolute/prior-observer-run/results.json",
 "annotations_path":"/absolute/frozen-annotations.json"}
```

or `{"slot":"horizontal","status":"MISSING","sample_id":null,"reason":"No qualifying content"}`.
Always provide all four slots in the listed order, each exactly once. The evaluator
uses the first independently decoded observation repeat and the exact sample ID.
It retains per-time p/m/q, validity, role, predictions, baseline and errors. Output
is a fresh Git-external directory with input_manifest.json and results.json.

Run `python3 -m experiments.stage1.evaluate_relation --manifest /absolute/eval.json
--output /absolute/fresh-evaluation-directory` only after samples, position definition,
annotations and observation inputs are frozen. Do not send annotations to the
observation runner or use evaluation results to resample/tune the frozen observer.

For q-blind annotation export, `python3 -m experiments.stage1.extract_annotation_frames
--video /absolute/video.avi --output /absolute/fresh-frames-directory
--sample-id chosen-id --position-definition USER_SELECTED_DEFINITION
--decode-config /absolute/frozen-decode-object.json` calls decode_video only.
It exports actual sampled grayscale PNGs with unchanged indices and timestamps,
plus annotations_template.json. It never calls observe_frames or generates video.
Fill positions/uncertainty manually, retain unavailable rows and freeze the
annotation file before the observer run. PNG encoding is stdlib only.

Unit fixtures run with `python3 -m unittest discover -s tests -p test_relation_development.py -v`.
They test a known affine heldout relation, collinearity, baseline isolation from
heldout q, missing annotations, zero-scale motion and fixed missing slots. These
fixtures do not establish real-video relation validity.
