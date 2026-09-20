# Frozen LAST existence, fragment and sequence experiment

Implementation only: no model, GPU, Colab or historical-data rescoring was run here. New orchestration reuses the successful native LAST49 full margin1 writer and original state_clock reader without modifying either. The new manifest fixes nine calibration OFF content/seed pairs (20270101–20270109) and two evaluation content/seed pairs (20270201–20270202), each evaluation with OFF/LAST_A/LAST_B. Exact prompts are in configs/flow_tube_detection.json; none is selected after seeing outcomes.

## One predeclared detector

For each received view, run the entire original blind search and use only local_state as the primary detector. The statistic is its maximum score over clock paths and both messages. Other original modes remain diagnostics; no posthoc choice of best mode is permitted. All four phase observations must be complete and finite for a measured statistic.

For each original source, the sequence statistic uses exactly crop0/crop4/crop5: first take each message's best clock score in each fragment, average its three scores, then maximize over the two messages. Missing any fragment makes sequence unmeasured. This is overlapping evidence within one source, not three independent videos or a search for the best fragment.

Each calibration OFF source contributes the maximum of its seven view scores and sequence score. Freeze tau as the maximum of the nine source maxima; accept only statistic > tau. Equality is rejected. Any missing/nonfinite required calibration view makes the threshold UNCALIBRATED; no reduced-n recalibration or replacement source. Evaluation still records ranking but existence is UNDECIDED. An exclusive threshold.json and its SHA must exist before any evaluation model load. Both generation and media children check that SHA, and the parent checks it again after evaluation.

The rank resolution is 1/(9+1)=0.1, conditional on source-level exchangeability. Nine hand-selected calibration scenes do not guarantee population false-positive control. There are only two evaluation OFF source clusters. Report each attack's OFF decisions and source-any union separately; an observed true acceptance remains a known union acceptance even when another view is missing. Preserve missing/undecided counts. Message uniqueness, correct-message margin, existence acceptance and geometry correspondence are separate fields. A tied message can still have existence accepted; wrong attribution does not turn existence acceptance into a rejection.

## Fixed actual media views

Every source: normal full181 MP4 (H264 CRF18/yuv420p, 8 fps); three real RGB crops of 129 frames at starts0/4/5 saved and reread as uint8.npy without another codec; single zero-based source frame90 deleted (180 frames); 1.25 speed map floor(1.25*i+0.5), i=0..144 (145 frames); unchanged frames with one additional H264 save. Delete, speed and resaved each undergo exactly the same second H264 settings. The latter controls codec effects. No latent crop substitutes for RGB operations. Every phase g=0..3 takes the largest 1+4k prefix of the received RGB; tail_discarded is recorded. In particular deletion's 180 frames are not silently treated as181.

The receiver only receives phase observation tensors and the public book. Actual received→source frame maps and selected windows' nominal RGB support indices join after blind read. Crops preserve the previous complete-observable-window reference. Rounded speed and deletion are not exact affine-clock/VAE inverse maps: their synchronization_success is null, with observable support correspondence instead. Mapping group supports does not identify the VAE receptive field or prove exact frame synchronization. The fixed 11-window score denominator and missing-window penalty are unchanged.

## Fixed budget and quality

15 generated sources: 9 calibration OFF + 6 evaluation sources. 105 views, 420 VAE phase encodes, 1100 Transformer calls, 554 native steps, 15 VAE decodes, 60 MP4 saves (15 base + 45 attacked), 45 crop saves. Successful same-scene evaluation branches share prefix49/v49/native history. Every failed source/view/phase remains in the fixed roster; no hardware eligibility gate, alternate model or automatic retry changes the method.

Evaluation quality differences use the same scene's OFF under the same view: RGB MSE/PSNR and temporal residual are diagnostics, not perceptual quality passes. The fixed blind-review roster contains six evaluation base videos and eighteen attacked full videos, shuffled into anonymous filenames. The public ratings template independently leaves image quality, subject integrity and motion smoothness PENDING. The mapping is stored separately in reporting_only_review_mapping.json and should remain closed until ratings finish. Missing files retain roster rows. No large-model evaluator is added and no human result is fabricated.

New fixed Run all notebook: flow_tube_detection_colab.ipynb. Cell zero is the independent two-line Drive mount. Publication binds an immutable source SHA; no parameter menu. Output is a new FlowTubeDetection run directory. Cal/eval results, input protocol, SHA references, media, observations, blind results, posthoc reports and failures are retained. Current CPU/synthetic tests only establish implementation behavior.
