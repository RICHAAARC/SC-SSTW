# Saved-media coefficient transmission diagnosis

Fixed input: grow_temporal_difference_media_20260918T171544536253Z. All12 videos/48 layers and128 marked bits are retained. The diagnostic pins the actual result.json SHA, verifies each of48 saved latent SHA256 values, rechecks all92-vote aggregate/sign/erasure records, and exports all23x4x16 continuous coefficients. Configs, deterministic codebooks, original source commits, script/method hashes and source tensor hashes are included. OFF signed projections are explicitly payload-A references, not truth errors or calibrated FPR.

Real offline CPU execution verified48/48 tensors. Hard votes and erasures agree exactly. Recomputed continuous means differ by at most2.0817e-17 across CPU reduction environments; comparison uses1e-12 absolute tolerance only for these continuous diagnostics. This tolerance does not change decoding, ties or thresholds.

| Layer | Hard exact/8 | Hard errors/128 (erasures) | Soft exact diagnostic/8 | Soft errors |
|---|---:|---:|---:|---:|
| terminal |8|0 (0)|8|0|
| float RGB |8|0 (0)|7|1|
| RGB8 |8|0 (0)|7|1|
| MP4 |4|6 (3)|3|10|

Mean signed continuous coefficient: .0656793 -> .0430511 -> .0425329 -> .0288768. Ratios of these means are .65548/.98796/.67893. These describe successive measured chains (VAE reconstruction, quantization, MP4 roundtrip plus subsequent VAE observation), not isolated causal codec/VAE effects, VAE bugs or quality judgments. Paired PSNR is not a quality gate.

All indices below are zero-based. Vote margins are correct-sign counts minus wrong-sign counts, with92 votes per bit.

| Case / arm / bit | terminal | float | RGB8 | MP4 | RGB8->MP4 correct-to-wrong / wrong-to-correct |
|---|---:|---:|---:|---:|---:|
| p0s0 B1 |30|20|16|-2|12 /3|
| p0s0 B15 |36|28|22|-4|17 /4|
| p0s1 A3 |36|22|20|-4|16 /4|
| p0s1 A9 |40|12|16|0|17 /9|
| p1s0 A4 |78|56|50|0|30 /5|
| p1s1 A6 |48|32|32|0|18 /2|

These six bits all have positive MP4 soft means, but using soft votes for them selectively would use truth. Across all128 bits, soft decoding introduces10 different errors and reduces whole-message recovery from4/8 to3/8. All MP4 soft-negative examples (case,arm,bit,mean) are:

- p0s0 A12: -.00438888
- p1s0 A2: -.02263932; A7: -.00042675; A9: -.01348426
- p1s0 B6: -.01289094; B10: -.00387446; B13: -.01974189
- p1s1 A3: -.00702111
- p1s1 B6: -.01209335; B13: -.01144410

All ten are hard-correct. No hard/soft hybrid is selected or implemented.

New tensor-level evidence: RGB8->MP4 has251,253,370,367,411,404,286,279 coefficient sign flips out of1472 for the eight marked videos in fixed case/A/B order. Zero-intercept transmission gains are approximately.898,.875,.800,.816,.986,.998,1.004,.994; residual RMS is.0759,.0778,.1179,.1144,.0660,.0645,.0578,.0602. Gain is a descriptive fit, not an independent calibration or deployable correction. Near-unit gain in p1 coexists with hundreds of sign changes: a single amplitude correction cannot explain these failures. Full JSON retains each pair and frequency/repetition flip pattern and bit transitions.

Method decision: keep the fixed hard receiver. The evidence supports pursuing robustness of individual signed carrier margins to the compression/reencode channel, rather than replacing hard votes by raw soft means or adding selected-bit repairs. A concrete next candidate to review is a fixed one-sided writer margin loss .5*sum(max(0,.5-b*d)^2), with the same carrier/window/eta and receiver. This changes only the writer objective, does not impose a new detection threshold, and does not penalize already-correct coefficients above the existing.5 target. Its compression benefit is unproven; this audit does not implement or run it. If approved, freeze it before running the same four-case OFF/A/B fullchain roster with actual control energy and all128-bit outcomes; no scans or post-hoc pair/frequency selection. The current evidence does not establish which future writer change will succeed.

Delivery: scripts/build_grow_media_offline_notebook.py creates a fixed-input direct Run-all CPU notebook. It checks out published source, uses existing torch only, reads the original48 stored latents, and writes GROWMediaOffline/grow_media_offline_<UTC> with diagnosis/source archive/log. No GPU, Transformer, VAE, MP4 reencode, regeneration, tuning or source overwrite. Human visual quality inspection, existence/FPR calibration, ablations and generalization remain outside this diagnostic.
