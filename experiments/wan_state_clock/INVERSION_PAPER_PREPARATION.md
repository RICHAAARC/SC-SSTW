# Minimal executable paper-experiment preparation

This is an initial-noise watermark fallback, not trajectory watermarking.
Preparation is executable but has no new real results. Do not execute generation,
VAE, inversion, Colab, GPU or actual-data scoring during implementation.

Frozen new roster (full prompts in configs/inversion_paper.json): calibration
OFF only: green tractor/wheat field seed20261101 and black rowing boat/canal
seed20261102. Evaluation: brown horse/fence seed20261201 and orange balloon/hills
seed20261202, each OFF + STATE_A/B + STATIC_A/B. Twelve generated sources total.
All content/seed pairs differ from previously viewed development and holdout.

STATE uses unchanged state_clock trajectories and initial-noise state.write.
STATIC retains identical pad, spatial order, channel0, absolute noise magnitudes
and source latent support1..44, but repeats the initial direction in all eleven
windows for A, and its negative for B; steps/drives are zero. This is a temporal
pattern ablation, not evidence for dynamic observer benefit. Same-scene arms use
the same seed/base noise; the OFF reference is shared across four marked arms.
STATE and STATIC also have different template code distances; this comparison
does not isolate temporal evolution as an equal-code-distance causal effect.

Three views per source: actual full181 MP4; actual lossless RGB129-frame crops
starting at16 and17 after source MP4 decode. Thirty-six views total. Full181 is
encoded/inverted with all46 latent slices, then its fixed first33 slices feed
the unchanged 31-slice/14-shift raw scorer. This is full-context inversion with
fixed-prefix evidence, not aggregation of all video evidence. Crops invert33
slices directly. Do not alter the scorer or normalize raw amplitude.

Every view is decoded by STATE and STATIC books, each with search14 and shift0
channels. Search14 maximizes over both messages and all fourteen shifts; shift0
maximizes only over its two message candidates. Existing observer on/off remains
auxiliary. Each channel gets its own threshold from the SAME calibration OFF
sources: max over three views within source, then max over the two sources.
Accept only score > threshold. No positive examples, previous holdout or eval
data can choose thresholds. Preserve score/message tie rules at1e-12.

All calibration outputs must precede an exclusive immutable threshold.json,
with its SHA passed to eval children before they generate or read eval data.
Any missing/nonfinite calibration view leaves that channel UNCALIBRATED with
no threshold. Evaluation rankings still run; existence decisions remain null.
No calibration replacement, retuning, per-video parameters or success filtering.

Evaluation has10 sources/30 views, eight marked sources/24 marked views and
two OFF sources/six OFF views. Aggregate by source: any-view false acceptance
for OFF; all-view acceptance and correct message for each matched marked
decoder. Also retain every view, missing decision and ranking. Cross-decoder
outputs remain available. Two negative calibration sources and two evaluation
OFF sources are much too small for a low-FPR or broad-generalization claim.
Source clustering avoids treating correlated crops as independent samples.

Save quality versus same-scene OFF for each view, raw state/score records,
MP4s, intermediate/recovered latents, source/config/codebook hashes, failures,
stage wall time and CUDA peak allocation/RSS. Paired PSNR is not perceptual
quality PASS; human inspection of prompt adherence/artifacts/motion is required.

Budget:12 generations=1200 Transformer/600 native steps;36 inversions=3600
Transformer/1800 inverse updates;12 VAE decodes/36 encodes;12 CRF18 MP4 saves
and24 lossless RGB crop saves. Transformer and VAE stages are separate. No
extra generation ablation or parameter sweep. New output root:
MyDrive/Video-WM/InversionPaperPreparation/inversion_paper_<UTC>.
