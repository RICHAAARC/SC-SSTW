# Media-Channel-Retention-V1 fixed diagnostic

This diagnostic reads only the two evaluation cases and five arms already retained by Window-State-MSE-V1.  The immutable input root is
`/content/drive/MyDrive/Video-WM/Window-State-MSE-V1/window_state_mse_v1_20260922T174349464768Z`.
Missing or invalid artifacts remain in the fixed denominator; the runner never substitutes another run.
The retained input seeds are 20260919 for `eval_p2_s2` and 20260920 for `eval_p3_s3`; no new randomness is used.

## Fixed layers and budget

Each of the 10 trajectories has four layers:

1. `TERMINAL`: the retained normalized terminal latent.
2. `FLOAT_RGB_REENCODE`: one deterministic posterior-mode VAE encode of the retained float RGB tensor.
3. `RGB8_NO_CODEC_REENCODE`: `quantize_rgb8_no_codec(rgb).float() / 255`, followed by one deterministic posterior-mode VAE encode.
4. `EXISTING_MP4_G0`: the retained phase-zero latent produced by the old FULL MP4 path.

Only layers 2 and 3 add VAE work: 10 trajectories times two encodes is exactly 20 attempted/completed calls in a successful run.  The runner does not decode latent video, generate, create, decode, or re-encode an MP4, create an attack, calibrate, tune a threshold, or run a deployment receiver.  It reads the retained FULL MP4 bytes only to hash them and bind layer 4 to the old record.  The VAE is loaded lazily once per case subprocess through `load_frozen_vae`.

The retained `result.json` must match SHA-256 `d2492462482b3be4285ababaa6ae3486307e0db56f08ccc7f17e4ede89f89859`, size 11,832,195 bytes, and source commit `8aff4025fd0f8dd656091bf83af1311263f4d99c`.  Terminal tensors are checked with the old `trajectory.fingerprint` field.  FULL MP4 and existing g0 are checked against their old SHA-256 fields.  The old run recorded no decoded-RGB hash, so this diagnostic records its current SHA-256 and explicitly reports that no historical hash baseline exists.

## Measurements and comparisons

Every layer is scored only on the fixed full-support IDENTITY path. `TOTAL` calls `fixed_key.score_path`; `A` and `B` each call `PartitionEvidence.score_path` with their frozen disjoint 80-support partition.  Each A/B partition must contain exactly 40 supports on each q axis.  Every successful layer must contain all 11 valid windows, 7,040 TOTAL matched components, and 3,520 components in each partition.  TOTAL innovation is computed directly; it is never reconstructed by averaging A and B innovation.

The result retains raw score, matched evidence, state innovation, all 11 q vectors, and support metadata.  It also retains marked-minus-same-source-OFF increments for 8 marked trajectories across four layers and three partitions (96 slots), plus the change in those increments across the three adjacent layer boundaries (72 slots).  The adjacent boundaries have narrow meanings: `TERMINAL` to `FLOAT_RGB_REENCODE` is the old VAE decode (including clamp) plus the current posterior-mode encode round trip; `FLOAT_RGB_REENCODE` to `RGB8_NO_CODEC_REENCODE` is the incremental effect of fixed RGB8 quantization propagated through the same VAE; `RGB8_NO_CODEC_REENCODE` to `EXISTING_MP4_G0` is the additional effect of the RGB24-to-YUV420/H.264/readback media chain, not an isolated pure-H.264 term. Raw cross-layer changes are also reported.  `TERMINAL` to `RGB8_NO_CODEC_REENCODE` is explicitly labeled a combined VAE plus RGB8 quantization change and is never called VAE-only loss.

No ratios are used.  Undefined, nonfinite, negative, near-zero, positive, and sign-changing values remain explicit.  The `1e-12` near-zero epsilon is an arithmetic presentation label only and is not a method threshold or PASS criterion.

The fixed denominator is 10 trajectories, 120 layer/partition slots, 96 marked/OFF increment slots, and 72 adjacent increment-change slots.  Process success is only engineering evidence for this diagnostic and is not a scientific or method PASS.

The old OFF calibration already used the complete blind path search and three-view source maximum; this diagnostic does not assume a proven missing-search correction bug. A weak p3 reference-path row means only that evidence remains insufficient under the current statistic and threshold even when that reference path is supplied. Layer and A/B increments can distinguish a weak terminal statistic, added VAE/quantization/media-chain change, and weak B confirmation, while receiver-statistic or threshold suitability remains a separate possibility. These increments are score-space diagnostics, not physical energy. The retained decoded RGB has no historical hash baseline, and the old and current software environments can differ; both facts limit attribution.
