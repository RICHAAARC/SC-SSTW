# GROW-style frequency video adaptation v1

An implementation candidate, not an official GROW reproduction or an executed
experiment. Source/notebook publication is authorized for the user to run in
Colab. The assistant has not executed models/GPU/Colab/Drive experiments.
Reference: https://openaccess.thecvf.com/content/CVPR2026/html/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.html
The original interval notation and earlier/small-r wording remain ambiguous;
this adaptation fixes explicit zero-based indices instead of claiming equivalence.

Frozen first-round parameters: Wan2.1 1.3B, normalized latent [1,16,46,40,64],
181 frames, 320x512, 8fps, 50 UniPC steps, native CFG5 followed by FP32 velocity.
Same two existing content prompts and two seeds; OFF/A/B =12 videos.
Every channel/time spatial slice uses the whole HxW coordinate system. Only
channel0 frequencies u,v=2..9 are selected (64 non-DC coefficients), with
orthonormal 2D DCT-II. Key-derived permutation assigns index modulo16 to a
16-bit payload, four coefficients/bit/time; identical payload repeats at all46
times, including the first slice. B complements A. No pairing, local blocks,
pilot, alignment/time search, observer or multichannel capacity expansion.

Selected coefficients target alpha*bit, alpha=.5, in normalized DCT units.
L=.5*sum((selected_DCT(z0hat)-target)^2), over all46*64 coefficients. There is
no adaptive spectral normalization, norm clipping or mean-loss rescaling.
At fixed indices10..29 and positive finite sigma, form detached post-CFG
z0hat=z-sigma*v, compute its local leaf gradient g, u=-.1*g, v'=v-u/sigma.
Then perform exactly one native UniPC step with v'. History is preserved.
All other steps are ordinary. No backward through Transformer, fixed tail
direction, shadow step, alternative solver, or copied eta100 is used.
Native flow_prediction/predict_x0/no-threshold/no-solver_p is checked.
Record intended20 and actual controlled counts separately; skips cannot count
as completed control. Actual-step delta means z_next-z, not a controlled-versus-OFF causal delta.
Local DCT descent is not a final-terminal guarantee.

Reader: sign of each selected coefficient, four votes/bit/time; sum all184
votes/bit across46 times. Zero coefficients and zero totals are explicit
abstentions/erasures. Soft coefficient means are auxiliary, never the main
payload decision. Record every slice and aggregate recovered16-bit signs,
erasures, and exact/BER comparisons against A and B. OFF coincidental exact
matches are reported without a calibrated detection/FPR claim. No truth is
passed to read(); candidate payload comparisons follow decoding.

Four layers from each same-run terminal/decode: terminal normalized latent,
floatRGB to VAE, RGB8 (rint rounding) to VAE, and CRF18 yuv420p MP4 readback to
VAE. Shared frozen VAE adapters restore normalized latent coordinates. Every
layer retains46 slice entries, even on failure. Save all terminal/receiver
latents, MP4s, initial noise, codebook, config, step controls and call counts.
Quality compares the same RGB layer against same-case OFF, recording RGB MSE,
PSNR, residual temporal MSE and temporal energy ratio. No quality tolerance or
scientific PASS is invented; execution completeness and bit recovery differ.

CLI for a later authorized run: python -m experiments.wan_state_clock.grow_frequency_run --output NEW_DIRECTORY
Each of4 cases runs in its own process; all3 arms share its initial noise and
conditioning, with independent fresh scheduler copies. Full plan:1200 Transformer
forwards,600 real scheduler steps,160 local gradients,12 decodes/MP4 saves,
36 VAE encodes. Failures and missing cases keep the fixed12-video/48-layer roster.

Published source: `e04f48ed2e85a6d9ae8d9b639c7c9bd6e9d7bddb`.
`notebooks/grow_video_frequency_colab.ipynb` binds this immutable source.
Run all writes a unique new directory under `MyDrive/Video-WM/GROWVideoFrequency`.
