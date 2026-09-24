# RGB-DCT-Local-Gradient-V1 fixed protocol

This source implements route A from the frozen 2026-09-24 RGB-DCT gradient
route card. It does not authorize or perform a GPU, Colab, Drive, or scientific
run.

The model is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at revision
`0fad780a534b6463e45facd96134c9f345acfa5b`, with 320 by 512 RGB, 181 frames,
8 fps, 50 UniPC steps, CFG 5, FP32 VAE, and the fixed key
`WanProjection-first-validation-key-v1`. The two immutable sources are
`eval_clock_s2431`, seed 2026092431, prompt `locked camera, a round wall clock
with black hands on a plain white wall, steady indoor light, no people, no
cuts`; and `eval_umbrella_s2432`, seed 2026092432, prompt `locked camera, a
yellow umbrella turning slowly against a plain gray background, steady soft
light, no people, no cuts`.

Each source has exactly three same-noise, same-prefix slots: OFF, the existing
no-gradient SINGLE49 positive reference, and LOCAL44_46_48. Every slot and
failure remains in the fixed six-MP4 denominator. MP4 uses libx264 CRF 18,
yuv420p, and the receiver reads ffmpeg RGB24. The blind receiver receives only
the original full MP4 and key. It computes all 30 frozen group scores and
decides H1 exactly when C=sum(q_g>0) is at least 24. Continuous proxy values do
not decide the final result.

At each of T44, T46, and T48, LOCAL44_46_48 recomputes CFG velocity from that
arm's live state and complete live UniPC history. It forms
`z0hat=z_t-sigma_t*v_t`, performs an autograd-enabled frozen FP32 VAE decode of
all 181 RGB frames, and reproduces the fixed 160-block RGB-DCT features,
181-frame temporal centering, key signs, and
`D=sqrt(mean(d^2)+(1/255)^2)` in float64 reductions. Before backward, all 30
differentiable q values must match the NumPy receiver q values on that exact
decoded RGB tensor within the fixed deterministic tolerance. The minimized loss
is `mean_g relu(-q_g)^2`. A nonfinite or zero masked gradient, proxy mismatch,
or broken input gradient makes that slot engineering-invalid.

Each gradient VAE decode uses the native causal-frame decoder with nonreentrant
checkpoint recomputation and CPU-packed explicit cache boundaries. The fixed
181-frame geometry has 46 latent-frame decoder chunks, so each source permits
exactly 138 forward chunk calls and at most 138 recomputation calls across the
three gradient decodes. Forward/recompute attempts, completions, cache boundary
sizes, and failures are recorded separately. Spatial tiling is disabled.

The negative velocity gradient is zero at latent time endpoints and supported
only on indices 1:45. At each controlled step, an unmodified same-history
native shadow step and a same-history unit velocity-response probe normalize the
direction to support RMS `R*/3`, where
`R*=0.042943312697648145`. The controlled native step must match that response
target, followed by normal recomputation and free sampling. There is no T49
LOCAL update. Per-step RMS values sum nominally to R*; this is not a claim about
squared budget, terminal displacement, or quality equivalence.

The run records attempted/completed calls, raw failures, pending and invalid
slots, gradient norms, proxy q/C/loss and parity, probe and actual responses,
state/velocity/history fingerprints, terminal support RMS versus OFF, MP4 RGB
RMSE versus OFF, peak CUDA and host memory, environment versions, paths, hashes,
and elapsed time. Workers are serial fresh processes with no automatic retry,
scan, tuning, source replacement, analytic lift fallback, or GPU model-name
gate. Static and CPU tests establish execution readiness only.
