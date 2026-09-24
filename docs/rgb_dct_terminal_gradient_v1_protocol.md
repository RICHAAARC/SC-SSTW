# RGB-DCT Terminal-Gradient-V1 fixed source protocol

This source implements route B from the frozen 2026-09-24 task card whose
SHA-256 is `d1ab995b1a1525e1996e1df9d670afe009a5a8a40a52f51e813ab2fbbdaff071`.
It is an execution-ready implementation contract, not a GPU or method result.

The two immutable sources are `eval_clock_s2431` / seed 2026092431 / prompt
`locked camera, a round wall clock with black hands on a plain white wall, steady indoor light, no people, no cuts`
and `eval_umbrella_s2432` / seed 2026092432 / prompt
`locked camera, a yellow umbrella turning slowly against a plain gray background, steady soft light, no people, no cuts`.
Each source has exactly OFF, original SINGLE49, and TERMINAL46, for six MP4
slots. All slots are persisted before model work and failures remain in place.

Wan is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision
`0fad780a534b6463e45facd96134c9f345acfa5b`, 320x512, 181 frames, 8 fps,
50 native UniPC steps and CFG 5. Media is libx264 CRF18 yuv420p; the blind
decision reads complete FFmpeg RGB24 and uses the frozen receiver C=sum(q_g>0),
C>=24:H1. Continuous RGB is never a decision input.
For every arm, the same frozen receiver records status, all 30 q values and C
at three evidence layers: decoded float RGB, the exact
`runtime.wan.vae.quantize_rgb8_no_codec` RGB8 raster divided by 255, and the
final MP4 RGB24 readback. Only the MP4 layer receives an H0/H1 decision.

TERMINAL46 differentiates the fixed loss L=mean_g relu(-q_g)^2 with respect to
the live CFG v46 through the native T46 step, attached UniPC history, true
T47-T49 CFG/steps, and frozen FP32 VAE decode. The implementation uses an exact
split VJP: a same-history no-gradient terminal is decoded with a gradient VAE to
obtain its terminal cotangent; after releasing VAE storage, a same-history
differentiable T46-T49 replay maps that cotangent to v46. The replay terminal
and final UniPC history must exactly match the no-gradient forecast. The full
181-frame differentiable proxy is checked against the NumPy receiver within
fixed tolerances. A detached z47-only gradient, local proxy, finite difference,
or analytic lift is invalid.

The negative gradient is zeroed at latent time endpoints and supported on
indices 1:45. A unit velocity perturbation is passed through the same native
T46 history; its immediate z47 response calibrates one controlled T46 update to
support RMS R*=0.042943312697648145. The arm then runs the free native T47-T49
tail. SINGLE49 keeps the original positive VAE lift and same-history response
calibration at the same R*. No scan, retry, fallback, source replacement,
threshold change, visual PASS gate, GPU whitelist, or CUDA build-suffix gate is
allowed.
If the shared OFF/cotangent phase fails, dependent SINGLE49 is explicitly
`NOT_RUN_ENGINEERING_INVALID`. A finite zero lift or native response without a
fixed MP4 is `ENGINEERING_INVALID` with its original zero reason; it is not a
method-negative media result.

Per source maximum top-level calls are: generation 1; Transformer prefix 100,
terminal forecast 6, differentiable replay 6, controlled tail 6; scheduler
prefix 50, forecast 4, replay 4, response shadow 1, unit probes 2, controlled
steps 2, controlled tail 3; gradient VAE decode 1, final VAE decode 2, VAE
encode 2; VAE VJP 1 and tail VJP 1; MP4 save/read/score 3 each. The experiment
totals are exactly twice those maxima. Checkpoint Transformer-block and VAE
causal-chunk forward/recompute calls have separate attempted/completed ledgers
and per-source caps of 256 and 64 respectively. Each source is a fresh serial
child process with a 14400-second wall timeout. Every stage, failure, top-level
call, recomputation call, fingerprint, proxy value, native response, terminal
delta, full-frame MP4 RMSE, CUDA peak, host peak and elapsed time is persisted.
The parent writes the six-slot result before starting children and monitors each
child's RSS and elapsed time in a separate atomic receipt. Timeout sends SIGTERM,
waits one fixed 10-second grace period, then uses SIGKILL if still necessary;
there is no retry. Environment/CUDA checks run only inside the fresh child and
preserve their original `ENVIRONMENT_CHECK` error. COMPLETE or NEGATIVE requires
six SCORED slots, two SCORED cases, no failures, successful release, completed
worker receipts, and valid environment/resource receipts.

PyTorch must provide CUDA, autograd saved-tensor hooks, non-reentrant checkpoint
and a working differentiable scheduler step. Diffusers is fixed at 0.40.0.
The complete PyTorch version and CUDA build suffix are recorded but neither is
a version-string gate; current `+cu130` and compatible builds are accepted when
the required APIs and CUDA execution are available.
