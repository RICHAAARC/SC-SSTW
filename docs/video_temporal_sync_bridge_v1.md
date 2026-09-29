# Video-Temporal-Sync-Bridge-V1

This candidate adds a finite temporal pilot to the successful image-to-Wan
payload reference. It is a fixed one-source implementation, not a calibrated
synchronization detector or a new real-video result. No model, GPU or Colab
execution was performed for this candidate. Earlier methods and evidence are
unchanged; the existing payload reference is not redefined.

## Fixed experiment

The baseline is commit `91afbc71aee9f999ff6f7c1738c5ac0be1249a52` on
`dev/video-trajectory-blind-validation-v1`. The model remains
Wan-AI/Wan2.1-T2V-1.3B-Diffusers at
`0fad780a534b6463e45facd96134c9f345acfa5b`, with the existing copper-kettle
prompt, seed 2026092501, 181 frames at 320x512/8fps, 50 native UniPC steps and
CFG5. OFF, PAYLOAD_LAST and PILOT_LAST run three complete independent native
histories from the same noise. Their z, complete scheduler history, conditional
prediction and unconditional prediction immediately before step49 must match.

OFF has no update. PAYLOAD_LAST retains the prior LAST49 payload: normalized
channels0..3, 32-bit OKOK, eight bits/channel, alpha0.5, the original 240
key-shuffled spatial coordinates per channel and all46 latent times. The band
is h8..19,w12..31, on the final40x64 spatial plane. The transform is the
published implementation's orthonormal FFT real part, not the formal paper's
DCT. Payload key seed remains the sum of Unicode code points, as before.
Wrong key is `watermark-wrong`; NOPE is only a later evaluation target.

PILOT_LAST adds channel4, with64 positions in that same band. SHA256 input
bytes are exactly UTF-8 of:

`domain + NUL + key + NUL + decimal_h + NUL + decimal_w`

There is no trailing NUL; coordinates are unsigned base10 without padding.
Keys must be NUL-free. Coordinate domain is `VTSB1/pilot/coords`. Sort the240
positions by their full SHA256 digest bytes, then numeric(h,w) for an exact
hash tie, and take64. Independently, domain `VTSB1/pilot/sign` gives spatial
PN `2*(digest[0] & 1)-1`. The layout receipt includes actual support, PN,
coordinate order and payload-assignment hashes. Correct/wrong support and PN
hashes must differ; key-string difference alone is insufficient. The pilot
never uses the payload's simple sum-of-characters seed.

The temporal code starts1111 and uses `a[n+4]=a[n+1] XOR a[n]`, giving exactly
111100010011010. Map bits to signs by2bit-1. The finite15-chip sequence does not
cycle. t0 uses chip0; regular t1..45 use chip `floor((t-1)/3)`. The target at a
pilot position is temporal sign times spatial PN times0.5. t0 is written but
excluded from receiver scoring and payload aggregation.

The writer differentiates the complete masked mean-MSE with respect to a
fresh conditional-clean leaf `x0_c=z-sigma*v_c`. PAYLOAD_LAST has N44160 and
eta5520; PILOT_LAST has N47104 and eta5888, including46x64=2944 pilot entries.
Both eta/N equal1/8. The original payload-channel delta is therefore preserved
up to FP32 rounding. Payload and pilot occupy orthogonal channel subspaces;
the receipt separately records their conditional delta squared L2 energy.
There is no total-energy matching. Support increases6.67%; this alone says
nothing about pixel distortion.

Map `delta` back with `v_c'=v_c-delta/sigma`, then one FP32 CFG
`v=u+5*(v_c'-u)`. There is no7.5/5 compensation, tail/denoiser VJP, old
three-direction controller, R* or positive guard. Native scheduler history is
continuous and never reset inside an arm. CUDA Transformer precision is BF16
when supported, otherwise FP16; CPU uses FP32. State, local control, branch
conversion, CFG and VAE remain FP32.

## Saved-media receiver and finite search

Each arm saves a source MP4. Read that saved source once and separately save
FULL_RESAVED181 and CROP5_129=[5,134), both through a second identical
CRF18/libx264/yuv420p codec. FULL is the matched second-codec control. Retain
all three source files and six derived files. No latent slicing substitutes
for RGB cropping and VAE reencoding.

`runtime.wan.video_temporal_sync_bridge.read_mp4_search` receives only MP4
path, public keys/protocol, frozen VAE and observational count/persist
callbacks. Callback return values are ignored. The receiver does not inspect
file or directory names, arm labels, crop origin, messages or writer data.
It uses the actual received frame count:

| Observed frames | Candidates | Phase and regular support |
|---|---|---|
|181|b0 only|g0, j1..45, R45; localization is trivial|
|129|all b0..52|g=(-b)%4, a=(b+g)/4; j1..31 maps to source j+a|

For each required local phase, drop the first g RGB frames and truncate only
the tail to1mod4 frames. Full g0 encodes181->46 latents. Crop g0 encodes129->33;
crop g1..3 encode125->32. All crop candidates use the same31 regular latent
indices. In particular g0's extra final latent is not included. Each actual
phase clears VAE caches before/after posterior-mode encoding and applies Wan
normalization `(raw-mean)/std`. Causal VAE context resets at the local start:
crop5/g3 nominally aligns source8, but is not a full-latent slice identity.

For candidate(b,g,a), X contains64 FFT-real values over its R regular observed
latents; P contains the corresponding spatial PN times finite temporal code
at source indices j+a. Score is soft cosine
`sum(X*P)/sqrt(n*sum(X**2))`, n=64R, with FP64 reductions. Exactly zero energy
has score0 and explicit NO_ENERGY. Nonfinite observations/reductions fail.
Scores are not hard-sign correlations. All1/53 candidates, scores, top gap
and ties remain visible. A failed phase produces SEARCH_INCOMPLETE; the
candidate set is never reduced to the available phases.

The public gate is S>=0.5, without calibration or scanning. Ties satisfy
`maxS-S<=1e-12` absolute tolerance. Below threshold gives NO_PILOT; multiple
above-threshold ties give AMBIGUOUS; a unique passing candidate gives LOCATED.
A numerical winner is not a confidence estimate. Full's sole candidate cannot
provide nontrivial localization evidence. Payload bits never pick a phase or
break pilot ties.

The bit voting rule remains the original hard-sign majority: zero votes0,
exact ties follow first vote order. The temporal aggregation support changes:
the old successful reader used all46 latents and1380 votes/bit; this receiver
excludes t0 and uses45/full or31/crop regular latents. It is not an unchanged
receiver. The same regular phase support gives1350 votes/bit for full
and930 for crop. Two public keys share each actual VAE phase encode. Thirty
raw phase payload observations and their pilot features are persisted before
truth joins. A pilot-selected phase alone supplies an accepted primary
payload. NO_PILOT/AMBIGUOUS are completed negative results with
`accepted_payload=false` and unavailable primary BER/exact; they are not
engineering failures. Fixed-g0 raw payload remains separately visible,
including expected PAYLOAD_LAST payload recovery without a pilot.

## Evidence, accounting and failure handling

The fixed primary roster is3 generation rows,3 source MP4s,6 derived MP4s,
15 normalized phase artifacts,30 phase payload reads,324 pilot candidates,
12 searches and24 posthoc rows (registered/wrong message for each search).
Expected pilot presence is only PILOT_LAST+CORRECT, one full and one crop.
The10 pilot negatives comprise OFF unwatermarked4, PAYLOAD_LAST payload-only4,
and PILOT_LAST wrong-key2. They are related controls, not10 independent
unwatermarked samples or an FPR estimate. Payload-only is not described as
having no watermark.

Normalized phase tensors are saved with byte hashes. Blind phase features,
candidates and decisions are atomically saved before truth evaluation.
The posthoc oracle registers b0 or b5 only after this persistence and reuses
the already saved phase feature and candidate score. It runs no new FFT,
search or VAE. Wrong-grid recovery of the temporally repeated payload does
not demonstrate correct localization. Both oracle and fixed-g0 results stay
separate from accepted primary payloads.

| Planned actual call group | Count |
|---|---:|
|Conditional / unconditional Transformer|150 /150|
|Native steps / local gradients|150 /2|
|VAE decode / actual phase encode|3 /15|
|Source saves / derived saves|3 /6|
|Source readbacks / primary readbacks|3 /6|
|Primary phase-feature reads / pilot candidate scores / searches|30 /324 /12|
|Separate terminal feature reads / terminal pilot scores|6 /6|
|Posthoc saved-feature references / new oracle FFT or VAE calls|12 /0|
|Separate quality MP4 reads|6|

Attempted and completed calls are recorded by stage. Terminal observations
exclude the first latent, report payload and pilot, and never enter the MP4
receiver. Primary30/324 are fixed primary operations, not totals including
terminal diagnostics. Missing terminal diagnostics are retained separately.

Generation and media run in serial fresh children of current `sys.executable`.
The generation child loads no VAE and exits before the FP32 VAE/media child.
All original slots are created before work. Phase/VAE/media boundaries are
persisted; per-key candidate scores are batched atomically at the search
boundary to avoid hundreds of large Drive rewrites. A hard exit leaves all
initial candidate slots and already saved phase data. Parent start/monitor
errors terminate and wait for a running child before reloading its latest
checkpoint. Completed rows survive; unfinished rows become explicit missing
or SEARCH_INCOMPLETE. User interruption does not start subsequent workers.

Quality reads each derived MP4 once more and reports framewise FP64 RGB RMSE
and PSNR for PAYLOAD_LAST vs OFF, PILOT_LAST vs OFF, and PILOT_LAST vs
PAYLOAD_LAST within the same view. Quality has no acceptance threshold and
cannot overwrite blind evidence. Source, derived, primary and quality I/O
counts are distinct. Notebook previews show retained source and derived MP4s;
preview errors preserve the saved path and do not change experiment results.

## Local validation and handoff

The CPU suite checks FP32 payload-delta preservation against the unchanged
prior implementation, orthogonal energy, independent SHA layout encoding,
LFSR/mapping, soft-cosine boundaries, finite-offset synthetic localization,
three complete native histories and pre49 identity, renamed synthetic MP4
blindness, partial phases, partial saves, normal-rejection completion, process
fault recovery, terminal accounting, batch hard-exit retention and notebook
static receipts. Synthetic media and fixture VAE are not real Wan evidence.
The main session also supplied a separate FP64 algebra/mapping receipt at
`diagnostics/temporal-sync-bridge-main-review-20260929/independent_algebra_and_mapping.json`
in the parent project; its arbitrary64-coordinate algebra test does not
replace the actual SHA layout and FP32 implementation tests.

The notebook starts with the exact two-line Drive mount. It retains the
working current-Python, Torch/Torchvision import probe, scoped dependency
ranges and fresh-child workflow from the prior notebook. There is no venv,
ensurepip, exact Python or GPU-model hard gate. Draft `SOURCE_SHA=None` is
intentional. Publication, verified immutable binding and user-run Colab
execution belong to the main session after review. This local candidate does
not push, load model weights, run GPU or alter previous evidence.
