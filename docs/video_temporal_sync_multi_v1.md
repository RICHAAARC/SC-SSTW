# Video-Temporal-Sync-Multi-V1

This is a fixed MULTI writer plus a public window-observation interface. It
inherits the bridge carrier and global-search diagnostic without changing the
previous GROW/bridge implementations or evidence. No new model, GPU or Colab
execution has occurred. Global exact localization is a diagnostic comparison,
not a prerequisite for the later state-path route. No DP or new threshold is
implemented here.

## Fixed writer and comparison

The base is `b255029ec508631f38a124ac49712a5b9299d2a1` on
`dev/video-trajectory-blind-validation-v1`. The model remains
Wan-AI/Wan2.1-T2V-1.3B-Diffusers at
`0fad780a534b6463e45facd96134c9f345acfa5b`, with the same copper-kettle prompt,
seed2026092501, 181x320x512 frames, 8fps, 50 native UniPC steps and CFG5.

OFF, PAYLOAD_MULTI and PILOT_MULTI start from the same initial noise and
independent copies of the entire native scheduler. Immediately before the
first update at index25, their z/history/conditional/unconditional fingerprints
must match. Marked arms then perform25 actual updates at indices25..49. Each
step recomputes a fresh conditional clean estimate and its local gradient,
maps delta back through `v_c'=v_c-delta/sigma`, performs FP32 CFG5 and takes
one native step. Nothing resets history or reuses the last-step gradient.
Every step records index, sigma and enabled state; controlled steps also
record N, eta, loss and payload/pilot/total conditional delta squared energy.
Trajectories may legitimately diverge after index25.

The pure module maps the new arm names to the unchanged bridge target
construction. Payload uses channels0..3,32-bit OKOK, eight bits/channel,
240 original key-shuffled coordinates per channel, alpha0.5 and46 constant
latent-time repetitions. Pilot uses channel4 and64 independent SHA256-ranked
coordinates/spatial PN, the same finite15-chip sequence111100010011010 and
three regular latent times/chip. t0 still writes chip0. The exact domains,
byte encoding and no-wrap temporal code remain those of
`main/tube_state/video_temporal_sync_bridge.py`.

At every controlled step PAYLOAD_MULTI has N44160/eta5520 and PILOT_MULTI
N47104/eta5888, so eta/N=1/8. There is no division by25 or total-budget matching.
The claim is unchanged local operator strength on the same input. It is not
cross-arm equality of actual later deltas and not a0.875^25 terminal-residual
law. The native denoiser, scheduler and successive controlled states change
the next input. The construction is the author code's FFT-real/mean-MSE
operator, not the paper's DCT; no denoiser/tail VJP or old three-direction
controller is introduced.

CUDA Transformer dtype remains BF16 when supported, otherwise FP16; CPU uses
FP32. State, local control, conditional/unconditional conversion, CFG and
frozen VAE are FP32. No GPU-model or exact-Python hard gate is introduced.
Generation and media run in serial fresh current-Python children, releasing
the Transformer before the VAE/media process starts.

## Five saved views and unchanged global diagnostic

Each of the three source MP4s is retained and read once. Five derived MP4s
per source use the same second libx264/CRF18/yuv420p codec:

| View | Stored-source frame interval | Posthoc true(b,g,a) |
|---|---|---|
|FULL_RESAVED181|[0,181)|(0,0,0)|
|CROP4_129|[4,133)|(4,0,1)|
|CROP5_129|[5,134)|(5,3,2)|
|CROP6_129|[6,135)|(6,2,2)|
|CROP7_129|[7,136)|(7,1,2)|

These source intervals belong only to derivation and later reporting. The
receiver sees MP4 bytes, public key/protocol and frozen VAE, and never uses a
file/directory label or true crop origin. Actual received length181 yields
one full candidate and phaseg0; length129 yields all53 candidate offsets and
all four RGB phase reencodings. Each phase is cropped locally and tail-trimmed
to1mod4 frames, encoded with posterior mode, normalized using Wan mean/std,
and clears VAE caches. It is not a slice from the full normalized tensor.

The reused global score remains FP64 soft cosine, threshold0.5 and absolute
tie tolerance1e-12, including NO_ENERGY, NO_PILOT, AMBIGUOUS and
SEARCH_INCOMPLETE behavior. Primary support remains full j1..45 or crop
j1..31:1350/930 hard votes per payload bit. Zero votes0; a total tie follows
the first coefficient vote. Pilot alone selects a primary phase. Normal
rejection is valid completed computation with no accepted payload/BER, and
oracle or fixed-g0 readouts never replace a rejected/incomplete primary.
Full localization is trivial. Correct bits at the wrong global grid do not
show synchronization of the repeated payload.

## Public window files

For each actual phase and key, one independently hashed JSON artifact records
all observed regular windows j>0. The resulting102 files contain3270 rows:

- Full:45 rows per phase/key.
- Crop phaseg0:32 rows; phasesg1..3:31 rows each.
- Per source and key:45+4*(32+31+31+31)=545 rows.
- Across three sources and two keys:3270 rows.

Cropg0's extra j32 is exclusively a future observation. It never enters the
current global candidate score, primary payload aggregation or threshold.
The51 full normalized phase tensors and previous primary pilot FFT/decoded
bits/votes are also retained. One float32 FFT per phase/key supplies both the
unchanged primary extraction and window statistics; export does not cause
another VAE pass.

Each artifact contains schema `video-temporal-sync-multi-window-v1`, public
key/layout fingerprints, received frame count, local phaseg, used received
frame interval, first-latent-excluded status and primary regular count. Each
window records:

- observed_regular_j and phaseg;
- nominal stride4 and half-open received new-input interval
  `[g+4j-3,g+4j+1)`;
-64 pilot FFT-real coefficients;
-32 payload-bit records, each containing FP64 sum and sumsq over its30
  key-ordered coefficients; positive, negative, exact-zero and support counts;
  and `first_bit=int(first_coefficient>0)`.

The nominal interval names newly supplied RGB frames on a stride4 grid. It
is not the VAE receptive field: causal cache and the first local latent retain
the same context caveat as the bridge. No source offset b, hidden tau, source
arm, attack origin, CORRECT/WRONG label or file path enters artifact content.
Path/hash/slot references live in the outer result table only. No redundant
mean is stored. Negative plus exact-zero counts are the hard zero-vote count.

`aggregate_ordered_windows` is a pure, truth-free utility over an explicitly
supplied sequence of observed rows. It supports reordering, repetition and
skipping; counts and soft sums aggregate in that order. If global hard counts
tie, the bit is the first coefficient bit of the first selected window, not
that window's majority. Tests include the case where these disagree.
This helper estimates no clock or path and receives no message truth.

The later state-space interpretation concerns a watermark state s_n and a
hidden correspondence tau_i from received observations to source state/time.
Advance, repeat and skip describe possible future state-path transitions.
This stage exports the local observations needed to study those transitions;
it does not infer tau_i, enumerate3233 paths, adopt DP parameters or calibrate
a new DP score/gate. The current repeated payload is not time-dependent and
cannot establish that synchronization is necessary for its recovery.
Time-dependent payload remains disabled and unverified.

## Fixed accounting and failure semantics

| Fixed artifact or primary operation | Count |
|---|---:|
|Source MP4 / derived MP4|3 /15|
|Full normalized phase tensors|51|
|Primary phase-key reads / searches|102 /30|
|Global pilot candidates / posthoc evaluations|1278 /60|
|Window files / exported rows|102 /3270|

Planned actual calls are300 Transformer forwards,150 native steps,50 local-loss
gradients,3 VAE decodes and51 phase encodes. Source reads3, primary
MP4 reads15 and extra quality reads15 have separate counters; source/derived
saves are3/15. Terminal diagnostics add6 feature reads and6 pilot scores,
separately from102/1278 primary operations. Posthoc oracle references30 saved
phase/candidate records and adds no FFT, VAE or score computation.

The expected pilot-positive conditions are five PILOT_MULTI/correct-key views:
one trivial full and four crops. The25 negatives are OFF10, payload-only10,
and marked-wrong-key5. PAYLOAD_MULTI is not unwatermarked, and these related
controls do not form25 independent unwatermarked FPR samples.

All fixed slots and expected window row counts exist before execution. A
window artifact is atomically saved once per phase/key; only its path/hash,
schema and row count are embedded in result/blind metadata. There is no
per-window rewrite of the large result. Canonical result.json commits before
its recoverable blind projection. The per-key candidate batch commits at the
search boundary, preserving already committed phases across failure.

Window export failure retains valid primary observations and search results,
but the interface and overall run cannot report complete until all102 files
and3270 rows are committed. Missing view/phase/window rows remain failed or
missing, with the original denominator. No export failure triggers a VAE retry.
Parent worker startup/monitor cleanup retains the bridge's terminate/wait
then reload behavior; user interruption does not start subsequent workers.

Within each of the five views, quality reports payload/off, pilot/off and
pilot/payload comparisons:15 metric pairs using15 additional MP4 reads.
Framewise FP64 RGB RMSE/PSNR has no threshold, modifies no blind evidence and
cannot convert an engineering result into a quality PASS.

## Validation and user-run notebook

The CPU target checks actual25..49 fresh updates and pre25 identity, complete
new rosters/four truth maps, independent window soft/count calculations,
ordered/repeated/skipped/tie behavior, primary equality with the unchanged
bridge, extra-j32 isolation, public metadata, renamed input, combined partial
view/phase/export failures, phase and window canonical interruption boundaries,
normal rejection completion and notebook setup/source requirements.

An optional local golden check reads an already saved LAST phase tensor and
the main session's independent NumPy float64 receipt. It does not call a model
or VAE and is not new real-video evidence. Its path/hash and explicit test
environment are in `diagnostics/video_temporal_sync_multi_v1/local_validation.json`.
On a fresh checkout without that external receipt, the optional golden test
is explicitly skipped; the recorded implementation run supplied it and had
zero skips. The borrowed CPU environment is not modified. CUDA disabling is a
local test setting only and does not enter the product notebook.

The new notebook retains exact Drive mount first, current sys.executable,
Torch/Torchvision import probe, the existing verified requirements file and
fresh children, with guarded `Video(str(path),embed=True)` previews. Setup
failure retains the full new roster. `SOURCE_SHA=None` is intentional until
review, source publication and verified immutable binding. No GPU/Colab/model
execution or source push is part of this local implementation.
