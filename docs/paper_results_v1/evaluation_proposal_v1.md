# Paper Results V1 evaluation proposal

Status: **proposal only; no roster, message, mapping, reducer, threshold, or
execution budget in this document is adopted.** The executable companion is
`experiments/paper_results_v1/real_eval.proposal.json`. It is never selected by
default because `real_cli` always requires an explicit `--config` and phase.

## Method and claim ceiling

The current main method is
`trajectory-payload-framewise-sync-v1` at repository baseline
`527c4800292c296222c2c0533809eccb82a29a65`. The current full method comprises
the P0 trajectory followed by the separate framewise/M05 branch. P0 itself is
only `PAYLOAD_NATIVE` and uses
`runtime.wan.video_trajectory_conditional_joint_v1.run_trajectory` exactly: one
50-step native Wan path, payload control at steps 25 through 49, full-space
real/orthonormal FFT target, repeated 32-bit payload, and pilot contribution
zero. Generation is not a local state-plus-payload encoder.

The proposed claim is limited to exact recovery of one predeclared 32-bit task
under a finite public offset/phase and single-deletion set. It does not establish
arbitrary 32-bit capacity, source independence, generalization, presence
detection, calibrated FPR, or a population guarantee. The 2026-10-08
development evidence remains a fixed `OKOK` result on one seen source. It is
not renamed as an unseen or arbitrary-message result.

The real runner accepts an explicit list of 32 bits. This is a new outer input
interface over the existing strict 32-bit target builder, not evidence that
arbitrary messages have been recovered. The proposal uses `0xA6D39C5E` so a
future confirmation would be visibly distinct from historical `OKOK`; this
message remains unadopted.

## Proposed source split

Run two integration pilots first, then discard them from the confirmation
denominator. Freeze the eight confirmation cases before any confirmation run.
All ten prompts and seeds below are candidates whose unseen status against the
complete archive is **unverified**. They deliberately exclude the known
yellow-sailboat seed `2026100701` and copper-kettle seed `2026092501`.

| Cohort | ID | Seed | Proposed prompt |
| --- | --- | ---: | --- |
| pilot | pilot_01 | 2026110101 | matte red paper windmill rotating slowly on a gray slate tile |
| pilot | pilot_02 | 2026110102 | small blue wooden tram rolling slowly along a straight tabletop track |
| confirmation | confirm_01 | 2026110201 | white porcelain fox figurine beside a fern frond on a dark oak shelf |
| confirmation | confirm_02 | 2026110202 | violet glass sphere drifting in a shallow clear acrylic tray |
| confirmation | confirm_03 | 2026110203 | green tin robot walking slowly across a plain cork board |
| confirmation | confirm_04 | 2026110204 | folded orange paper crane turning slowly on black felt |
| confirmation | confirm_05 | 2026110205 | silver toy submarine gliding through a clear rectangular water tank |
| confirmation | confirm_06 | 2026110206 | purple wool ball rolling slowly along a pale concrete channel |
| confirmation | confirm_07 | 2026110207 | yellow wooden spinning top rotating on a matte blue tabletop |
| confirmation | confirm_08 | 2026110208 | black ceramic cat figurine on a slowly rotating white platform |

The JSON companion contains the full prompts, negative prompt, seeds, and all
planned observation rows. Pilot output may repair plumbing, but it cannot set a
threshold, choose a source, replace a failed confirmation row, or enter the
eight-source confirmation summary.

## Arms, codec, and finite edit set

For each source, OFF and P0 start from the same prompt, seed, initial noise, and
pristine 50-step scheduler. OFF uses the same conditional/unconditional model
calls, FP32 CFG `u + 5*(c-u)`, and native scheduler steps while disabling the
payload term. P0 calls the frozen conditional-joint trajectory directly. A
Wan-native decode creates OFF PRE and P0 PRE. P0 PRE is encoded once by the
frozen framewise VAE; P1 decodes the untouched latent and M05 decodes an
independent written copy. All four main arms then use the same explicit
8-fps/libx264/CRF18/yuv420p codec path.

Each main arm has nine logical received views over eight unique index maps:
FULL181; six other public global phase/window maps; and two single-jump views.
GLOBAL views predeclare RAW and GLOBAL receiver rows. SINGLE_JUMP views
predeclare RAW, GLOBAL, and PATH rows. With two keys this is 40 blind rows per
arm and 160 per source. The eight confirmation sources therefore have 1,280
blind rows and 40,960 planned recovered bits. FULL181 remains in the artifact
and payload denominator but is a geometry/control row, not one of the eight
primary synchronization-effect views. No oracle row enters blind selection.

Edits are deterministic index selections from complete POST RGB8 artifacts and
do not add a codec pass. If edited media are later re-encoded, those operations
need separate manifest and cost rows rather than being absorbed into the counts
below.

The proposed paper-facing external comparison predeclares exactly one main
readout per source and logical view: the full M05 arm with correct key K0;
non-FULL GLOBAL views use `GLOBAL`, SINGLE_JUMP views use `PATH`, and FULL181
uses `RAW` in a separate geometry table. No result-dependent best-mode choice
or average over all 160 receiver rows is allowed. For eight confirmation
sources, the eight non-FULL views give 64 paired rows per external method; the
eight FULL rows remain a separate control table. All 160 receiver rows per
source remain in the package as wrong-key controls, arm ablations, RAW
controls, and synchronization-effect evidence. The 72 source-by-logical-view
rows are exactly eight sources times nine views: 64 non-FULL comparison rows
plus eight FULL controls. They are clustered by source and are not 72
independent samples.

## External baselines and 32-bit task mapping

Both baselines embed the same OFF PRE content and then traverse the same shared
codec. Missing weights, incompatible APIs, non-finite outputs, missing frames,
or wrong shapes are retained failures. Expected bits enter only the final
evaluation phase, never embed-free extraction or blind receiver planning.
Each baseline has nine logical extraction rows over the same eight unique edit
maps used by the main arms. Implementations may reuse one native detector call
for byte-identical expanded maps, while every logical row and failure remains
in the fixed denominator.

The proposed VideoSeal model is the official VideoSeal v1.0 256-channel card at
commit `870ca7fb33578b90f14c602016b6c2788096226e` (2026-07-02, `Fix header
copyrights`), loaded with
`videoseal.utils.cfg.setup_model(card, local_checkpoint)`. At that commit the
card declares `args.nbits=256` and names checkpoint object
[`y_256b_img.pth`](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth).
The runner does not call that download URL: the user must supply the local
object and its actual digest. For the effective-32 task, channel `j` carries
bit `j mod 32`, giving eight native channels per information bit and rate 1/8.
After removing the presence channel, average every received frame/spatial soft
value for each native channel, then average the eight channels belonging to
each effective bit. A value `>0` decodes as one; `<=0` decodes as zero. Exact
zero retains the native zero-bit decision and is reported in `tie_count`.
The recommended paper comparison keeps that native bit and reports the tie;
it does not turn the row unevaluable. The current runner also implements an
unadopted, stricter candidate that marks any reduced zero
`UNEVALUABLE_ZERO_TIE`. Its explicit proposal rule name is
`CHANNEL_J_MOD_32_REPEAT_MEAN_STRICT_GT_ZERO_ZERO_TIE_UNEVALUABLE`. If native
`K` is not divisible by 32, this mapping fails
closed; no prefix, truncation, or filler is inferred.

That mapping changes the 256-channel codeword distribution and adds channel
redundancy. It compares the same 32 information bits with different native
encoding cost; it does not match code rate, embedding strength, or redundancy.
It also does not prove 256 independent information bits. A true native-K table
requires a separate independent K-bit message mode. If adopted, that optional
mode adds eight VideoSeal embeds, eight complete codec roundtrips, and native
extraction for every edit; it is a separate table and changes the eight-source
codec count from 48 to 56. Historical prefix scoring cannot substitute for
either proposed mode.

RivaGAN uses source pin
`efffa72a4ca46d4d5051f6970c96424c2cdab441` and the explicitly identified
Peachypie98 community 32-bit checkpoint, not official DAI-Lab weights. The
tensor backend bypasses upstream's hidden mp4v/20-fps path, follows BGR uint8,
`value/127.5-1`, `[1,3,1,H,W]`, encoder clamp, and uint8 truncation, then uses
the shared codec. Its proposed sequence rule is an equal-weight soft-logit mean
for each bit over every predeclared received frame, followed by native `>=0`.
Mean zero retains native bit one and is reported in `tie_count`. The
recommended paper comparison keeps that bit. The current runner's stricter,
unadopted candidate instead marks the row `UNEVALUABLE_ZERO_TIE`; its explicit
proposal rule name is
`ALL_DECLARED_FRAMES_EQUAL_LOGIT_MEAN_GE_ZERO_ZERO_TIE_UNEVALUABLE`. Empty video,
missing or extra frames, non-finite logits, or any frame shape other than
`[32]` fails the whole row; frames are never skipped to improve a result.

The loader implementation is based on pinned official VideoSeal
[`videoseal.py`](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/models/videoseal.py),
local-only [`utils/cfg.py`](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/utils/cfg.py),
and [`videoseal_1.0.yaml`](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/cards/videoseal_1.0.yaml).
The official
[`e00b98c...870ca7f` comparison](https://github.com/facebookresearch/videoseal/compare/e00b98c7ca77eb1fb5b9b68260e7c6c8fc207a84...870ca7fb33578b90f14c602016b6c2788096226e)
changes only four copyright-header files; the card, `cfg.py`, and algorithm are
unchanged. RivaGAN loader evidence uses pinned
[`rivagan.py`](https://github.com/DAI-Lab/RivaGAN/blob/efffa72a4ca46d4d5051f6970c96424c2cdab441/rivagan/rivagan.py).
These links specify interface evidence; no source or weight is fetched by the
runner.

The native-tie recommendation and the stricter implemented candidate are both
still unadopted. If the user selects the native-tie recommendation, the runner
needs a small explicit evaluation option plus focused review before any real
execution. Existing code does not silently switch semantics, and this document
does not claim that the recommended option has run. Main recovery continues to
use its existing Counter tie decision. Applying strict zero-tie rejection only
to the baselines would therefore reduce the number of evaluable external rows
and change exact-success counts, including rows whose native bit decision would
otherwise be correct. Retaining every fixed-manifest row makes that loss
visible, but does not make the two decision rules fair. The strict candidate's
rationale is that exact zero supplies no signed soft evidence; it should be
reported only as an unadopted sensitivity analysis. The primary recommendation
keeps each method's native bit decision and reports `tie_count`, avoiding that
additional baseline-only rejection.

Main embeds each bit across 46 latent times and repeated spatial coordinates;
the receiver reads 44 or 22 latent times with 30 frequency votes per bit/time.
RivaGAN repeats observations over frames, while VideoSeal propagates its native
message over its configured video steps and, in the effective-32 mode, across
eight channels. These carrier observations are dependent and never counted as
independent samples.

## Quality, operations, storage, and staged execution

The confirmation proposal has, per source, two 50-step trajectories, 200
conditional/unconditional transformer forwards, 25 active P0 payload-control
steps, two Wan native decodes, one framewise encode, and two framewise decodes.
Across eight confirmation sources this is 16 trajectories, eight framewise
encodes, and 16 framewise decodes. Four main arms plus VideoSeal effective-32
and RivaGAN produce 48 complete codec roundtrips. Receiver VAE encodes, edit
construction, key reads, and native extraction are additional rows computed
from the frozen manifest; 48 is not a total evaluation budget.

One full RGB8 raster is 88,965,120 bytes (about 84.844 MiB). PRE+POST for six
arms is about 1.06758 GB per source and 8.54 GB for eight sources, excluding
MP4, latents, native soft output, and optional materialized edits. Optional
VideoSeal native-K increases this to about 1.24551 GB per source and 9.96 GB
for eight. One Wan terminal FP32 latent is about 7.53664 MB; one framewise FP32
latent is about 7.41376 MB. With batch size eight, the confirmation set has 184
framewise encode batches and 368 framewise decode batches before receiver work.
These are deterministic operation/storage counts, not measured wall time,
VRAM, or a hardware recommendation.

The receiver manifest adds 36 framewise synchronization encodes and 160 blind
Wan reads per source. The RAW path has 32 unique `(arm, expanded index map)`
Wan encode identities; the remaining GLOBAL/PATH rows add at most 88, so the
manifest-derived upper bound is 120 Wan receiver encodes per source. For the
eight confirmation candidates this is 288 synchronization encodes, 1,280
reads, and at most 960 Wan receiver encodes. Pilot and confirmation counts are
reported separately, and these carrier reads are not independent samples.

Quality rows pair same-source/noise/codec POST artifacts: P0 vs OFF, P1 vs P0,
P1 vs OFF, M05 vs P1, M05 vs OFF, VideoSeal vs OFF, and RivaGAN vs OFF.
MSE/RMSE/PSNR are absolute pairwise metrics; PSNR differences are not an
additive decomposition. Exact identity is represented by `psnr_db=null` plus
`psnr_status=IDENTICAL_ZERO_MSE`, never non-finite JSON. P0 contains payload. VideoSeal/RivaGAN
native uint8 truncation and main raster `np.rint` quantization are disclosed
method differences, not silently rewritten to manufacture equality.

`real_cli` uses one phase per process:

1. `preflight`, `plan`, and `init` import no model library and execute no media.
2. `generate` holds the Wan transformer for one source and saves OFF/P0 terminals.
3. `decode` holds only the Wan VAE and saves OFF/P0 PRE RGB8.
4. `framewise` holds only the framewise VAE and saves the shared latent, P1, and M05.
5. Each `baseline-embed-*` phase loads one baseline and saves its complete PRE media.
6. `codec` records all six planned full-video transports and retained failures.
7. `quality` computes seven fixed POST pairs with bounded-memory array chunks.
8. Each `baseline-extract-*` phase saves all nine logical edit rows and full
   soft output, forcing lossless NPZ for VideoSeal tensors.
9. `receiver-sync` seals truth-free blind plans with the framewise VAE;
   `receiver-read` separately loads the Wan VAE and writes truth-free bits.
   Both persist completed and failed rows incrementally.
10. `evaluate` joins the fixed manifest with saved readouts and truth, retaining
    every missing or failed row and keeping pilot and confirmation summaries separate.

Wan and framewise configuration requires complete local snapshot directories.
The Wan snapshot path replaces the Hub ID passed to the existing repository
loader, while the original repository ID and immutable revision remain in the
receipt. The framewise loader calls `AutoencoderKL.from_pretrained` on its local
snapshot with `local_files_only=True`. Incomplete snapshots fail preflight or
load; neither path falls back to the network.

The current lightweight validation interpreter finds `ffmpeg`/`ffprobe` but
does not expose torch, NumPy, diffusers, transformers, safetensors, accelerate,
OmegaConf, VideoSeal, or RivaGAN. This is only the current test interpreter's
preflight result. No package was installed and no real source tree, checkpoint,
model, VAE, or codec was loaded in preparing this proposal.

## Decisions to make once, before real execution

| Decision | Recommendation | Cost or implication |
| --- | --- | --- |
| pilot/confirmation roster | two excluded pilots, then freeze the eight confirmation candidates after archive identity review | pilots cannot enter or repair confirmation denominator |
| 32-bit task message | adopt one explicit non-`OKOK` 32-bit list, proposed `A6D39C5E` | new interface evidence; historical arbitrary-message claim remains unavailable |
| VideoSeal source and weight | adopt official commit `870ca7f`, its 256-bit card, and the named `y_256b_img.pth` object; verify the supplied local file digest | source/card/API identity is resolved; weight adoption and local identity remain user decisions, and no weight was downloaded here |
| VideoSeal comparison | adopt effective-32 repeat-by-`j mod 32`; keep native-K optional and separate | rate 1/8, different redundancy/strength; native-K raises codec count to 56 |
| RivaGAN weight | explicitly accept or reject the community checkpoint | not official DAI-Lab weights |
| external main readout | use M05/K0 with GLOBAL for non-FULL GLOBAL, PATH for SINGLE_JUMP, and RAW FULL in a separate table | 64 non-FULL pairs per method plus eight FULL controls; 160 rows/source remain controls and ablations |
| zero-tie semantics | keep each baseline's native bit (`>0` VideoSeal, `>=0` RivaGAN) and report `tie_count` | current code exposes only the stricter unevaluable-tie candidate; selecting this recommendation requires a minimal explicit option and review before execution |
| RivaGAN sequence rule | adopt all-declared-frame equal soft mean with native `>=0` | missing/nonfinite/shape error fails whole row |
| thresholds/FPR | no threshold for exact recovery; design a larger independent calibration/evaluation roster before any presence/FPR claim | ten proposed sources cannot support low-FPR claims |
| compute and storage budget | approve the recorded calls, about 8.54 GB base raster storage, plus receiver/native outputs | GPU peak and wall time remain unknown until a separate authorized pilot |

No strength scan, alternative reducer scan, extra attack family, additional
baseline, or automatic run is part of this proposal.
