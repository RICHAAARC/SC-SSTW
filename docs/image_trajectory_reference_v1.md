# Image-Trajectory-Reference-V1: fixed published-code candidate

This candidate chooses the author's **GROW released code** at
`6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`. Its writer, loss, transform,
receiver, scheduler and default image settings remain the upstream execution.
It is a one-image-source reference, not a paper DCT reproduction or a new
video method. No real model has been loaded or run for this delivery.

## Source and asset decision

`diagnostics/image_trajectory_reference_v1/official_source_manifest.json`
records all 21 GROW and 63 Guidance files, each SHA256 and Git blob SHA1,
matched against complete, nontruncated GitHub trees at the fixed commits.
The original six handoff files plus `grow/utils.py` match (7/7); the original
handoff is unchanged. Source archives live only in the stated `/tmp` cache.
They are not vendored or patched. The loader verifies the complete GROW file
manifest before importing the original `config.py`, `watermark.py`, `codec.py`
and their original dependencies. A namespace wrapper skips only the package
facade that imports unused AttackSuite/torchvision; no method body is copied.

GROW has a complete writer/receiver, packaging and declared dependencies, and
needs no separate learned detector or whitener. Guidance's actual demo needs
VideoSeal plus missing whitener samples and unresolved asset instructions.
GROW is therefore the more complete source-level baseline. This does **not**
mean all assets were accessible at that delivery. The historical anonymous401
probe remains in `asset_probe.json`. Following the user's explicit adoption of
a third-party source, the current runner uses
[sd2-community/stable-diffusion-2-1-base](https://huggingface.co/sd2-community/stable-diffusion-2-1-base)
at `4e63672c03103b6c636b8fb4119ba982469b2955`.
`community_model_receipt.json` records public/no-gating metadata, all required
Diffusers files, downloaded configuration hashes and successful HEAD requests
for the three full-precision safetensors components. This is a 512-base,
epsilon-prediction model source; the original author methods still use DDIM.
No claim of independently proving byte equality with unavailable official
weights is made. No weights were downloaded or run by the agent.

The runtime loads this recorded revision directly, without a live model-info
API prerequisite or mandatory HF token. It chooses CUDA when available and CPU
otherwise. Actual component download/load errors are retained with all fixed
rows; environment differences themselves do not count as method failures.

## GROW: formal paper versus fixed released execution

Primary paper: [formal CVPR 2026 proceedings PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.pdf),
SHA256 `484969c867bf2b879aa675ca1eec6de4093d10f7225547c301d0ee32bede4fce`.
Root and A2 independently verified formal CVF pages35980-35982; the author review copy was also checked separately. The separately fetched
author-hosted Submission #8345 review copy has a different identity; its hash
and source are preserved in `paper_source_receipt.json`, not substituted for
the formal paper.

| Item | Formal paper | Released code used here |
|---|---|---|
| Transform | p35980 section 4.2.1: DCT | `grow/utils.py:36-48`: `fft2(...,norm='ortho').real`; inverse helper is real IFFT, not DCT-II |
| Objective | p35981 Eq5 prose says MSE; displayed formula is squared L2 without 1/N | `watermark.py:275`: `F.mse_loss` default mean over 1600 selected values |
| Gradient path | Eq6 gradient in predicted clean latent | `watermark.py:258-285`: conditional x0 estimate; local gradient requested with respect to x0; UNet under no_grad; no explicit per-step detach; x0 minus eta*g then conditional epsilon |
| CFG placement | Eq7 guided conditional prediction then CFG | `watermark.py:286-290`: same ordering, CFG 7.5 |
| Timing | p35982 Algorithm1 uses descending t=T..1 and `t > T*r_start` | forward enumerate step index >=25 of50, last25 updates; notation/execution difference retained without assuming equivalence |
| Payload/strength | section5.2:16 bits, first channel, alpha0.5, eta100 | dataclass defaults: UTF8 `OKOK`=32 bits, channels0..3, alpha0.5, eta200 |
| Model/runtime | section5.2: PyTorch1.13/T4,50 steps | `GrowConfig`/constructor: original SD2.1-base, DDIM,FP32,512px,50 steps; requirements only lower bounds (torch>=2,diffusers>=0.21), no exact lock |
| Blind input | EXTRACT(image,key) pseudocode does not spell out length | `_read_bits` also needs public bit length32; it does not need message content |
| Readout | signs/votes in frequency representation | VAE encode mean times scaling, FFT real, keyed coordinate sign votes; `Counter` tie order, UTF8 decode ignores bad bytes |

Fixed code links:
[defaults](https://github.com/luopengchen/GROW/blob/6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870/GROW/grow/config.py),
[writer and reader](https://github.com/luopengchen/GROW/blob/6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870/GROW/grow/watermark.py),
[actual transform](https://github.com/luopengchen/GROW/blob/6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870/GROW/grow/utils.py).
`run_demo.py:52-64` constructs the dataclass; it does not load the YAML.

## Guidance: actual demo chain, not the similarly named class

Pinned source is [Guidance d1f8e489](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/tree/d1f8e48946f830087baf32356e5ce0eeec68e27e).
Paths below are relative to `guidance-watermarking-for-diffusion-models/`.
Primary paper: [Guidance sections4.1-4.5](https://arxiv.org/html/2509.22126).
The paper itself uses an identity backward-Jacobian approximation and clipped
gradient divided by max(eta,norm); it is not a claim of full tail backpropagation.

| Execution item | Source-grounded finding |
|---|---|
| Entry/class | `scripts/{sd2,sana,flux}.sh` -> `guidance_benchmark.py:23-61` -> `configs/watermarker/prob-guidance.yaml` -> factory `watermarking/__init__.py:13-15,52-54` -> `ProbGuidanceWatermarker`, not Norm |
| Loss | config selects PCGrad/cossim. `watermarking/losses.py:76-89` returns cosine itself, not 1-cos or log. Keep distinct from paper Eq6 1-cos and Eq9 gradient of log loss; do not repair or assert equivalence |
| Gradient | `watermark.py:521-576`: remaining-tail backward with ones gives dx; separate image loss and VAE VJP gives dy; final elementwise dx*dy |
| Denoiser boundary | `watermarking/modified_diffusers/stable_diffusion.py:358-367,638-653`, `watermarking/modified_diffusers/sana.py:364-385,631-641`, `watermarking/modified_diffusers/flux.py:385-414,758-776` wrap denoisers in no_grad. Scheduler sample dependence remains; this is not the full denoiser tail Jacobian VJP |
| Update | original gradient L2 norm is measured before top10% amplitude clipping, then clipped vector divided by that original norm with cap; CFG precedes injection `eps -= wm_scale*sqrt(1-alpha)*grad` |
| Fixed configs | SD2 Euler FP16 scale250/top.1/cap.3; Sana FlowEuler modelFP16 latentFP32 scale600/.1/.5; FluxFP16 scale500/.1/.5, scheduler from model config |
| Selection | `watermark.py:578-581,470-479`: minimum message-bit errors selects best_latents, decoded at finish; not the same as GROW's unconditional final latent |
| Assets | `detector/__init__.py:15-34`: external `videoseal_y_256b_img_dec.pt`, placeholder WHITENER_PATH and missing samples.pt; source has no setup/pyproject despite install instructions |
| Payload/receiver | `configs/detector/videoseal_w.yaml`: M=256. `guidance_benchmark.py:30-37` uses CLI key or dataset generation; `benchmarks/datasets.py:21-23,149-159` samples torch.rand(M)>.5 and caches by index. Decode/logit signs need no message content; zero-bit cosine/p-value reporting needs a preregistered key and is not unconditional blind payload detection |
| Blind API fault | `detector.py:66-82` forward(key=None) uses unassigned pval_0bit (A3 CPU AST reproduction). Lower-level preprocess/decode_message/process_message can read without truth if assets exist; adding truth to avoid the exception is not a blind read |
| Save boundary | benchmark scores memory first; tests save PNG later. That score is not saved-file readback evidence |

## Why neither existing video path is this image reference

| Mechanism | Historical GROW-Video-Frequency adaptation | Current three-direction saved-tail candidate |
|---|---|---|
| Carrier/observation | true orthonormal DCT-II in Wan channel0,64 coefficients,16 bits,4 repeats per frame across46 latent times | RGB-DCT thirty-group q; deterministic RGB carrier/VAE lift split into three latent time bins [1,15),[15,30),[30,45), normalized on support |
| Objective | one-half sum of squared masked residuals over46*64, no mean reduction | mean weighted squared hinge `(1e-4-q)^2_+`, weight4 for baseline-positive groups else1, plus1e-4 times coefficient squared norm |
| Direction/gradient | detached post-CFG clean estimate z-sigma*v, local DCT gradient; no transformer backward | baseline plus3 real native-scheduler tail rollouts, decode terminal q; J column=(probe_q-base_q)/measured immediate support RMS; no full gradient |
| Update/constraints | steps10..29, u=-.1*g, velocity v-u/sigma, native UniPC | nonnegative3 coefficients, sum <=remaining R*=0.042943312697648145; deterministic continuous convex solver with global surrogate objective-gap certificate; unavailable directions fixed0 |
| Real decision | historical adaptation-specific controls, not official GROW | scale1,.5,.25 true tails; native RMS budget and peak<=.25; no positive-group loss; true hinge improvement>=1e-8. Surrogate improvement alone never accepts |
| State/denominator | video settings181frames,320x512,CFG5,Wan1.3B | fixed two sources T46/47/48 each restored independently with full saved scheduler history; no sequential chain inferred; sequential MULTI must recompute q/J after accepted state |

Historical implementation is
`worktrees/GROW-Video-Frequency/main/tube_state/grow_frequency.py` and its
`experiments/wan_state_clock/GROW_VIDEO_FREQUENCY.md`; it explicitly calls
itself an adaptation. Current three-direction source is fixed commit
`39b312669ff45de08617cae439f4d07a7ca6c37d` in
`worktrees/Video-Trajectory-Blind-Validation-V1`: entry
`experiments/wan_state_clock/video_trajectory_blind_validation_run.py`, shared
`rgb_dct_structured_terminal_feedback_run.py:476-771`, pure
`main/tube_state/rgb_dct_structured_feedback.py`,
`rgb_dct_continuous_solver.py`. These are comparison references only and are
not imported. The recorded prior result remains **6/6 points,18/18 candidates,
0/18 ACCEPT,16/18 loss worsening** (`diagnostics/six-point-audit-20260929/audit.md`
in the parent project). It is a negative for that tested construction, not
proof against the author's image mechanism. No old video bytes are changed.

## Frozen minimum case and independent save/read/evaluate chain

- One prompt: `a sharp wildlife photo of an owl with feather details, perched on an old tree branch`; seed42.
- Two arms call original `generate_normal` / `generate_with_watermark`; all dataclass defaults listed in `config.json` are checked against upstream. SD2.1-base community source with DDIM/FP32; original model_id remains in the author-default record, actual asset identity is separately recorded.
- Save each output as512x512 PNG and discard the in-memory result. OFF and GROW each have correct/wrong-key independent `_read_bits` calls:4 readouts.
- Correct key `watermark` has seed974. Wrong `watermark-wrong` must differ in both seed and coordinate-to-bit assignment hash. A different string can collide (`kramretaw` does). The entire band mask is the same; key changes the bit assignment, not mask support.
- Each reader gets only image, original pipeline, key and public geometry/32-bit length through `PublicReaderConfig`, which has no message field. It calls the actual official reader, not `extract_with_confidence` or robust/truth-selected search.
- Persist all4 raw32-bit rows in `blind_readouts.json` before evaluation. Reopen that JSON in a separate evaluation stage, then join `OKOK`/`NOPE`:8 rows. The blind-file hash must remain unchanged.
- Primary metrics are raw bit errors/32,BER,bit accuracy,exact32bit. UTF8 string is diagnostic only because upstream silently discards invalid byte sequences. OFF/wrongkey/wrongmessage are controls, not population FPR estimates.
- Paired saved-PNG RMSE/PSNR and displayed saved images support quality inspection. No invented quality or significance threshold. `FIXED_REFERENCE_COMPLETE` means all reads finished, not that recovery succeeded.

The original demo saves a PNG then evaluates its in-memory image. The explicit
reopen-and-independent-reader boundary above is an evaluation-envelope change,
not an undisclosed modification to the upstream writer or reader.

`result.json` starts with all2 images/4 reads/8 evaluations before source/model
work; retains every failure/missing row; records model asset revision, config,
source file hashes, seed, call ledgers and paths. Observational forward hooks
count attempted/completed UNet,VAE encoder,VAE decoder calls by phase without
extra model evaluations. `model_loaded` and `model_calls_executed` are separate.
An abrupt worker termination may leave RUNNING/PENDING rows and a nonzero exit
receipt; those are incomplete evidence, never successful method outcomes.

## Candidate runtime, notebook and local evidence

The current notebook uses the active Python interpreter and a venv with
`--system-site-packages`, reusing Colab's working Torch/Torchvision pair.
Only a failed actual Torch/Torchvision import triggers installation repair.
`requirements-colab.txt` specifies compatible library API ranges, not exact
Python, Torch or CUDA versions. `pip check` is recorded and does not stop the
run because of unrelated base-environment package conflicts. Actual imports
and model/component operations determine compatibility. A CPU runtime is
allowed with a speed note; there is no Python-version or GPU-name gate.
Actual installed versions are saved with the results.

Historical environment probes and v1/v2/v3 snapshots remain unchanged as
records of the older candidate. The community-source update was checked with
CPU fake components and the unchanged original GROW functions: 50+50 UNet
calls, four saved-PNG VAE reads and two decodes. Component failure tests retain
all 2/4/8 rows, and loading tests prove both CPU and CUDA selection, one mirror
revision for all components, and no preliminary Hub model-info API call.
Notebook environment-cell tests exercise Python3.13, working/broken Torch
imports and a nonzero pip-check report without executing installs or models.
Full Colab installation and real-model/GPU execution remain user-run.

For local reproduction, set `GROW_REFERENCE_SOURCE_ROOT` to the verified GROW
archive root and run `python -m pytest tests/test_image_trajectory_reference.py -q`.
Source-dependent tests skip if that cache is absent; skipped cases are never
counted as successful upstream validation.

`notebooks/image_trajectory_reference_v1_colab.ipynb` retains the exact two-line
Drive mount in cell zero, a child-process run and fixed 2/4/8 setup records.
The final cell reopens OFF/GROW PNGs for display. Run all executes the fixed
case without parameter modes or model-type selection.

Publication uses the already-authorized dedicated development branch. Publish
source first, verify its immutable SHA, build the bound notebook with
`scripts/build_image_trajectory_reference_notebook.py --source-sha <verified SHA>`,
then publish that notebook and deliver its fixed Colab link. A builder invoked
without a SHA still produces an explicitly unbound development draft. No
real-model success, video migration or blind temporal detection is inferred
from this delivery.
