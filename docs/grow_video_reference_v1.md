# GROW-Video-Reference-V1: fixed image-to-Wan construction

This is one preselected video-source comparison of OFF,MULTI,LAST. It adapts
the successful image reference's published FFT-real/mean-MSE/conditional-clean
construction to native Wan Flow. It is not the old three-direction controller,
a formal-paper DCT reproduction or a temporal synchronizer. The retained real
run below establishes fixed-source saved-MP4 blind payload recovery. This main
integration performs CPU checks only; it does not rerun the real experiment.

## Fixed reference and construction

The image reference used source `fea8dc9c8c48802d31d34316d6e2f7df2dd8fe08`,
run `20260929T110323836911Z`, audited by the main session:2 PNGs/4 reads/8
post-read evaluations complete, GROW/correct-key recovered OKOK with0/32 errors.
That run used the explicitly selected SD2.1 community mirror; it was not a byte
identity proof against inaccessible official weights. Its original method
source remains GROW `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`.

The author helper named DCT executes `torch.fft.fft2(...,norm='ortho').real`;
this construction preserves that execution, including differentiating through
the entire real FFT and using masked mean MSE. The formal CVF paper's DCT,
eta100/16-bit/first-channel setup is a distinct reference. The code's image
setup is mean MSE,N=1600,eta200,32 bits,4 channels,conditional update thenCFG7.5.

| Fixed item | This video reference |
|---|---|
| Model | Wan-AI/Wan2.1-T2V-1.3B-Diffusers at `0fad780a534b6463e45facd96134c9f345acfa5b` |
| Source | Existing first copper-kettle prompt,seed2026092501; one source without selection |
| Geometry |181frames,320x512,8fps; native normalized latent[1,16,46,40,64] |
| Sampler | Original nonthresholded flow-prediction UniPC,50steps,zero final sigma; full live native history |
| Carrier | First4 channels,8 of32 bits/channel,FFT over the final2 spatial dimensions; h8..19,w12..31,K=240 |
| Time | Same32-bit payload repeated across all46 latent times; this is repetition,not synchronization |
| Target | Alpha0.5,messageOKOK,keywatermark; wrong-messageNOPE is evaluation-only |
| Step | Actual mask countN=4*46*240=44160; eta=200*N/1600=5520,computed from the tensor |
| Arms | OFF no local update; MULTI indices25..49; LAST only49. All start from identical noise and independent complete pristine scheduler copies |
| Precision | CUDA Transformer BF16 when supported,otherwiseFP16; CPU TransformerFP32. State,conditional/unconditional branches,local gradient,CFG and VAE areFP32 |
| Media | One H.264/libx264,CRF18,yuv420p save at8fps for every arm |

The common new loop converts both Transformer outputs toFP32 before any CFG.
OFF uses exactly that same arithmetic as MULTI/LAST; it is not required to be
bitwise equal to historical OFF,which used different arithmetic. Existing
shared-loader callers retain their old CUDA/BF16 defaults; optional device and
dtype arguments enable this new CPU/compatible-CUDA path. No GPU-name or exact
Python-version gate is introduced.

For each controlled step, compute conditional clean `x0_c=z-sigma*v_c` and
`L=mean((FFTreal(x0_c)[mask]-target[mask])**2)`. A fresh local leaf gives the
full gradient with respect to x0_c; no Transformer,previous-step or tail VJP is
performed. With `delta=-eta*gradient`, map back as `v_c'=v_c-delta/sigma`, then
form exactly one CFG `v=u+5*(v_c'-u)`. Do not substitute DDIM coefficients or
apply an extra7.5/5 correction. The chosen invariant is conditional
per-coefficient step scale,not equal post-CFG displacement with the image run.

The main session's independent
`diagnostics/video-image-reference-migration-20260929/gradient_scaling_check.json`
(in the parent project) establishes the local algebra: selected FFT-real
coefficient residual is multiplied by0.875 for both image1600/200 and
video44160/5520; unselected12 channels have zero local gradient. The band has
no conjugate pair/DC/Nyquist collision. ImageCFG7.5 gives local clean gain0.9375;
WanCFG5 gives0.625. These are local clean-coordinate identities,not native
scheduler displacement,RMS budgets or final decoded-media guarantees.

The input sigma at index49 is positive; sigma50 is the terminal zero.
Each arm calls native `scheduler.step` once per index,with continuous
model_outputs,timestep_list,last_sample and cursor. No reset,set_timesteps,
manual converted-x0 replacement or speculative tail occurs within an arm.
MULTI and LAST use the same local step; there is no total-energy budget
matching. Because MULTI includes49 and there is no FREE49 arm,this experiment
compares two fixed schemes and cannot isolate a non-last contribution.

## Save/read/truth boundary and fixed denominator

All3 arms save their raw normalized terminal tensors. A separate media process
loads only the FP32 VAE. For each arm it decodes once and saves the MP4 through
the existing common RGB8/H.264 adapter. Diagnostic reads observe raw terminal,
float-RGB reencoding,and RGB8 reencoding; the primary read is saved MP4 bytes
->FFmpeg RGB24->FP32 Wan VAE posterior mode->`(raw-mean)/std`->FFT-real bits.
Decode uses `normalized*std+mean`; the SD image scaling factor is not reused.
VAE cache clearing before/after calls is retained. Actual181x320x512 geometry
is checked; there is no implicit crop,pad,resample or frame repair.

Primary API `runtime.wan.grow_video_reference.read_mp4_payload` accepts only
path,public keys/protocol,frozen VAE and an optional observational call counter.
It receives no writer terminal,writer RGB,target,message or truth-bearing
configuration. The two keys share one observed MP4 and one VAE feature;
separate key reads expose the same feature hash and file hash. Diagnostic
latent/RGB APIs are separate and never substitute for the primary entrypoint.

Every layer reads with `watermark` and `watermark-wrong`. Their seed and actual
coordinate-to-bit assignment hashes must differ; string difference alone is
insufficient (`kramretaw` collides with `watermark`). Mask support stays the
same. Each bit gets240/8*46=1380 sign votes; zeros vote0 and exact ties follow
the first-vote Counter rule,as in the image reader. Public length is32 bits;
no truth-selected confidence or robust search occurs.

The fixed denominator is1 source,3 MP4s,4 dependent layers*3arms*2keys=24 raw
readouts,and48 evaluations after joining registeredOKOK/wrong-messageNOPE.
All rows exist before execution. Raw reads are persisted in
`blind_readouts.json` before truth joins; evaluation reopens that file and
checks its hash remains unchanged. UTF8 text is auxiliary only; raw32-bit
errors/BER/accuracy/exact recovery are primary. Failed/missing rows remain.
A failed second key retains a completed first key. Failed diagnostic layers
continue to the MP4 primary attempt. Worker failures preserve all rows and
completed work. `EXECUTION_COMPLETE` describes counts,not successful recovery.

## Runtime resources and call accounting

Generation and media use two serial fresh child processes of the current
`sys.executable`. The generation process loads no VAE; it finishes all arms,
saves CPU terminals,and exits before the media process loads the VAE. No
accelerate offload hooks,gradient graph spooling or alternating GPU model
residency are introduced. Each arm's media tensors are released before the
next arm; each normalized terminal is about7.19MiB. Model loading still needs
sufficient available resources; failures are records,not method results.

Planned completed calls are300 Transformer forwards (150 per conditioning
branch),150 native steps,26 local gradients,3 VAE decodes,9 VAE encodes,
3 MP4 saves,3 primary MP4 reads and24 bit reads. Actual attempted/completed
calls are logged by stage; readout slots are not mislabeled as VAE calls.
No VAE is encoded twice merely to supply two public keys.

After blind-read persistence and post-read evaluation,a separate quality stage
reads each saved MP4 once more:3 `quality_mp4_read` calls,distinct from primary
reads. It computes codec-readback RGB RMSE/PSNR relative to OFF by framewise
FP64 squared-error accumulation,without full-video FP64 copies or another VAE
call. No thresholds are introduced. A quality failure cannot overwrite raw
reads/evaluations or their execution outcome. Notebook display shows all
available saved MP4s for human quality inspection.

## Historical evidence and actual differences

| Earlier construction | Mechanism and retained real evidence |
|---|---|
| GROW-Video-Frequency,e04f48ed | True orthonormal DCT-II,channel0/16bits,64coefficients,half-squared-error sum,eta0.1,post-CFG clean local gradient,steps10..29.12 videos/48 layers complete; marked8 exact0 at all four layers; aggregate errors73/71/69/67 |
| GROW-Spatial-Carrier,a66db918 | Same spatial DCT/half-sum/channel0;steps30..49; LAST matched sum of native squared-response RMS.20 videos/80 layers complete; MULTI terminal3/8,float0/8,RGB8 2/8,MP4 0/8; LAST0/8 at each layer |
| GROW-Single-Step-Jacobian,001bc592 | Spatial DCT then23 temporal differences,half-sum,eta0.1,step30. JAC used a real current-input VJP and matched LOCAL native response energy.20 videos/80 layers complete,8/8 VJPs complete; LOCAL/JAC exact0/8 at each layer,MP4 errors70/128 and62/128 |
| Historical temporal-difference paired control | Partial positive evidence remains:run20260919T110131665623Z hard MP4 MULTI4/8,LAST7/8;soft3/8 and5/8. It is incorrect to call every historical GROW video construction unsuccessful |
| Three-direction continuous saved-tail candidate,39b3126 | RGB-DCT lift/time-bin basis,true-tail finite-difference J,weighted-hinge three-variable constrained solve plus native-budget/positive-group real acceptance.6/6 points,18/18 candidates,0/18 ACCEPT,16/18 loss worsening remain unchanged |

A2 checked the historical implementation against execution commits and original
records under the corresponding worktrees' `evidence/` directories. The older
FREQ/SPATIAL methods already differentiated the complete local loss and used
the same DCT for writing/reading; JAC really did use an input VJP. Those are not
new distinguishing claims. The present substantive change is FFT-real,
actual-mask mean scaling,multichannel32-bit construction,and conditional-clean
update before CFG with native Flow conversion. It also differs from the
three-direction proxy by directly using a local carrier gradient. The retained fixed-source result below establishes recovery for this construction;
it does not turn the earlier negatives into successes or establish generality.

## Local validation and notebook handoff

`tests/test_grow_video_reference.py` tests local algebra,exact50-step zero-eta
state/history equivalence to OFF,Flow/CFG placement,layout/voting,actual
synthetic MP4 readback with a fixture VAE using posterior mode and Wan
normalization,geometry rejection,per-key/layer failure retention,separate
quality failure,device/dtype selection,optional loader arguments and notebook
setup persistence. The CPU fixture VAE does not validate real VAE survival.
Test commands/results and package versions are saved in
`diagnostics/grow_video_reference_v1/local_validation.json`. No real-model
or GPU evidence is produced by these tests.

The notebook starts with the exact2-line Drive mount. It uses the current
Python,probes Torch/Torchvision together and repairs them only if that import
fails; a small candidate dependency-range file inherits the successful image
reference API ranges and adds Wan's ftfy/sentencepiece. Metadata pip-check
conflicts are recorded,while actual imports determine availability. There is
no venv,ensurepip,exact-version gate or hardware-name gate. Its first working
cell creates full3/24/48 setup rows before source/environment work. Run all
executes only this fixed case. Failed setup keeps receipts and the denominator.

The builder supports an explicit unpublished draft with SOURCE_SHA=None.
The committed notebook is bound to published immutable source. New bindings
are created only after publishing source; historical SHA links remain valid.
This integration does not run Colab, load weights or alter historical results.

## Retained real run and evidence boundary

Run 20260929T124706736859Z used source
[24c44353289909b10d08bf5e2be90c49ed0b9b0f](https://github.com/RICHAAARC/SC-SSTW/tree/24c44353289909b10d08bf5e2be90c49ed0b9b0f).
Its original [SHA-pinned notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/e2dc386dd0a816763b8bd4d6bfee7508d7d4e209/notebooks/grow_video_reference_v1_colab.ipynb)
remains a reproduction record. The method module, runtime reference adapter and
fixed experiment config imported into main are byte-identical to that executed
source; subsequent orchestration/preview repairs preserve its method.

| Arm | Terminal / float RGB / RGB8 / MP4 bit errors | MP4 PSNR relative to OFF |
|---|---|---|
| OFF | 21 / 21 / 18 / 19 | same reference |
| MULTI | 0 / 0 / 0 / 0 | 32.3714 dB |
| LAST | 0 / 0 / 0 / 0 | 36.1010 dB |

Three videos, 24 blind reads and 48 post-read evaluations were retained.
The audit reconciled source/config and artifact hashes, all message evaluations,
terminal bit reads and saved-MP4 quality. The compact
[evidence record](evidence/grow_video_reference_fixed_run.json) retains source
and result hashes, original Drive-relative location and all primary MP4 rows.
Large media and raw logs remain in the original run.

This supports real native MULTI control and full fixed-video blind repeated
payload recovery. It does not establish temporal alignment, variable clips,
time-dependent segment payload, calibrated rejection/FPR, robustness or
generalization. MULTI contains the successful LAST step, so non-last causal
contribution and MULTI superiority are unproven and are not prerequisites
for continuing the proposal. Visible texture changes were observed in sampled
frames; PSNR is diagnostic, not an imperceptibility pass.

The original notebook's preview exception happened after persisted execution
completed. The positional Video path and guarded preview are retained in the
imported builder. No experimental result is replaced by that repair.
