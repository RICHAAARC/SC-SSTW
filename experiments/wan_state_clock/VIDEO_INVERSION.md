# Wan initial-noise inversion mechanism baseline v1

Independent local candidate from d5877c5, not a GROW method, PRC/SIGMark
reproduction, trajectory-writing method or formally secure watermark. Source
publication and a SHA-pinned user-run Colab notebook are now authorized. The
assistant does not execute pretrained models, GPU, Colab or Drive experiments.

Reuse Wan1.3B, 50-step native UniPC flow forward, BF16 completed CFG5 followed
by FP32 scheduler velocity,181 frames at320x512/8fps. Same two contents and two
seeds as GROW, OFF/A/B=12 videos. The manifest freezes the exact same16-bit A/B
values as GROW's published codebook; no runtime import/dependency on GROW.

Initial noise carrier: channel0 all40x64 coordinates in key-derived fixed
permutation, position modulo16 assigns160 repetitions/bit/time. An independent
key-derived pad(time,coordinate) supplies +/-1. Write abs(base_noise)*bit*pad;
all15 other channels stay unchanged. OFF is the unmodified base, A/B share
magnitudes and complementary bits. All46 latent slices repeat the same message.
The reader receives only keybook(order,pads), not candidate/true payloads.
It sums sign(recovered_noise)*pad into7360 votes/bit; individual zero and total
zero are explicit abstentions/erasures. Every slice and aggregate bits/BER/exact
match against A/B are reported after decoding, including OFF coincidences.
These votes are correlated, not7360 independent statistical samples. Fixed-key
sign constraints do not preserve an unconditional security guarantee or prove a
joint Gaussian distribution. No FPR threshold or confidence claim is supplied.
Full-space redundancy differs from GROW's184 votes: no strict carrier-fairness claim.

Receiver: actual saved CRF18 yuv420p MP4 -> read_mp4 -> deterministic VAE mode ->
normalized latent. Known prompt/negative/CFG5 are public conditioning; this is
not prompt-free. Native forward sigmas have51 nodes and timesteps50. Reverse
nodes r=flip(actual_sigmas), including0; left-node model times are
[0,t49,...,t1], not flip(timesteps) and not recomputed sigma*1000. UniPC's original
integer timestep rounding is retained. The extra clean endpoint explicitly uses
t=0. For50 steps, z_next=z+(r_next-r_current)*v(z,t_left). This one Euler
integration is approximate, never the exact inverse of the forward UniPC
history. Endpoint/model error, discretization, VAE and codec losses can compound.

Generation, media and inversion have separate model lifetimes. Generate three
arms, release Transformer, load VAE for three MP4s, release VAE, then reload
Transformer for inversion. The receiver loader uses independent seed0 and
immediately discards its random noise. It receives only MP4-derived latent,
public schedule and conditioning/model. Writer noise/terminal are saved separately;
noise correlation/RMSE is computed only after recovered noise is saved and the
receiver model released, never used to initialize/correct/select/stop inversion.

Fixed total plan:2400 Transformer forwards (1200 generation+1200 inversion),
600 native forward steps,600 inverse updates,12 VAE decodes,12 VAE encodes and
12 MP4 saves. Persist stage attempt/completed counts, time, GPU allocated/reserved
peaks and cumulative process RSS. Missing cases retain3 arms and46 slice rows.
Quality is same-case persisted MP4 versus OFF; report MSE/PSNR/temporal residual
metrics without invented tolerance. Execution completeness, bit recovery,
quality, OFF coincidences and posthoc diagnostics remain separate.

Code is independently implemented, not vendored from VideoShield/VideoMark/SIGMark.
Fixed source references from the main-task official-source review:
- [VideoShield](https://github.com/hurunyi/VideoShield/tree/a61efa73abb15d30f50ee535c3301cea7dec3075): sign/pad repetition idea.
- [VideoMark](https://github.com/KYRIE-LI11/VideoMark/tree/9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be): PRC/TMM as a later comparison, not implemented here.
- [SIGMark](https://github.com/JeremyZhao1998/SIGMark-release/tree/3713243f2e002cb21bbc3da896094d39b2b528c9): flow inversion reference for Hunyuan, not a Wan implementation.

PRC/GF/LDPC machinery, alignment, trajectory search and extra inverse detectors
are outside this first candidate. Main-task source review identified Hunyuan and
scheduler/runner assumptions that cannot be transferred to Wan without adaptation.

Fixed run entry: python -m experiments.wan_state_clock.video_inversion_run --output NEW_DIRECTORY
Use the independent MyDrive/Video-WM/VideoInversion/video_inversion_<UTC> root.
The delivered notebook runs this fixed entry directly and pins the published
source SHA. No real model/GPU result is produced by this delivery.
