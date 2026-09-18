# Fixed real-crop observation diagnostic

Source run: `inversion_state_holdout_20260918T140243514638Z`, six existing
holdout OFF/A/B MP4 files. No new video generation. Decode the saved source
MP4, take 129 RGB frames beginning at 0, 16 and 17, save each as RGB lossless
`libx264rgb -crf 0 -pix_fmt rgb24` MP4 at 8 fps, then read that file and require
exact RGB equality to the selected decoded source frames. This avoids adding
another lossy CRF18 operation. Never pad, concatenate, resample or substitute
missing frames. A codec/readback mismatch is a retained failure.

18 fixed clips = six sources x three positions. Each receiver sees only its
opaque clip filename, public prompt/negative/CFG/schedule and 129-frame geometry.
It never receives the true crop start, source frame map, message or writer
noise. The source-to-clip association is an attacker manifest used only for
crop construction and the later oracle report. Source timing is not parsed
from a receiver filename.

Wan causal VAE length is 129=4*32+1, yielding [1,16,33,40,64]. Reuse posterior
mode/normalization and the existing 50-step Euler inverse. No 181-frame padding.
The loader's seed-zero dummy noise is discarded. Separate VAE and Transformer
lifetimes; no VAE decode or forward generation. Store normalized clip latents,
recovered noise and raw channel-0 coordinates without time-pad demodulation.
Raw slice summaries are not decoded state observations.

After the receiver has returned and released its model, oracle evaluation uses
the original public book and true start. Start 0 uses nominal index shift 0;
start 16 uses shift 4. Start 17 has fractional source phase 4.25: report both
nominal floor shift 4 and ceil shift 5 separately, without selecting a winner.
Local j>0 has nominal source frame group [start+4*j-3,start+4*j]; for start17
each group overlaps two original latent groups. This is an indexing hypothesis,
not a claim that the causal VAE or inverse recovered the same source latent.
Local slice 0 is a reset-context singleton and is never included in core scores.
The final local slice is marked as an endpoint; it still contains a full group.

For each oracle map retain all 33 local rows and 11 original core windows,
including absent and partial windows. Only four-of-four observed source indices
with nonzero q norm are valid for complete-window phase recovery. Partial q is
preserved but not counted as a complete state. Observer on/off uses the existing
state trajectories, fixed gain and missing cost 1; it is auxiliary oracle
analysis. Different maps have different missing support, so scores must not be
compared as if they had identical observation coverage. OFF has no true message.

24 oracle maps are repeated views of 18 clips from six videos, not 24 independent
experiments. Keep fixed denominators: 18 clip receivers, 594 local slices,
24 maps/264 core windows. Of these, 12 clips and 16 maps are marked; marked core
denominator is 176, with incomplete/missing windows reported separately. No
blind synchronization, crop-robust watermark, FPR or observer benefit claim.

Budget: 18 lossless MP4 saves, 18 VAE encodes, 1800 Transformer calls, 900
inverse updates; zero model generation and zero VAE decode. Runtime may fail
individual clips; all entries remain. Outputs and source archive use a fresh
`MyDrive/Video-WM/InversionCropObservation/inversion_crop_observation_<UTC>`.
Only CPU/fake/static checks are executed locally, not pretrained models.
