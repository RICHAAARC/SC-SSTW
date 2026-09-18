# GROW one-pulse control transmission diagnostic

This new same-state diagnostic fixes dev_p0_s0 and zero-based index10. It does
not replay an old run or change the original GROW full experiment. The original
DCT target amplitude .5, eta .1, loss and payload are retained. No scans or media.

Run OFF steps0..9 once; persist latent, complete native UniPC history, schedule,
conditioning, source/runtime/model identity and the shared completed CFG velocity
at10. Clone that same state for OFF/A/B. Apply exactly one original local
predicted-clean pulse at10, then each clone performs one native scheduler step.
The corrected last_sample and model_outputs history are preserved separately for
each arm. Closing control afterward means u=0; it never resets history to OFF.

Phase one:22 Transformer forwards,13 native scheduler steps,2 local gradients.
Separately count2 CPU scalar native solver probes with zero/one velocity, retaining
history presence/order/config. They give independent K=-h/sigma and corrected
sample response, without Transformer calls or Euler substitution. Khat fitted
from actual D is descriptive only, not its own independent validation.

Numerical validity checks: finite tensors, identical cloned snapshots, cursor and
history structure; clean algebra, converted history and actual D versus K*u.
Fixed max-absolute tolerance is2e-5*max(1,reference maxabs)+2e-4*expected maxabs.
Report expected delta maxabs/RMS and tolerance ratio. Compatibility at the
rounding scale does not prove adequate transfer. Zero/weak/negative response,
loss or signed gain, and bit recovery never gate continuation.

If numerical validity holds for all3 arms, continue11..49 with no further control
using each arm's own original poststep history. This adds234 TF/117 native steps;
whole plan256 TF/130 native steps/2 local gradients plus2 scalar probes. No VAE,
MP4 or Transformer backward. Failures retain the3-arm and46-slice denominators.

Persist one-step D/Khat/residual and A-B contrast, clean/history differences;
same-index selected46x64 DCT coefficient differences and signed gains; and
next-clean delta decomposition delta_z-sigma*delta_v. Also measure actual
Q(z_arm)-Q(z_OFF) after model-input dtype cast: RMS, nonzero fraction and carrier
projection, using already available states and no extra model calls. Terminal
readout uses the original184 sign votes/bit with BER/erasures reporting afterward.
The local .81 loss ratio is(1-.1)^2 from orthogonal DCT, not a retention fraction.

Only this case's single-pulse transmission and later joint model/history response
are identified. No claim of full20-control failure causality, FPR, quality or
cross-case generalization. Model revision may be unresolved; save what the loader
actually exposes and conditioning tensors, do not invent a resolved revision.

User-run notebook writes only under
MyDrive/Video-WM/GROWControlTransfer/grow_control_transfer_<UTC>.
Source publication and notebook delivery are authorized; the assistant does not
run pretrained models, GPU, Colab or Drive experiments. CPU/Fake checks are
engineering evidence with the actual scheduler source/version reported.

Official solver reference: [diffusers v0.40.0 UniPC](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_unipc_multistep.py).
CPU checks may load this exact source into the existing0.39 package only for the
scheduler; that is not a full0.40 environment or a pretrained-model run.
