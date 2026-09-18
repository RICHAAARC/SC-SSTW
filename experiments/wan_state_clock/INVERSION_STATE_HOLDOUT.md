# Fixed independent state-candidate validation

Reuse the already locked `configs/velocity_calibration.json` holdout roster:

| Case | New content | Seed |
|---|---|---:|
| holdout_p0_s0 | white swan swimming slowly across a quiet pond | 20261001 |
| holdout_p1_s0 | blue cable car moving slowly across a mountain valley | 20261002 |

Full prompts are copied unchanged to `configs/inversion_state_holdout.json`.
Neither prompt nor seed overlaps the state development roster. No outcome-based
selection, retry substitution, new strength, or per-video parameter adjustment.
This reuses a historically predeclared roster; it does not claim those contents
have never appeared in any other project experiment.

State encoding, coordinate carrier, key, messages, four-slice core windows,
boundary handling, raw amplitude readout, observer gain, negative-innovation
score and invalid/tie tolerance are frozen to the existing state candidate.
The holdout manifest is checked against its development manifest. Existing
generation, MP4 save/read, VAE and approximate inverse implementations are reused.

Run `python -m experiments.wan_state_clock.inversion_state_holdout_run --output
NEW_DIRECTORY`, or Run all in the holdout notebook. Two cases each retain
OFF/A/B: six videos, four marked sequences, two OFF references. Marked scores
use 44 core windows and 88 state components; these are correlated state signs,
not independent message bits. Across all six videos retain 66 core windows,
12 boundary observations and 276 slice observations, including missing rows.

Report raw window/component/full-trajectory recovery, separate observer on/off
unique-message decisions and margins, per-video decision/margin changes, and
same-case OFF MP4 quality. OFF has candidate ranks but no true label, detection
rate, or low-FPR claim. Observed errors and missing denominators remain distinct.
Raw observations must not be replaced by updated observer states. No scientific
PASS or new quality threshold is introduced.

Fixed budget: generation 600 Transformer / 300 scheduler steps; media six
decodes / six MP4 saves / six encodes; inverse 600 Transformer / 300 updates.
No additional latent experiments or observer tuning. Models retain separate
generation/media/receiver lifetimes. Use a new
`MyDrive/Video-WM/InversionStateHoldout/inversion_state_holdout_<UTC>` directory.

This preparation includes local CPU/fake and notebook static validation only;
the assistant does not run the pretrained model or Colab experiment.

Development review found marked-versus-OFF content/composition changes, including
dev_p1_s1 A with MP4 PSNR about 10.41 dB. The frozen holdout tests transfer to new
contents and seeds; preparing it does not certify invisibility or acceptable
quality. No parameters are adjusted in response to that review.
