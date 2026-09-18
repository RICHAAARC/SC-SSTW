# Inversion state candidate: frozen protocol

This is a new real state-sequence experiment, not a re-scoring claim on the
previous repeated 16-bit messages. No model run has been performed locally.

Reuse `state_clock.trajectory(key, message)` unchanged for message 0/1, giving
11 four-phase states and 10 rotations. Reuse `initial_noise.codebook` coordinate
order and pads. In channel 0, latent slices 1–44 form fixed four-slice windows
[1,5), [5,9), ..., [41,45). Sorted coordinate parity encodes x/y respectively:
`abs(base) * pad * state[window, sorted_coordinate % 2]`. There are 1280
coordinates per axis per slice and 5120 per window. Channels 1–15 and boundary
slices 0 and 45 remain exactly unchanged.

Keep four content/seed cases and OFF/A/B, generation, 50-step UniPC, actual
MP4 save/read, VAE encode and 50-step Euler inverse identical to the baseline.
Known prompt and original latent-time coordinates remain assumptions. No
cropping, clock search, new model run, noise recalibration or fitted weights.

Readout first computes raw double-precision amplitude times pad, grouped by
coordinate parity, for all 46 slices. The 11 core windows average four slices;
two boundary observations are retained separately and never scored as states.
Persist raw q, norm, signs and zero counts. A window with norm <= 1e-12 is
invalid, kept with innovation cost 1 and prediction-only propagation.

For each public candidate trajectory, call the unchanged `state_clock.observe`
with update False/True and fixed gain 0.5. Each mode score is ONLY negative mean
innovation; no old matched-minus-weighted-innovation mixture. Differences <=
1e-12 are ties. All-invalid input has no message decision. Receiver functions
never take the true message; truth joins only in posthoc reporting.

Report separately raw four-phase window matches (88 marked windows, 176 state
components, 8 full trajectories), unique correct message decisions per mode
(8 marked videos), true-minus-wrong margins, and per-video on-minus-off margin
changes. Updated observer states are never substituted for raw observations.
OFF has candidate ranks, no true message or FPR claim. Missing and failed rows
retain 12 videos x 11 core windows, 2 boundaries and 46 slice slots.
The 22 state-component signs per video are not 22 independent payload bits:
there are only two message candidates, one shared initial phase, and ten
constrained rotations.

Budget remains 1200 generation Transformer calls / 600 forward scheduler steps,
12 VAE decodes / 12 encodes / 12 MP4 saves, 1200 inverse Transformer calls /
600 inverse updates. Outputs use a fresh `MyDrive/Video-WM/InversionState`
directory. Resources, public trajectories, provenance and failures persist.
No scientific PASS, low-FPR, crop robustness, or observer-gain claim follows
from engineering tests or a full-video message decision alone.
