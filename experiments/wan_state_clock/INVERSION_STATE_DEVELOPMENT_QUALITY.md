# Development video quality audit, 2026-09-18

Run: `inversion_state_20260918T080248089836Z`, source `d85894b7ab001a5a4b40782b5aa5788aa50a6e57`.
Drive results: https://drive.google.com/drive/folders/1vhIrNgEVRNINtmxoq-dMGgymyf7m6T-y

Downloaded all 12 existing MP4s via the Drive connector; ffprobe confirms each
has 181 frames, 512x320 pixels, 8 fps, duration 22.625 s. No generation or GPU
experiment was run. Actual visual inspection used 84 representative frames:
frames 0, 30, 60, 90, 120, 150, 180 from each MP4, grouped OFF/A/B per case.
This was contact-sheet inspection, not full-motion playback or a human study;
short flicker between sampled frames cannot be ruled out.
Local evidence: `D:/CodexData/.codex/inversion-quality-20260918/` contains the
12 videos, four contact sheets, downloaded result JSON and quality_inventory.json
with SHA256 and per-video metadata. Download hashes are local provenance, not
comparison to a generation-time video checksum (the runner did not store one).

## Visual findings

- dev_p0_s0: OFF shows a small distant complete red sailboat. A instead places a
  tiny red target near the lower frame boundary; B retains a clearer distant
  sailboat but with changed appearance/location. The A content change matters
  despite its 26.04 dB PSNR because most pixels are smooth lake/background.
- dev_p0_s1: A/B show changes in sailboat size/appearance and horizon relative to
  OFF. The sampled later frames remain broadly continuous.
- dev_p1_s0: A/B change tram appearance/scale and tree canopy relative to OFF;
  later sampled scene composition is broadly stable.
- dev_p1_s1: A has clearly sparser tree crowns, more visible sky and different
  road framing than OFF. This supports substantial content drift as a source
  of its 10.41 dB PSNR. B is closer in composition to OFF but is not identical.
- First-frame tree/canopy striping is visible in the tram cases including OFF.
  It cannot all be attributed to watermarking. The sampled later frames do not
  show cuts, but no full-motion flicker or motion-quality clearance is claimed.

## All-sample quantitative records

Reference is the same-case OFF saved MP4. OFF PSNR is null because its error is
zero, not because it is a missing sample. Temporal ratio is temporal-difference
energy relative to OFF, not a calibrated perceptual quality metric.

| Case | Arm | PSNR dB | Temporal energy ratio |
|---|---|---:|---:|
| dev_p0_s0 | A | 26.0428 | 0.6719 |
| dev_p0_s0 | B | 28.3511 | 0.6540 |
| dev_p0_s0 | OFF | self / null | 1.0000 |
| dev_p0_s1 | A | 19.9013 | 0.7545 |
| dev_p0_s1 | B | 23.6623 | 0.5943 |
| dev_p0_s1 | OFF | self / null | 1.0000 |
| dev_p1_s0 | A | 18.3039 | 1.5616 |
| dev_p1_s0 | B | 18.0644 | 1.1008 |
| dev_p1_s0 | OFF | self / null | 1.0000 |
| dev_p1_s1 | A | 10.4067 | 2.8383 |
| dev_p1_s1 | B | 20.3873 | 1.0500 |
| dev_p1_s1 | OFF | self / null | 1.0000 |

## Decision and claim boundary

No missing quality rows and no fixed quality threshold were introduced after
seeing the data. State recovery succeeded on development data, but visual
invisibility/content preservation has NOT been established. In particular,
dev_p0_s0/A and dev_p1_s1/A retain visible content deviations.
The fixed six-video independent run is a generalization and quality
characterization experiment, not a declaration that development quality passed.
Keep encoding, carrier, observer and readout unchanged; do not lower strength or
choose a new carrier to conceal these observations. Report all six videos and
inspect their quality before any quality acceptance claim. Low FPR, unknown-time
synchronization and high capacity remain outside this experiment.
