# Bidirectional-Cross-Confirm-V1 CPU development diagnosis

Status: **PASS**. Replayed 24/24 existing views and 96/96 stored phase tensors on CPU. This did not run the model, VAE, GPU, media generation, or a new experiment.

The new A-locate/B-confirm direction matched the frozen existing split receiver across selection, held-out confirmation, C1/C2 scores, all three fixed references, and alignment reporting for 24/24 views. The full 4,284-path joint observation+codebook partition-role swap check was PASS: directions exchanged and the arithmetic mean was preserved.

Using only the old two OFF calibration sources for same-batch diagnosis (not the new four-source formal program):

| Receiver | eval OFF detected | eval marked detected |
|---|---:|---:|
| existing single-direction C2 | 0/6 | 4/12 |
| bidirectional C2 mean | 0/6 | 6/12 |

For p2 SPEED5_4, the existing single-direction C2 has fixed-reference pass but blind failure in both marked arms. The bidirectional mean passes both blind and fixed-reference thresholds in both arms on this old development batch; this is the only source of the marked-count increase from 4/12 to 6/12.

For p3, the existing single-direction C2 remains below threshold even on all six supplied reference paths. The bidirectional fixed-reference mean is above its own same-batch threshold in all six, while all six blind scores remain below threshold. This separates the old single-direction confirmation insufficiency from a remaining bidirectional blind-localization gap; it does not prove generalization. No physical-signal-destruction or MP4-causation claim is made.

All detailed per-view scores, thresholds, gaps, margins, equivalence checks, timings, and retained failures are in `cpu_result.json`. Evidence ceiling: existing same-batch CPU development diagnosis, not independent validation, real GPU resource evidence, or a scientific PASS.
