# Video-Inversion original run records

Downloaded read-only from Google Drive on 2026-09-20 UTC. This archive contains 25 original JSON/log files, not reconstructed summaries. `manifest.json` records each Drive ID/URL, run, immutable source commit, download time, expected and actual byte size, and SHA-256. All JSON files parse and sizes match Drive metadata. Raw files under `raw/` are local evidence only and must not enter Git; only this index and manifest are intended for versioning. No model, video scoring, method change, or rerun occurred during archiving.

| Run | Original Drive folder | Execution source | Files |
|---|---|---|---:|
| inversion_paper_20260919T110148658074Z | [Drive](https://drive.google.com/drive/folders/1vZXYUCiTJHcocCo5eJBYGWHeQYbeE8wA) | `7fa72efea8a6ef04c2e62748e9fcccaeb5549d78` | 16 |
| inversion_blind_sync_20260919T023520531625Z | [Drive](https://drive.google.com/drive/folders/1ldw0md5bfP15CgRttt1SKO1GsMWzHGrT) | `9669f794d7d474a7bc51893310fbd889cd2238ff` | 4 |
| inversion_crop_observation_20260918T171553283563Z | [Drive](https://drive.google.com/drive/folders/1AGlKgcoIjyv47NEZbRmLEyKVRjSaOWGJ) | `8ad796e1bfa5d3f5dc697c3f14f62df0f24f0d29` | 5 |

Each `raw/<run>/result.json` is the complete original root result, retaining nested cases/views/candidates/failures. Paper additionally includes original threshold/protocol, four case configs and schedules, four child logs and launcher log. Blind includes protocol, search-phase completion record and launcher log. Crop includes manifest, two child logs and launcher log. The downloaded threshold SHA matches the paper root and both evaluation case references.

No requested result file was unavailable. Large media, latent tensors, codebooks and source ZIPs were intentionally not downloaded. Separate child result/readout copies were not duplicated because the complete root result embeds their evidence. No user-pasted summary or existing-local copy is labelled as Drive raw data.

## Evidence limits

Paper: 12 sources / 36 views include 2 calibration OFF sources and 10 evaluation sources / 30 views. STATE/search14 and STATIC/search14 each accepted and correctly attributed all three views of 4/4 matched marked sources; both are positive limited-protocol baselines. STATE/shift0 achieved 2/4 and STATIC/shift0 3/4 all-view correct source decisions. Each channel observed zero false acceptance among only 2 evaluation OFF sources, insufficient for a low-FPR claim. Crops of one source are correlated. STATIC's different code distance prevents a clean causal claim that dynamic state evolution is beneficial. This is known-prompt initial-noise watermark inversion, not trajectory watermarking; full181 scoring uses fixed-prefix evidence after full inversion.

Blind search: 18 clips / 504 fixed candidates, 12/12 marked messages and 8/8 nominal aligned-shift/message pairs. OFF is ranking-only, not calibrated detection. Crop17 provides no exact within-latent RGB phase estimate. Crop observation's nominal floor/ceil and oracle-window evidence are separate diagnostics and must not be selected posthoc as blind synchronization successes. These records do not establish large payload capacity, general temporal-attack robustness, or perceptual quality acceptance.
