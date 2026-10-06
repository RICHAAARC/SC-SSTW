# Current mechanisms and evidence limits

Updated 2026-10-07. Main contains a bounded hybrid: generation-trajectory frequency payload, terminal framewise-VAE tubelet synchronization, blind offset/phase reception, and a fixed single-jump receiver diagnostic. Main retains GROW's complete original native sampler/decoder/media path unchanged.

| Mechanism | Supported evidence at its original version | Limits |
|---|---|---|
| GROW | Native MULTI25–49 and LAST fixed-source payload recovery through real VAE/MP4; 3 videos / 24 reads / 48 evaluations | Same 32 bits repeat in time; MULTI superiority/necessity unproven |
| M05 terminal sync | Target .5 with original full tubelet directions and float32 writing, real framewise/MP4 survival; source e87af1c… | Terminal sync is not trajectory sync; latent local support is not independent RGB support |
| Blind offset/phase | Stage2 CROP/SHORT final errors 13/14→0; 1A p2/p3 four windows 11/1/11/1→0 | Fixed overlapping development windows; offset, phase, final bits and time-bit margin remain separate metrics |
| Fixed new-source check | 1B four windows correctly localized and aligned payload read; final errors 0→0, weakest margin rises | Not final-bit correction or proof alignment is necessary; one source does not establish population generalization |
| Fixed internal deletion | Correct-key C=H0 b2 and D=H1 b2 k88, each 177/177 correspondence; D H0 only 88/177. RAW/GLOBAL/PATH final errors all 0; D minimum margin .4/.5696969697/.6272727273 | Fixed developed source and one deletion, not BER benefit, existence detection, FPR or unknown-edit robustness |

The reports and immutable source identities are linked in [the evidence index](mechanism_evidence_index.md). These are separate original-version executions, not one newly executed main version. This integration has CPU arithmetic/interface equivalence and portable engineering checks; it does not silently inherit a new combined real-run claim.

Payload uses keyed FFT coordinates over the full 40×64 latent spatial extent, not spatial patch-local payload. GROW keeps its full 46-time reader. The stable temporal receiver uses unchanged FFT2 real/ortho, t1:R+1, channels0..3, time-major bit::8 and Counter first-encounter ties; strict zero maps to a negative vote. The inherited `video_trajectory_payload_gt_v1` runtime filename now exposes only this length adapter, not GT or temporal smoothing methods.

Sync uses 4-frame × 4×4 spatial patches and original age slices (including the final source180 normalization). The H0/H1 receiver uses 885 frame/displacement cells and 709 paths per read. H1 wins are not detections; adjacent k paths differ at one frame. Previous-frame insertion restores a nominal index grid, not missing RGB. C/D truth is allowed in preparation, but never enters the blind selector/correction. Oracle maps are parsed after blind payload sealing; message semantics only after oracle sealing. Source/config byte hashes may be read early for identity. Complete-map aliases and failed cache entries remain recorded without retry.

Presence/no-watermark, wrong-key rejection, uncertainty/erasure, broader dynamic edits and population FPR remain unfinished. Spatial-local payload and state-dependent segment encoding are gaps relative to original goals, not automatically adopted new work. Same-version joint mechanisms, quality, edit cost and applicability still need their explicitly necessary evidence. Mechanism closure is reported to the user before discussing paper experiments; paper scripts/test sets/chart templates/large comparisons are not being prepared.

Main core has no runtime/experiment/governance imports; runtime has no experiment imports. Three explicit-input CLIs work without notebooks/scripts. Source provenance uses the actual runtime manifest in a no-Git directory and only inspects Git if root/.git exists, never an unrelated parent. Config and manifest identities are evidence, not model inputs or method thresholds.
