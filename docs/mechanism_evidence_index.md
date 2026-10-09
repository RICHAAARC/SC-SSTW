# Original-version mechanism evidence

These records retain their original source and run scope. The local audit paths identify retained evidence; the compact index includes SHA256. No new main real-model run is claimed.

| Mechanism | Immutable source | Retained local audit |
|---|---|---|
| M05 | [e87af1c55e](https://github.com/RICHAAARC/SC-SSTW/tree/e87af1c55e102b8f5cc7d9caff18ae56d0463aac) | `/home/richar/projects/Video-WM/diagnostics/trajectory-framewise-m05-real-run-audit-20261005/report.md` |
| Stage2 | [079c766ac9](https://github.com/RICHAAARC/SC-SSTW/tree/079c766ac910961492c5e50c8386adbf3e2eb55f) | `/home/richar/projects/Video-WM/diagnostics/trajectory-receiver-estimated-align-stage2-real-run-audit-20261006/report.md` |
| 1A | [f5e4725724](https://github.com/RICHAAARC/SC-SSTW/tree/f5e4725724a88fd4e4eafa516688873c50577b91) | `/home/richar/projects/Video-WM/diagnostics/trajectory-receiver-phase23-milestone1a-real-run-audit-20261006/report.md` |
| 1B | [b30583f5a8](https://github.com/RICHAAARC/SC-SSTW/tree/b30583f5a8965847a01b29b7ef22ea1956673c00) | `/home/richar/projects/Video-WM/diagnostics/trajectory-receiver-independent-source-milestone1b-real-run-audit-20261006/report.md` |
| fixed_single_deletion | [33f1671f41](https://github.com/RICHAAARC/SC-SSTW/tree/33f1671f417069c791db5d6e3366f2fc497fe09d) | `/home/richar/projects/Video-WM/diagnostics/trajectory-internal-single-deletion-real-run-audit-20261006/audit.json` |
| conditional_joint_v1 | [ac111d0fed](https://github.com/RICHAAARC/SC-SSTW/tree/ac111d0fed253767651929d115c343fe1636c525) | `/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/report.md` |

See [machine-readable identities](evidence/temporal_mechanisms.json), [original GROW evidence](grow_video_reference_v1.md), and [current scope](current_method.md). Historical JSON snapshots under configs/historical are unchanged provenance inputs; exact original runners stay at their source commits. Main portable templates require explicit input paths/SHA and hash-bound companion configs.

The fixed deletion audit recomputed 2836 candidates and 512 final-bit rows, with 20 source hashes matching S. It supports the fixed C/D path correspondence and weakest-margin comparison, not BER correction, detection/FPR, broad dynamic robustness or population generalization.

The conditional-joint audit verified result SHA256 `a3e3d64286514817d5ee3c6ce1fe9bd3486b0752a65477ec0682f48991941a44`, 549 audit checks and an independent numeric replay with no issues. It retains 44 logical payload slots (40 blind + 4 oracle), 21 physical encodes and 32 key reads. The 8/8 K0 primary result is a fixed conditional-case result over seven physical inputs from one seen source; all 22 K0 slots were already zero-error, so it does not establish BER improvement. D correspondence was GLOBAL 88/177 versus PATH 177/177 with b2/k88; its minimum margins .504545/.587879/.659091 were not improved on every final or time bit. Seven large media files were checked only by stored metadata in this audit.

## 2026-10-09 research-branch audits (not main implementations)

| Mechanism | Immutable source / Notebook publication | Completed real audit |
|---|---|---|
| Attribution V1 | [source cd5e212](https://github.com/RICHAAARC/SC-SSTW/tree/cd5e21221cb3c4bc95727a2a0ba15ea94b5aea83) / [Notebook e1a2017](https://github.com/RICHAAARC/SC-SSTW/blob/e1a201766f42733c9ca4ac28234bc0d3d34110ee/notebooks/video_trajectory_attribution_v1_colab.ipynb) | C1/C2 **two** confirmation sources, 128 queries: 16 ACCEPT / 112 REJECT / 0 UNCERTAIN / 0 technical missing. `diagnostics/trajectory-attribution-v1-real-run-audit-20261008/root_raw_audit.json` |
| Uncertainty follow-up | [source 1fb1b129](https://github.com/RICHAAARC/SC-SSTW/tree/1fb1b129ba27b58705e407c7ef2d7627aaf47b1e) / [Notebook c8dda3e](https://github.com/RICHAAARC/SC-SSTW/blob/c8dda3ef18bd9df3af8b677d1befeb5e4489a1de/notebooks/video_trajectory_attribution_uncertainty_v1_colab.ipynb) | 22/22 complete, 16 ACCEPT / 6 REJECT / 0 UNCERTAIN / 0 technical missing. Same ten primary queries give both 0/10 coverage denominators; **complete but not covered**. `diagnostics/trajectory-attribution-uncertainty-v1-real-run-audit-20261009/root_raw_audit.json` |

Both are already implemented, user-run, audited and published on the research branch. The follow-up reuses C1/C2 and adds zero independent sources. Its positive/negative controls remain 2/2 and 6/6, false accepts 0/22, no actual same-action alias tie. It does not validate correct abstention or establish population FPR. Exact source/run/result/audit identities and main-inclusion flags are in [machine-readable evidence](evidence/temporal_mechanisms.json); the [formal requirement comparison](research_status_2026-10-09.md) preserves the Stage1 counterexample, known DEV calibration and remaining mechanism gaps. No raw audit or experiment was repeated for this documentation update.
