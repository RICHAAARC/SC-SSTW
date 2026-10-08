# Bounded development evidence index

Updated 2026-10-08. This is an index of construction-specific outcomes, not an
import of development implementations. Source links are immutable. Removed old executable entrypoints are listed in
[the historical entry index](historical_entries.md).
Large original results and media remain in the user's Video-WM Drive archive.
The project-local audit paths below are provenance pointers, not files shipped
in this repository.

| Route | Retained boundary | Source / audit |
|---|---|---|
| GROW fixed video reference | 3 MP4, 24 reads, 48 evaluations; MULTI/LAST MP4 0/32 errors. Repeated payload only. | [Compact evidence](evidence/grow_video_reference_fixed_run.json); project audit diagnostics/grow-video-reference-real-run-audit-20260929 |
| OLD8 local Fourier RM | Same native MULTI version has terminal/DIRECT 1/1 paths, RAW420 3/2 and MP4 22/30; final-media path not recovered. | [Construction](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_v1.md); diagnostics/local-fourier-rm-lowband-real-run-audit-20261003 |
| Adjacent-difference reception | Limited RGB8 positive; tested MP4 still failed. No retuning/selection converts it to success. | [Protocol and recorded outcome](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_difference_v1.md); diagnostics/local-fourier-rm-difference-real-run-audit-20261003 |
| LOWBAND16 / CONTRAST | LOWBAND failed already at terminal; contrast improved some layers but did not recover the true path and failed MP4. | [LOWBAND](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_lowband_v1.md), [CONTRAST](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_contrast_v1.md); diagnostics/local-fourier-rm-contrast-real-run-audit-20261004 |
| Two spatial replicas | Same-run actual state norm matched; B consumed 85.1% of summed update energy. Own-protocol path ranks 10/3,19/7,27/40,34/30 across terminal/DIRECT/RAW420/MP4. | [Construction](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_replica_v1.md); run 20261003T203404941538Z; diagnostics/local-fourier-rm-replica-real-run-audit-20261004 |
| DWELL4 | CPU-only proposal: ideal paths distinguishable, minimum separation reduced to 1/3; no new real media result. | Project-only diagnostics/old8-state-information-20261004; no published implementation imported |
| Saved-terminal gradient / STE / direction-budget series | Objective improvement did not imply actual raw420/path improvement; retain exact tested negatives. | [Historical comparison](https://github.com/RICHAAARC/SC-SSTW/blob/794600e2717f18017e3f1ec057f003ae688a42fa/docs/video_local_fourier_rm_lowband_v1.md); diagnostics/raw420-forward-accept-real-run-audit-20261002 |
| Older PN / fixed-key / affine paths | Carrier-specific partial evidence; algorithm/CPU correctness is not portability to current observations. | Historical branches remain unchanged; no receiver or DP dependency block copied into main |
| Stage1 presence authorization counterexample | Source ab60d457c58eb6fb1ec3e5bd402f3dc86eded5bd, run 20261007T122708953536Z: 177 positives 2/2 but null 1/6 ACCEPT (4 REJECT, 1 UNCERTAIN); 89 positives 2/2 and null 0/6 ACCEPT, but all six nulls UNCERTAIN. | diagnostics/terminal-sync-path-decision-real-run-audit-20261007; retained as negative evidence, not a conditional-joint dependency or gate |
| Trajectory attribution V1 | Fixed DEV64 → confirmation128 user run completed at source cd5e212…; the audited confirmation rows were 16 ACCEPT / 112 REJECT / 0 UNCERTAIN, with all 128 technically complete. | [Protocol](video_trajectory_attribution_v1.md); diagnostics/trajectory-attribution-v1-real-run-audit-20261008 |
| Attribution uncertainty follow-up V1 research candidate | Fixed saved-input receiver follow-up reuses C1/C2 post-codec RGB and the original rule for 10 shared primary probes, 4 alias controls and 8 regression controls. No thresholds, inputs or attacks are selected after scoring. | [Protocol](video_trajectory_attribution_uncertainty_v1.md); no new real result until the user runs the user-run notebook |

The same-source local-state comparisons are dependent. Repeated-payload 9/9
recovery across marked arms and media layers is not nine independent videos,
nor evidence that synchronization aided recovery. Any future shared-component
extraction requires a separately bounded interface and validation.

The later accepted M05/global-alignment/fixed-deletion whitelist and its original-version evidence are indexed separately in [mechanism evidence](mechanism_evidence_index.md). The historical negatives above are not imported as current method dependencies.
