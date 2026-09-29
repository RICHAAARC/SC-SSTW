# GROW-Video-Reference-V1 final engineering review

Candidate v2 is ready for the main session's source publication and notebook
binding. This is an engineering milestone only: real video generation,
real-model VAE survival, GPU execution and Colab execution are NOT_EXECUTED.
No recovery, quality, generalization or scientific PASS is inferred.

## Frozen identity and review scope

- Branch: `dev/video-trajectory-blind-validation-v1`.
- Base: `92583b6bdd58733427434ad54943c82dc7a9dcc2`.
- v2 snapshot: `snapshot_v2.json`, SHA256
  `3c2696dc1b375f54679da3bc4b1b7087247f7f9ebf598e6131382be21e55842e`.
- v2 patch: `candidate_v2.patch`, SHA256
  `516a7b78ee9ee27f17b2ec88d8f3591a80f2a582ec5b3c54eb87c4e25d78fbc6`.
- v2 14-file tree SHA256:
  `7dbcaf6d718347136116fa3f06d6b32f3fc29556f8017612e81d7b357bf652d8`.
  This is the snapshot's sorted path/NUL/file-SHA/LF aggregate, not a Git tree ID.
- v1 snapshot SHA256:
  `c83043579e2dc493a253cd87ce1f0d670d112c6d2a9cca740d7539fac3c40524`;
  patch SHA256:
  `f8a06362172dc9f583c29fbf357a10c05caedbab62cc900764319dee89854d99`;
  tree SHA256:
  `a51aea6133134f5044b545a4e51eaaa92916ccb1cbd5d6728cd592f7ed768122`.

The v1 snapshot, patch and test log remain intact. v2 changes only the parent
worker lifecycle, its three fault regressions and local validation records;
the method, runtime, shared loader, configuration, dependency ranges, notebook,
builder and method document retain their v1 bytes. This report is outside the
14-file freeze and changes none of those files. The delivery comprises those
14 files, both versions' four snapshot/patch artifacts, and this report:19 files.

## Independent review and evidence attribution

| Reviewer | Version and evidence | Engineering conclusion |
|---|---|---|
| A1, sole writer | v1:12 tests passed in14.32s. v2 lifecycle:3 passed,12 deselected in0.95s; complete target:15 passed in13.30s,zero skips. AST,source whitespace and reverse-patch checks passed. Logs and environment are in `local_validation.json` and `pytest_v*.txt` | Implementation and failure handling pass on CPU fixtures |
| A2, mechanism/numerical review | v1 full method review passed,including independent NumPy adjoint comparison with maximum error1.11e-16; v2 identity/delta review confirmed the method remained unchanged | v1 mechanism evidence inherited into v2; no new method claim |
| A3, runtime review | Independently reproduced the v1 parent-startup P2; v2 original counterexample closed. Independent monitor/interruption regressions:2 passed in0.88s | Runtime and recovery review passed after the narrow fix |
| Main session | Independently checked v1 frozen hashes/patch and12 tests in12.75s; v2 new3 tests in0.94s and identity checks passed. Its v1 receipt is in parent diagnostics `video-image-reference-migration-20260929/main_candidate_v1_review.json` | Same-version source and targeted CPU validation passed |
| A4/root, synthesis | Independently reconciled14/14 v2 file hashes,patch/tree identities,scope and review conclusions | Final source scope and evidence attribution passed |
| A5, final milestone audit | Independently combined a MULTI partial MP4-save failure with a LAST/mp4/WRONG reader failure; details below | v2 final engineering milestone passed |

Reviewers' results above are attributed to their independent review reports
relayed by the coordinating session; they are not additional A1 executions.
The v1 and v2 source-tree hashes differ. v1 mechanism evidence is inherited
because the mechanism bytes are unchanged,not relabeled as a v2 rerun.

A5's combined CPU/fake execution retained all3 generation,3 MP4,24 raw-read
and48 evaluation rows with terminal states. Completed counts were3 generation,
2 MP4,21 raw reads and42 evaluations. Within the separate primary-MP4
6-read/12-evaluation denominator,only3 reads and6 evaluations completed;
diagnostic layers did not replace missing primary evidence. Blind-read file
bytes were unchanged before and after evaluation and quality reporting.
Quality used2 extra reads for the2 saved MP4s,with no threshold. These are
failure-retention checks,not generated-video or real-VAE recovery evidence.

## Closed P2 and operational behavior

v1 handled a child returning nonzero but did not handle parent log-open,
process-start or monitor exceptions. An injected `Popen` OSError left the
result RUNNING/INITIALIZE,all rows PENDING and no blind-read file.

v2 records the failed phase. A started child is terminated and waited for
before the parent reloads its latest disk Store,so a checkpoint written during
shutdown is preserved. Ordinary phase failures finalize unfinished rows and
continue feasible subsequent work. Even failure to start either child leaves
all fixed rows and48 missing evaluations persisted. A keyboard interruption
stops further media/quality work,retains results and exits130. A successful
execution outcome also requires completed worker phases. Normal model,reader,
method and sampling calls were not changed by this repair.

## Fixed construction and claim limits

This is the adopted transfer from the successful image reference's published
FFT-real/mean-MSE construction to native Wan Flow. It preserves the author's
actual `fft2(...,norm='ortho').real` operator; it does not claim to reproduce
the formal paper's DCT setup. The image source/run are recorded in the method
document and remain reference evidence,not a video success.

The single preselected copper-kettle source uses seed2026092501,Wan1.3B at
revision`0fad780a534b6463e45facd96134c9f345acfa5b`,181 frames at320x512/8fps,
50 native UniPC steps andCFG5. Four latent channels carry32 bits across all46
latent times; the actual44160-element mask gives eta5520 from image1600/200.
The complete local mean-MSE gradient updates conditional clean x0 before one
CFG,using native `x0=z-sigma*v` conversion and continuous scheduler history.
There is no denoiser/tail VJP,7.5/5 compensation,three-direction solve,R* or
positive guard. OFF,MULTI25..49,LAST49 share initial noise and precision;
MULTI includes the last step and cannot isolate non-last contributions.

The primary reader receives only the saved MP4 path,public protocol/keys and
frozen VAE. Its chain is MP4 readback -> posterior mode -> Wan normalization
-> FFT-real payload. Three MP4s,four dependent layers,two keys and two
post-read truth targets yield24 raw reads and48 evaluations. Raw reads are
persisted before truth is joined. Actual key assignments differ; a different
key string alone is insufficient. RMSE/PSNR are separate diagnostics without
quality gates,and three dependent arms are not independent samples.

CPU tests use synthetic tensors,a real native scheduler,synthetic FFmpeg
media and fixture VAE/child processes; no model weights were loaded. Historical
results are unchanged,including the six-point continuous candidate's0/18
ACCEPT and earlier temporal-difference paired MP4 partial positives. The old
FREQ/SPATIAL implementations already used full local gradients and consistent
DCT writing/reading; the historical JAC used a real input VJP. No blanket
claim that all previous video constructions failed is made.

The notebook remains an unpublished draft with `SOURCE_SHA=None`. Under the
current authorization,A1 creates only the reviewed local commit. The main
session handles publication of this dedicated branch,verifies its immutable
published source SHA,binds the notebook and supplies the user-run Colab link.
A1 does not push,merge,load models,run GPU/Colab or start any new experiment.
