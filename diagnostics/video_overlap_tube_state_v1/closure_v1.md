# Overlap tube state: local review closure

A2 and A3 passed the frozen B/C1 implementation with no remaining code-change requests. A4 independently matched the B twelve-file and C1 ten-file snapshots, trees and patches, and found no implementation blocker. A5 completed the final read-only milestone audit with PASS. These are local engineering conclusions, not real-video scientific validation.

B: 15 CPU tests passed (1.05 s, no skips); one fixed grid completed 1,680 primary conditions plus 24 counterexamples in 18.40 s. All 420 marked/correct-key conditions retain truth in the top set: 390 unique finite-model hypotheses and 30 retained structural ambiguities. The 60 marked/correct no-edit conditions do not require a false edit. All outputs remain uncalibrated and unaccepted; OFF/wrong-key fits do not establish FPR. Exact-copy insertion and repetition are observationally indistinguishable; null and midpoint insertions remain unsupported-family counterexamples.

C1: 9 CPU/static tests passed (18.12 s, no skips), including synthetic-Transformer/native-scheduler trajectories, FP32 gradient/cap/Flow, four-phase blind adapter, missing phases, canonical persistence boundaries, worker startup failure and notebook validation. C1 uses a new common full support R44 and crop R31; historical R45 support remains unchanged.

Frozen identities:

- `snapshot_b_v1.json` SHA256 `07228109c05a26c2e7462a892431248169d18c7fa956b553de4d42ed37a88fbf`; tree `63c880dffa4ec3bedb6b998e890ba44e7f3d192243d802a9cca04a7908b8fb40`; patch `d9ef89162990071424236cb500563601dda1c5fea6a9562a7c6ff86b5f049f71`.
- `snapshot_c1_v1.json` SHA256 `29058a96fdb3d8cbb1d2175e95136f524002947d3344b4089fae9926b55ef0f1`; tree `cf841f7450f6d6e6f25cabe32488baecdce8da1e7c5f12cd138cb7ef4a5eb5fe`; patch `20f74412a8183d3cc1da4448a582aa120ed26ff0dc074796c7600fdb8b0ad83d`.

The original B run recorded source identities at completion, not contemporaneously at start. Its executed runner is preserved at `/home/richar/projects/Video-WM/diagnostics/overlap-state-a1-development-20260930/executed_video_overlap_tube_state_run.py`, SHA256 `9ea161f9fdcfee5792e43b044e3bc83ab90cdfd773d9183b032dce55659f7a9e`. The final runner adds only start/end identity capture and persisted-cost completeness checking. No original result was rewritten and no scoring rerun was used to mask that receipt difference. The independent audit checked all 1,211,646 costs and 1,750 output hashes; original results remain indexed by offline_result_manifest_b_v1.json.

Primary review receipts (read and hashed for this closure):

- [stage_B_saved_review.json](/home/richar/projects/Video-WM/diagnostics/overlap-state-main-review-20260930/stage_B_saved_review.json): SHA256 `a592461de5992bcf7e7ed27c268f8db9071d50d85f3719a6a6eef97b42a9ae4d`.
- [pilot_control_review.json](/home/richar/projects/Video-WM/diagnostics/overlap-state-main-review-20260930/pilot_control_review.json): SHA256 `43715fe7a24646d4631023a8fdd94dfd1f13efbf274b21abe3b5ef6d056a441d`.
- [c1_phase_adapter_review.json](/home/richar/projects/Video-WM/diagnostics/overlap-state-main-review-20260930/c1_phase_adapter_review.json): SHA256 `2f8df971d9e4bcfb5ae03f4fa47c5ed14c899ab92bc1417df4c1d0dd64932755`.
- [c1_snapshot_review.json](/home/richar/projects/Video-WM/diagnostics/overlap-state-main-review-20260930/c1_snapshot_review.json): SHA256 `c7200633328115ba419a6715c8f81a6417cf64eaf4293d6425ba5dedf8d1e214`.

The saved-B review independently matched 1,704 conditions / 1,211,646 costs (maximum absolute score error 1.93e-17), plus eight actual tensor recipes and 36 projections. The pilot review checked below-cap, above-cap and target cases against a closed-form gradient (maximum element error 1.35e-8; L2 1.000000019 is FP32 rounding). The phase review used five physical tensor fixtures at the VAE boundary and matched all registered full/crop phases and nominal starts; no real VAE or codec ran. The snapshot review matched all ten C1 files and notebook AST/binding.

Unresolved scope: no local implementation blocker remains, but actual model/VAE/codec survival, real-video synchronization, quality and detection calibration are untested. C2 real-edit entry points are prepared but unexecuted; arbitrary RGB edits and position-dependent payload remain unverified. Old negative and successful historical evidence is retained.

This local source candidate may proceed to the separate publication/binding handoff. SOURCE_SHA is still None; source has not been published or notebook-bound by A1. This is not yet a Run-all delivery link. The main session must publish the reviewed source, verify that immutable SHA and bind the notebook. No push, merge, model/GPU/VAE/media or Colab execution is part of this closure.
