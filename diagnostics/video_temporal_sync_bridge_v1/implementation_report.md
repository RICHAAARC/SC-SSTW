# Video-Temporal-Sync-Bridge-V1 implementation handoff

A1's local implementation is complete for independent review. This is an
engineering-only candidate: real model, GPU, Colab and new video results are
NOT_EXECUTED. A2 v1 numerical evidence is inherited; A3/A4 v2 delta review and A5 final
adjudication are pending.
No local commit or source publication has been made for this candidate.

The base is `91afbc71aee9f999ff6f7c1738c5ac0be1249a52` on
`dev/video-trajectory-blind-validation-v1`. All task changes are new files;
the previous successful GROW method, adapter, notebook and historical evidence
are unchanged. `snapshot_v2.json` provides the current precise file list and SHA256s,
aggregate tree definition and patch identity.


## v1 findings and narrow v2 repairs

The main session independently ran all12 v1 tests in18.02s with zero skips and
verified the11-file snapshot/patch identity. Its receipt is in the parent
project at `diagnostics/temporal-sync-bridge-main-review-20260929/main_candidate_v1_review.json`.
A2's independent v1 numerical review found the original payload update
bitwise identical, the joint writer's original four-channel maximum delta
difference2.9802e-8, and independent NumPy adjoint maximum error2.6844e-8.
These are attributed independent checks, not A1 reruns.

P1: the main review found that the generated notebook referenced a nonexistent
requirements-temporal-sync-reference.txt. v2 restores the actual existing
requirements-grow-video-reference.txt and rebuilds the notebook. The static
regression extracts each pip `-r` path from code AST and verifies it exists in
the repository, rather than merely matching a string.

P2: A3 independently interrupted the actual reader after atomic blind-file
replacement but before result.json committed. Parent recovery loaded the old
result and erased the already published phase READ/32bits. v2 makes result.json
the canonical checkpoint: event and recovery commit it first, then write the
blind projection. Recovery regenerates that projection from canonical state.
Tests interrupt before canonical commit (no READ may be claimed) and after
canonical commit but before projection (completed READ/32bits must survive).
The original candidate-batch hard-exit test still preserves committed phases.
No transaction framework, method change or new model call is introduced.

The v2 targeted command appends `-k 'notebook_static or checkpoint_boundary or
terminal_diagnostics_and_batched_search_hard_exit'` to the pytest target.
Result:4 passed,10 deselected in1.60s; no skips. `pytest_v2_delta.txt` retains the
output. v1 snapshot, patch and full-test log remain unchanged. The method,
Wan runtime, configuration and method document retain v1 bytes. A2's numerical
evidence is inherited on that basis; v2 is a distinct source tree and the full
v1 suite is not described as rerun for v2.

The frozen method is OFF / PAYLOAD_LAST / PILOT_LAST, one copper-kettle source,
LAST49 control, unchanged Wan revision/seed/geometry and native history.
PILOT_LAST adds channel4's64 independent SHA256-selected coordinates and
finite15-chip LFSR pilot. N47104/eta5888 preserve the prior payload's
N44160/eta5520 conditional per-coefficient step; total energy is not matched.
FP32 payload deltas match the unchanged implementation within absolute1e-7
and relative2e-6 tolerances in the implemented regression.

The receiver observes only saved MP4 bytes, public keys/protocol and frozen
VAE. Full has one trivial candidate. Crop129 uses all53 candidates and four
actual RGB phase encodings. Soft cosine uses FP64, threshold0.5 and absolute
tie tolerance1e-12; missing phases cannot shrink the search. Pilot alone picks
the accepted payload phase. The old1380-vote payload support changes to1350
full/930 crop votes by excluding the first latent and using fixed regular
support. Bit voting rules remain hard-sign/zero0/first-vote tie.

All3 source files,6 derived views,15 normalized phase tensors,30 raw phase
reads,324 candidate slots,12 searches and24 posthoc slots are fixed before
execution. Six terminal/key diagnostics are separate. Candidate persistence
is batched at each key's search boundary; already saved phase features survive
an interrupted candidate batch. Parent startup/monitor cleanup reloads the
child's final checkpoint after termination/wait. Normal NO_PILOT/AMBIGUOUS
is an evaluated negative result, not engineering failure; accepted payload
bits/BER remain unavailable. Oracle references saved phase/candidate evidence
only, without new FFT/VAE/search. Quality has no threshold or blind-data write.

There are2 expected pilot-positive controls and10 related negatives:
OFF unwatermarked4, PAYLOAD_LAST payload-only4, PILOT_LAST wrong-key2. These
are not10 independent unwatermarked samples or a calibrated FPR experiment.
Expected payload recovery in PAYLOAD_LAST is not a pilot false positive.
Wrong-grid recovery of the repeated payload is not localization success.

Retained v1 full-target command:

`/tmp/image-trajectory-reference-a1-official-20260929/cpu-venv-torch26/bin/python -m pytest tests/test_video_temporal_sync_bridge.py -q`

v1 result:12 passed in19.24s,zero skips. `pytest_v1.txt` and
`local_validation.json` retain the exact result and environment. Tests cover
actual FP32 implementation algebra, independent hash-byte layout/LFSR/mapping,
soft-cosine boundaries/zero/nonfinite normalization, finite-offset synthetic
localization, three native50-step histories and pre49 branch identity, actual
synthetic MP4 filename invariance with fixture VAE, partial phase/save failures,
parent failures, valid negative completion, terminal and batched-persistence
accounting, and static notebook receipts. No test loaded model weights.
Earlier working iterations are identified separately in the validation record;
they are not independent replications. The main session's separate FP64
algebra receipt is referenced with its own file SHA and attribution there.

The notebook is an unpublished draft with `SOURCE_SHA=None`, exact first-cell
Drive mount,current-Python/pip/fresh-child workflow and guarded embedded MP4
previews. Environment setup and full-model resources remain unverified for
this candidate. After same-version reviews close, source commit/publication
and immutable notebook binding belong to the coordinating/main session's
current authorization. No push,GPU execution,new experiment or method scan
is part of this implementation handoff.
