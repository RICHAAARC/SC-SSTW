# Image-Trajectory-Reference-V1 final engineering review

The frozen v3 candidate has passed the local engineering milestone. A1 is the
sole implementation writer; A2/A3 independently reviewed the same snapshots,
A4 reconciled scope/identity/evidence, and A5 completed the milestone audit.
This authorizes a reviewable local artifact, not a scientific-success claim,
publication, model execution or transfer back to video.

## Final identity and scope

- Branch: `dev/image-trajectory-reference-v1`.
- Base: `92583b6bdd58733427434ad54943c82dc7a9dcc2`.
- Origin: `https://github.com/RICHAAARC/SC-SSTW.git`.
- Frozen v3 files:18, listed individually in `snapshot_v3.json`.
- Snapshot SHA256: `9cd650cbc1026bfd1629d7668c383bb9ac3e6003a2097080848e6bc94f684d61`.
- Patch SHA256: `169021681a724c5b9c8a742ed026e110d837082441f6107b1113887f65baf7ca`.
- Source-tree SHA256: `d333bf5294a465e6a89956b96949655f1c8ea56fcafea2a78369caee703da8c7`.

This report is added after the freeze and does not alter its18 files. Original
v1/v2/v3 snapshots and patches remain intact; the earlier patches preserve the
older versions of records updated during repairs. The runtime,config,official
bridge,notebook and mechanism document are byte-identical between v2 and v3.
The v3 change is only a test import precondition plus its validation records.
Upstream author source files and the old video worktree remain unchanged.

## Review and test attribution

| Version/reviewer | Actual evidence and conclusion |
|---|---|
| A1 v1 | Complete source trees verified:21 GROW/63 Guidance files against fixed Git trees,7/7 handoff-core matches. Official writer/reader bodies ran with fake model components and real DDIMScheduler. Isolated Python3.12.3,torch2.6.0+cpu,torchvision0.21.0+cpu:7 tests passed in5.00s,zero skips; pip check passed. Missing-cache behavior was a separate2 passed/5 skipped check, not method PASS |
| A2 v1/v2 | Core implementation faithful to pinned author code; public32-bit independent reader, actual wrong-key bit assignment and post-read evaluation passed. Flagged incorrect documentation saying x0 was detached. Independently checked formal CVF pages35980-35982 during review; source record now distinguishes formal paper and author review copy |
| A3 v1/v2 | Runtime/notebook/component asset errors,OOM distinction and same-revision loading reviewed. Found MODEL_INFO incorrectly classified429/503/timeout as access-required. After v2 repair, core engineering review passed |
| A1 v2 |7 mocked metadata cases added:401/403/404 access-required;429/503,timeout and invalid revision engineering failure. Complete14-test file passed in5.76s,zero skips. That test order already imported Diffusers; it did not prove isolated metadata-only execution |
| A5 v2 | Runtime/mechanism review passed; combined partial OFF PNG-write failure and GROW/CORRECT read failure retained INCOMPLETE with1/2 images,1/4 reads,2/8 evaluations. Found metadata tests depended on earlier dependency imports, so the engineering milestone remained open |
| A1 v3 | Official dependencies explicitly preloaded before the test intercepts model-loading CUDA admission. Separate new processes:metadata-only7 passed/7 deselected in4.31s; all14 passed in5.02s;zero skips |
| A2/A3 v3 |18/18 file identities and limited delta confirmed. Runtime/config/bridge/notebook/docs unchanged; inherit their v2 core engineering PASS rather than claim a new method rerun |
| A5 v3 | New-process isolated metadata selection:7 passed,zero skips,in4.25s. Test-isolation P3 closed; local engineering milestone PASS. Retains its prior mixed PNG/read failure evidence and fake-versus-real distinction |
| A4 final | Independently reconciled18/18 identities,scope,origin and dedicated branch; final engineering delivery accepted |

The repaired P2 is failure taxonomy only: metadata401/403/404 and missing
resources remain `ASSET_ACCESS_REQUIRED`; non-access metadata failures become
`RESOLVE_FAILED` and top-level `ENGINEERING_FAILURE`, with stage/error and all
fixed rows retained. No retries or alternative assets were introduced.

Two P3 findings are closed. Documentation now states that the official writer
requests a local gradient with respect to conditional x0, runs the UNet under
no_grad, and has no explicit per-step detach. The test isolation repair loads
dependencies before interception. Dependency import may query CUDA
availability; the intercepted actual model-loading CUDA check is proven
unentered on metadata failure. This is not a claim that no availability query
occurs anywhere in the process. Neither repair changes author method code.

## Method decision and evidence ceiling

Choose GROW's fixed published implementation
`6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`: it provides complete writer/receiver
source and packaging without an additional learned detector or whitener.
Guidance `d1f8e48946f830087baf32356e5ce0eeec68e27e` has an actual ProbGuidance
configuration chain, but missing VideoSeal/whitener assets and incomplete
asset instructions; it is documented, not silently substituted or executed.

The GROW source's nominal DCT helper is actually the real part of an
orthonormal FFT. The formal CVF paper specifies DCT and different loss
presentation/payload/channel/eta settings. This candidate deliberately keeps
the published FFT-real/mean-MSE/default32-bit execution; it does not claim to
reproduce the paper's DCT experiment. No video hinge,positive-group guard,
three-direction basis,R_STAR or Wan code is imported into this image baseline.

The preselected case is one official owl prompt with seed42. Original OFF and
GROW functions produce2 PNGs. Each PNG is reopened and independently read with
correct and verified-different wrong-key layouts, yielding4 raw32-bit rows.
Those bytes are persisted before joining OKOK/NOPE in8 separate evaluations.
Raw bit accuracy/BER/exact bits are primary; UTF8 strings are only diagnostic.
Saved-image quality metrics/display have no invented pass threshold. This is
one fixed comparison, not a population FPR,robustness or generalization test.

The original SD2.1-base API and model_index anonymously returned401; the
underlying access cause is not established. This is an asset-access failure,
not a method failure or a license to use another checkpoint. A credential does
not guarantee access. Model resolution/loading must use the original asset and
one immutable revision, with failures and the full2/4/8 denominator retained.

No complete model was loaded; no real image generation,GPU or Colab execution
occurred. Local tests used fake components with original function bodies.
The full pinned Colab installation and model path remain untested. The notebook
is an unpublished draft with `SOURCE_SHA=None`; there is no runnable published
candidate link yet. Historical video evidence remains6/6 points,18/18
candidates,0/18 ACCEPT and16/18 loss worsening. That negative is preserved and
is not reinterpreted as either image success or a proof that image guidance
cannot transfer.

## Next publication scope

The only proposed publication is this new
`dev/image-trajectory-reference-v1` candidate: obtain current main-session/user
authorization, publish that dedicated branch, verify its immutable published
SHA, then bind the notebook with the builder and supply its Colab link. The
current action is a local commit only. No push,merge,GPU execution,video
continuation or additional experiment is authorized by this report.
