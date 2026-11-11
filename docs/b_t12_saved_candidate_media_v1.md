# Saved t12 candidate media observation

This independent adapter observes the unchanged saved run02 coefficients through the original causal head and media/receiver chain. It does not solve for another candidate. The existing CPU numerical and budget failure remains explicit; a finite, complete native output can proceed even if its measured original budget fails. Engineering completion is never scientific PASS.

## Inputs and user execution

Open `notebooks/b_t12_saved_candidate_media_v1.ipynb`, choose a GPU and Run all. The first code cell only mounts Drive. Source downloads normally from `dev/local-joint-state-payload-v1`; no local code upload, source digest, exact-version or Git-state admission checks exist. The actual notebook and builder are maintained together.

Defaults use private `Video-WM/T12-Saved-Candidate-Media-V1/inputs/run02`, existing `Video-WM_Result/main_method/paper_cohort_v1/paper-joint-t12/attempt01`, and its previously located host cache. The 46 saved features, BASE and head must remain readable; they are not regenerated. Because the upload channel limits single files, the saved prediction may be transferred as two raw byte fragments (`r_predicted_packed.npy.part000` and `part001`). Runtime joins these bytes and applies the same ndarray geometry, dtype and finite-value checks. This is a transport operation, not a recomputation or candidate change. A complete `.npy` also works directly.

New output is private `Video-WM/T12-Saved-Candidate-Media-V1/attempt01`. Input packages, formal results and failed old packages are read-only. Weights and downloaded source use runtime scratch, outside retained result roots.

## Unchanged method

Original head replay uses first singleton plus 45 four-frame chunks, saved coefficients, modified two-frame causal history, original bias/padding and RGB scaling. It measures the full 181-frame original L-infinity, support RMS and outside-support residual, with original 8/255, 2/255 and 1e-7 acceptance tolerance. No K/KT, solver, q0/fixed-target gate, projection or rescaling is introduced.

The original RGB8 and first H264 decode feed FULL, CROP, DELETE1, REPEAT1 and DELETE3; each uses the original second H264 recipe. Received native Y feeds the original BCH/relative time path. Received RGB from the same media is shared with the original 60 E41, DCT2 and CDC32 chain. All five time plans freeze before payload reads. Original key00/key01, H/L, B/K/U and 20 downstream QMatch claims remain unchanged. Supporting references carry actual saved E41/project/key operation statuses and original bit-file paths.

Fixed denominators are 822 time nodes, 817 edges, 8,220 physical R0/R1 words / 517,860 physical bits, 4,110 derived F roles (not a third physical copy), 19,200 raw payload slots and 480 final B/K/U slots. Five P0 reference media input5..9 supply the original MSE/PSNR/maxabs comparison. Their maximum five probe/RGB decodes are separate from candidate costs. SSIM and LPIPS are NOT_MEASURED.

The post also writes all 1,440 local payload slots in their actual slot-address order. Missing local files retain unknown rows and operation receipts. The CPU historical result summary is optional post context; its absence cannot block native execution or remove the fixed disclosure of the old run02 budget failure.

## Recovery and interpretation

Each operation is reserved before physical work. Returned calls read saved artifacts; STARTED, failure or interruption consumes the call. Resume can continue only unstarted operations whose dependencies remain available. Lost saved artifacts are reported and never reconstructed by repeating consumed work. A completed post stage makes later Run all read-only. Resource time/RSS hard caps are absent; actual model-load lifecycles are recorded separately.

Failed shared model loading is remembered within the backend instance. It does not cause repeated load attempts or falsely started E41 forwards. A corrupt saved artifact keeps its original return receipt but is unavailable for current support; readable saved derived results remain present. Structurally invalid claim, edge or plan fields become local unknowns with recorded reasons, without dropping the other fixed output rows. Old correct to technical unknown is retention unconfirmed, separately counted from observed wrong/unresolved losses; both prevent a strong preservation claim.

Truth and original paired post records open after blind outputs close. All failures and unknowns retain their fixed slots. WRONG to CORRECT is repair; WRONG to UNRESOLVED removes an error but does not recover the slot. Existing unresolved slots may stay unresolved. Missing old references prevent a paired preservation claim without erasing candidate measurements. System local repair needs all five observed views, all old recorded errors repaired, no new wrong times, and all old correct time/payload support preserved. Single physical replica failure is reported separately. This cannot establish original-budget validity, generalization, time authentication or population FPR.

## Engineering validation

Targeted artificial checks cover call counts, truth ordering, consumed interruption, read-only completion, missing references/artifacts, actual support-reference structure, the original causal-tail replay on tiny fabricated tensors, notebook synchronization and raw transport reconstruction. No saved scientific arrays, model weights, video codecs or GPU are executed during delivery checks. See the task diagnostic directory for preserved failed and final logs and independent review receipts.
