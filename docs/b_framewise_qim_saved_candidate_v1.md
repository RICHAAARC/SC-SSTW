# Saved QIM candidate: independent native diagnostic

This prepares one future user-run diagnostic of the unchanged saved finite coefficients, targets, features and BASE. The original stopped solver/attempt remains read-only and numerically unresolved; optimality is unproved. No solver continuation, extra iteration, coefficient scaling, projection or new candidate is included. The mechanism change is to observe the previously unexecuted saved-candidate → actual head → media → reads path, not repeat an old optimizer or latent/velocity writer. The historical failed/closed routes and DCT2+CDC32 semantics remain unchanged.

The actual thin notebook is `notebooks/b_framewise_qim_saved_candidate_v1.ipynb`, assembled by `scripts/build_b_framewise_qim_saved_candidate_v1_notebook.py`. First cell: only the two-line Colab Drive mount. Settings are the old attempt, truth source, new output, optional actual model id/path and ordinary editable source location. Default Run all starts the fixed physical package only when the user runs it. No notebook experiment cell was run during implementation.

Defaults:

- Saved attempt: `/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1`.
- Truth source: `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`.
- New output: `/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1-saved-candidate-native-v1`.
- Preflight: NEW_OUTPUT's sibling with suffix `-preflight-v1`. Logs and ordinary entry cache are siblings with `-logs` and `-entry-cache`.

The new output and its derived preflight/log/cache write destinations must be outside both read-only input directories. Bootstrap and Session check actual resolved paths before creating/downloading to cache or preparing source; writable source/scratch locations are checked as well. A derived sibling that aliases an old input is reported before any write. A NEW_OUTPUT that is an existing ordinary file is an explicit OUTPUT_CONFLICT and stays untouched. Preflight reads core paths, 46 required feature indices, JSON method settings, NumPy format/shape/dtype headers and PyTorch container headers. It does not unpickle tensors or certify finite content. Missing assets produce a sibling report with the fixed plan and zero physical started counts; repairing those inputs allows preflight again. Original BASE media, P0, BASE raw/identity and identity truth are not preconditions for new QIM physical observations. Unavailable saved BASE observations make the corresponding comparison missing; there is no new BASE read to fill the gap. Tensor shapes/finite values and actual head compatibility are checked only after reservation.

`attempt.json` alone identifies a consumed attempt. Unknown content in NEW_OUTPUT without an attempt is an output conflict, never overwritten. After exclusive reservation, every actual call is counted started before invocation and returned only after it returns. Failure and interruption consume the attempt; a subsequent invocation is read-only. The notebook detects consumption before downloading source, setup or importing scientific modules. A complete ordinary cache gives detailed standard-library reporting; an unavailable cache still gives raw saved attempt/result state without download or rerun. Cache/source/version equality is not required.

The independent CLI is:

```bash
python -m experiments.wan_state_clock.b_framewise_qim_saved_candidate_v1_run \
  --saved-attempt /path/to/fixed-e01-181-v1 \
  --truth-source /path/to/fixed-four-new-v1 \
  --output /path/to/fixed-e01-181-v1-saved-candidate-native-v1
```

`--model-id` optionally supplies an actually available model id/path. Import, `--help`, preflight and consumed recovery require only the standard library. The experiment CLI does not depend on the optional notebook, setup or report files. Scientific runtime import happens only after reservation. Existing successful dependency repair and controlled worker behavior are reused by ordinary entry helpers, not copied into notebook cells.

Fixed new budgets: model load lifecycle **1**, K **1** (136 internal `conv_transpose2d` calls), BASE constraint P **1**, Aa constraint P **1**, native head **46**, full time reads **4**, E41 **8**, codec encode/probe/decode **1 each**. D181, baseline head, solver, KT, DiT, training and BASE new physical calls are **0**. Historical attempt counts are separate and are not debited against these new limits.

The runtime restores the same FP64 coefficients/targets and mmap BASE, loads the existing VAE once and streams the original feature files one chunk at a time. One full BASE P projects packed blocks; one K forms cached Aa; one Aa P checks the unchanged nearest/tie targets, 1e-3 gray-level target tolerance, support, RMS2/255, L∞8/255 and 1e-7 RGB budget excess. Saved targets are compared, never rewritten. No split/stationarity criterion substitutes for these direct constraints. A failure stops before head replay.

The same cached Aa goes into native replay; there is no second K or baseline replay. Native FP32 head executes the original first-singleton plus 45 four-frame chunks, carrying the modified last two frames. Bias, causal padding and `/2` scale remain. The same materialization observes full-canvas outside-support delta, actual targets/budgets and prediction differences. Chunk seams use the same per-frame target/delta/prediction records, without extra head calls, PSNR or new quality analysis. Prediction error is observational, without an added threshold or causal attribution claim.

Finite complete native output continues into fixed reads even when native target/budget tolerances fail; that failure remains visible. Nonfinite/incomplete output stops dependent work. Codec failure preserves the first three time layers and leaves h264/identity slots unmeasured. H264 is stored and passed to identity as uint8 0..255; the time reader divides by 255 once. The receiver receives no truth, writer active mask or j; raw q/d0/d1/soft/mask persist before fusion. Identity truth opens only after every fixed read has terminated.

Fixed new denominators: **456120 time slots** (4 × 114030, 112770 active each), **1280 identity local slots**, **1280 paired slots**, and **320 eligible complete-cell slots**. Support changes are separately reported for local1280 and eligible320. A comparable pair requires both read receipts RETURNED and each slot AVAILABLE, ERASED or UNSUPPORTED. A real AVAILABLE→ERASED/UNSUPPORTED is lost support; the reverse is added support. Failed, not started, unresolved or missing observations are unmeasured and excluded from change counts. Common BER additionally requires both AVAILABLE and eligible; an empty intersection or missing truth gives None. The old saved `lost_support` is never rewritten and its previously unexecuted QIM slots are not presented as demonstrated damage.

Pure constraints live in `main/tube_state/b_framewise_qim_saved_candidate_v1.py`; restore/cached apply in `runtime/wan/b_framewise_qim_saved_candidate_v1_backend.py`; experiment, records, post-evaluation and optional report in distinct `experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_*` modules. Runtime never imports experiment/entry/report. The older protocol, solver, DCT2 and CDC32 implementations remain unchanged.

Validation uses artificial headers/JSON, fake backend calls and tiny heads only. Actual saved coefficients/targets/features were not loaded or computed, and no actual model/weights, GPU, head/E41/codec, Drive/Colab or physical experiment was executed. The delivered implementation cannot establish actual feasibility, optimality, media survival, blind synchronization, authentication, quality or scientific PASS.
