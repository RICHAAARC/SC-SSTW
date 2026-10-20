# C73 fixed identity feasibility package — Card74

Adopted single C73 synchronization extension: 64 unused spatial directions per ROI, 46 orthogonal cell identities including guards, A_ID=0.5, rho<1/16. Existing native correct-key 22 directions preserved in continuous arithmetic; actual FP32 drift, wrong-key changes and energy recorded. Net32 and A/default receiver are unchanged. No payload rewrite or FFT.

The user runs one fixed package: two saved P/S terminals, one S_ID construction; all three arms fresh D, quantization, CRF18/yuv420p/fps8 MP4/readback/probe; six different nine-frame reset E per arm (18E total). A/B/C/D clean starts61/60/62/109; E/F after deleting source63 starts61/64. G after deletion at108 reuses same-arm D, including failure, with no new call. F has complete target and discontinuous context and remains in the complete-target denominator.

Run locally with available source:

    python -m experiments.wan_state_clock.c73_identity_v1_run --config experiments/wan_state_clock/configs/c73_identity_v1.json

For Colab, open notebooks/c73_identity_v1.ipynb and Run all. First code cell only mounts Drive. The next cell automatically downloads the ordinary portable source ZIP from https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/c73_identity_v1_portable_source.zip. No manual ZIP upload is needed. SOURCE_LOCATION also accepts an available source directory, local ZIP or ordinary URL. Actual download, extraction or import failures are reported directly. Dependency setup and the fixed runner follow automatically. The config already names existing Drive P/S endpoints and a new durable same-Drive output. The agent does not launch Colab/GPU or write Drive.

Before each operation, attempt/STARTED state is written, flushed and closed; file fsync is attempted where supported. Unsupported file-fsync capability is recorded without blocking; directory fsync is not required. Actual write/replace/I/O failures still fail the operation. Failure or interruption spends its slot and the entire attempt remains one-shot. Later Run all reads saved records only, even after interruption, missing checkpoint or unreadable JSON; it does not regenerate, fill missing calls, or resume physical work. Unentered/failed/undefined rows are retained. No hash, manifest, Git state, exact version or resource hard limit is required.

36 physical y vectors ×46 =1656 ID slots (3 geometric positives,1653 negatives); 42 logical vectors/1932 slots include276 reused checks. 30 control vectors plus3 correct-key SID noncomplete vectors are checked for every ID, not just same identity. All native/local y/s/rho/status and failure placeholders are durable before post-only source/role evaluation. A code member is never emitted as a phase certificate or READ path. C and E can have identical ideal mixtures; no full gap recovery, full receiver, FPR or full32 payload-regression claim follows. All rejection is not success.

Quality is three matched pairs × RGB8/MP4,1086 frame rows/1080 adjacent rows, plus3 preclip records. P0/S_base uses legacy nominal support; pairs involving SID include nominal guard support, neither is actual causal support. Actual clipping, host-removal/target/cross energy, FP32 error, native target error, old22 drift and observed resources are retained without a quality PASS threshold.

CPU tests and one fixed real-saved-endpoint construction have separate local diagnostics/ledgers; they are engineering evidence only. Notebook GPU/Drive/model/media execution remains for the user. Optional 18FFT payload layer is not adopted and is absent from this runner.
