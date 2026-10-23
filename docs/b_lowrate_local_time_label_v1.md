# Low-rate local time label V1

This isolated fixed development candidate adds synchronization metadata once to the cached normalized P0 terminal. It leaves the user's net32 payload generation and all historical defaults/evidence unchanged. It is a new low-rate method supplement, not a repair or complete reproduction of LFG-WM.

The writer uses21 eight-frame nominal time IDs, binary RM(1,5) length32/radius7, separately keyed16-chip alternating pilot, normalized three-axis Haar LLL channels4..15 and time units1..21. Each channel's balanced Walsh PN rows share one keyed spatial permutation. All data/pilot channels are disjoint. Exact ideal native budget is .15²*21*640*sum(sigma4..15²); no post energy fitting, clipping or correction. Actual pilot/data energies, materialization drift, protected channels and boundary bitwise equality are recorded. Equal alpha is not equal disturbance or quality.

The receiver receives only52 video frames, key and public protocol. All8 fixed pixel phase slices are freshly reset-encoded E45(T12), retaining Haar rows1..4. It never slices a full latent to replace a local encode. Phase score is the mean of all64 expected-sign times raw pilot correlations; unique global maximum required, exact ties refused, and the selected hypothesis must pass all16 signs in at least3 rows. RM cannot rescue phase. Zero correlations erase, unique RM radius7 valid IDs may still miscorrect. Final rows require pilot, valid RM and parity. At least2 consistent accepted IDs yield only a CONDITIONAL_UNCALIBRATED provisional origin; no subset rescue. Partial51-frame input refuses before any E.

Source geometry is known to the private fixture actuator. It is never passed to the blind decoder; post evaluation joins it only after raw and blind outputs are saved and read back. MIXED/partial/absent IDs and false coverage retain all slots; nominal FULL is not a guarantee about VAE causal support. Four sources are already seen development, with host and initial phase confounded. No FPR, independent confirmation, authentication, capacity gain or finished temporal path is claimed.

The fixed user package performs new matched P0/TAG D181×8, codec×8, E181×8 and local E45×96. OFF uses the historical actual MP4 for local background and native terminal null only. Both keys share116 received LLL extractions; writer Haar forwards8/inverses4. Original32 checks share FFT72 with80 keyed reads/Counter11776, and retain full128 group+1408 cell pairs and2816 local raw bit pairs. Quality covers16 paired full-video records/2896 frames/2880 adjacencies plus8 clipping records. No extra old-gradient, DiT, train, retry, scan or amplitude increase.

Standalone entry:

    python -m experiments.wan_state_clock.b_lowrate_local_time_label_v1_run --source INPUT_DIRECTORY --output NEW_OUTPUT_DIRECTORY

The default source is the cached four-source P0/OFF development folder. The first attempt is persisted before model load; STARTED is durable before each call. Interrupted/failed attempts are consumed. Later Run all reads only saved JSON. Optional source/environment notes do not consume or block an attempt. Missing actual inputs/imports/shapes/numerics remain real failures.

Build the actual notebook with scripts/build_b_lowrate_local_time_label_v1_notebook.py. Its first code cell only mounts Drive, then it directly uses available editable GitHub source. No source hash/manifest/version/Git-clean requirements or manual ZIP. The notebook directly clones the available dev/local-joint-state-payload-v1 branch from RICHAAARC/SC-SSTW and Run all executes this fixed task without manual ZIP upload. Agent validation is CPU/synthetic/static/stub only; all physical model/media execution is left to the user.

Exact protocol, physical budget and bounded paper audit are in diagnostics/b-line-lowrate-local-time-label-package-20261023/INTERFACE_PLAN.md and BUDGET.json under the parent repository diagnostics directory. The low-rate package keeps the historical432-bit LFG result frozen and does not make its full recovery a prerequisite.
