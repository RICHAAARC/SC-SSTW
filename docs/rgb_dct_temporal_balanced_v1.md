# RGB-DCT-Temporal-Balanced-V1 development candidate

`main/tube_state/rgb_dct_presence.py` defines the complete score specification in
`SPEC_TEXT`; `SPEC_SHA256` hashes that text. The candidate takes exactly 181
complete 320×512 gamma-encoded RGB frames with values in [0,1]. Input storage
may be NumPy `float32` or `float64`; luminance, DCT projection, and reductions
are computed in `float64`. It uses every frame and pixel, without resizing,
cropping, frame selection, or transfer-function linearization.

Call `score_rgb(rgb, key)` with a nonempty bytes key. It returns a continuous
uncalibrated score or `INVALID`, the specification and keyed-code identities,
and used-frame/feature counts. It has no arm, writer, truth, threshold,
probability, or decision input. The output always has `decision=None`.

`runtime/wan/rgb_dct_presence_adapter.py` adds `score_mp4(path, key)`. It uses
the existing `runtime.wan.io.read_mp4` ffmpeg RGB24 decode, moves its returned
tensor to CPU NumPy, and calls the same `score_rgb`. Decode failure is
`INVALID`. The adapter does not inspect filenames for arm, truth, or writer
state.

This code has no learned weights or calibration. The old Wan marker is not an
H1 sample for this new receiver. CPU tests establish the implementation's
mechanical properties only; they do not establish e1/e2, low FPR, or a real
video detection result. No real media has been processed by this candidate.
