# C2 2A terminal-latent interface preparation

This is a prepared, unexecuted real-model adapter. It does not change the
synthetic-only claim ceiling of `main/`, and it creates no default workload.

`protocol.py` fixes the short interface carrier to six arms: `OFF`, `ZERO`,
`PLUS_E1`, `MINUS_E1`, `PLUS_E2`, and `MINUS_E2`. `ZERO` is an actual
covariance normalization arm, not a no-op. The three ordinary latent groups
are `(1, 2, 3)` after special group zero; all three are written and only the
middle group is read. Channel numbering is zero-based `(0, 1)` and the block
is the central `8x8` support using floor rounding.

The negative-axis arms are holdouts for this short carrier-axis check only.
They do not test the fourth public point `rho*c_i`, a full public-layout fit,
or the 27-state packet; those remain later 2B/2C work if 2A is supported.

The writer has no epsilon, pseudoinverse, or fallback. A non-finite or
near-singular empirical covariance is an unsupported retained arm when
`lambda_min <= 1e-8 * max(lambda_max, 1)`. This is an engineering choice made
before execution, not a claim that phase one had already fixed it.

The prepared runtime chain is:

```text
one generated normalized terminal latent
  -> six cloned terminal-latent arms
  -> frozen FP32 VAE decode
  -> RGB24 H.264 MP4 and ffmpeg readback
  -> frozen VAE encode, posterior mode, re-normalization
  -> fixed C2 q readout
```

The old phase2 resources are implementation references only:
`phase2.py::decode_float` supplies normalized-latent decode scaling, and
`phase2_execution.py` supplies FFmpeg RGB24 export/readback. Their RGB luma
readout and OFF/Y-minus controller are not C2 code.

`generation.py` reuses the old `load_wan` choices that are relevant to the
endpoint: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, the existing cube prompt,
seed 1275, 320x512, 49 frames, 50 steps, CFG 5, BF16 transformer and an
independently loaded FP32 VAE. It runs one ordinary no-gradient scheduler path
to the terminal latent, then C2A clones that one tensor six times. The old
feedback, autograd/checkpoint logic and its Y-minus arm are deliberately not
reused.

The predeclared output is descriptive rather than a performance PASS: all six
rows, covariance failures, paired quality measures before codec and after the
RGB24 MP4 readback, two response singular values and the two negative-axis
holdouts. The documented 15% / 40 dB / 5% figures remain visibly unadopted
candidates. One terminal latent cannot estimate noise, so it cannot support a
thresholded response claim.

The historical model/prompt/seed are selected as a fixed first interface
configuration because they were the actual old entrypoint, not because their
old scientific outcome transfers to C2. The revision is recorded when
available but is not a version whitelist. The Colab setup follows the prior
notebook's normal dependency and model-loading path; ordinary load failures
are retained as run failures rather than converted into a separate preflight.

`run_2a.py` calls this one terminal-generation adapter itself; it no longer
requires a user-provided `.pt` latent. It records the actual scheduler and
terminal-latent metadata beside the six-arm result.

The Colab notebook fetches this prepared runtime from its named public GitHub
branch. It contains no repository credential and does not use Drive as a
source-delivery channel. A future enabled six-arm run writes its arm videos,
metrics, configuration and failure records directly to
`MyDrive/Video-WM/C2A_2A/<UTC-run-id>/`; it never reuses an existing run
directory.
