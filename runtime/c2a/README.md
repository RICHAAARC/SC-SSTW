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

`run_second_axis_diagnostic.py` and
`notebooks/c2a_second_axis_diagnostic_colab.ipynb` are a separate, fixed
three-arm localization entry (`ZERO`, `PLUS_E2`, `MINUS_E2`). They create one
new fixed terminal rather than claiming to recover the earlier one. For each
arm they preserve the shared terminal, actual FP32-written 2x8x8 blocks for
all ordinary groups, lossless pre-codec float RGB, pre-codec q, MP4, and
post-MP4 q. Actual-write covariance is recomputed after FP32 `copy_` from the
block that is decoded; the float64 transport identity remains separately
labeled analytic metadata. The package writes only below
`MyDrive/Video-WM/C2A_SecondAxis_Diagnostic/<UTC-run-id>/` and retains all
three arm outcomes. Its declared work is one 100-forward terminal generation,
three VAE decodes, and six VAE encodes; it does not perform a parameter scan,
new fit, or 2B activity.

`run_quantization_diagnostic.py` and
`notebooks/c2a_quantization_diagnostic_colab.ipynb` are a distinct, fixed
three-arm readout of the persisted second-axis run
`c2a_second_axis_20260914T145106Z`. They reuse the exact RGB8 rounding used by
`ffmpeg_roundtrip` (`np.rint(clamp(rgb)*255).astype(uint8)` followed by
`/255`) but do not invoke FFmpeg, colour conversion, a video codec, terminal
generation, a transformer, or VAE decode. A single FP32 VAE loader performs
three posterior-mode encodes and saves the RGB8 uint8 tensors, q at the source
pre-codec/RGB8/source-post-MP4 layers, direct comparisons, and O2/M2
vectors/norms below `MyDrive/Video-WM/C2A_Quantization_Diagnostic/<UTC-run-id>/`.

`run_yuv420_diagnostic.py` and `notebooks/c2a_yuv420_diagnostic_colab.ipynb`
then read one fixed RGB8 quantization package and isolate the rawvideo
`RGB8 -> YUV420p -> RGB8` conversion. They reuse the original RGB24 geometry,
frame rate, FFmpeg defaults and `yuv420p` sampling, save raw YUV intermediates
and RGB8 readbacks, and perform one frozen VAE load plus three VAE encodes.
They do not call `libx264`, create an MP4, generate a terminal, load a
transformer, or decode a VAE latent. This raw stream shares the conversion
stage but cannot strictly match the original H.264/MP4 path: it omits H.264
quantization and carries no container/codec colour metadata. Commands and
that boundary are persisted below
`MyDrive/Video-WM/C2A_YUV420_Diagnostic/<UTC-run-id>/`; it must not be used to
automatically attribute any remaining difference to H.264.

`run_multiblock_diagnostic.py` and
`notebooks/c2a_multiblock_diagnostic_colab.ipynb` read the exact saved
second-axis terminal and perform a fixed spatial-repetition comparison: one
original block `(16,28)` versus four non-overlapping blocks `(16,28)`,
`(4,16)`, `(4,40)`, `(28,28)`. Shared OFF plus five fixed states for each
layout makes 11 independent outputs; it is not a public temporal layout or a
blind calibration experiment. Every written block and ordinary group receives
its own covariance transport and actual post-FP32-write measurement. Each
output performs one FP32 VAE decode, CRF18 H.264/YUV420p readback and one VAE
encode; generation and transformer calls remain zero.

For each layout, `b_b` and `A_b` are fitted only from ZERO/+E1/+E2. A state
is calibrated per block before the fixed equal-weight mean is taken. No
pseudoinverse, failed-block omission, or reweighting is allowed; a failed or
near-singular block makes the four-block layout unsupported while preserving
all records. Negative directions are retained holdouts. Full post-MP4
reencoded normalized latents and lossless pre-codec RGB tensors are saved for
each readable output; the latter are about 11 × 92 MiB, so the package needs
roughly 1 GiB before MP4 and metadata overhead.
