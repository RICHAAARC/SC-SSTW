# Channel-margin fixed continuation V1

This user-run diagnostic continues the completed single-update run20261001T133512881485Z. Its result SHA256 is236799579e960f9a34295129eb32a66e6a845efa2dcb4925e6ec59821fccf005; the actual AFTER terminal SHA256 is e11dd7fe61f6150cde77661f230f9c6d3a55802dd8706625424b7ca9b218e1a3. It is a full saved terminal, not a receiver re-encoding or projected sidecar. Original writer terminal and historical YUV444 controls remain separately verified inputs.

## One fixed sequence

POINT1 is a fresh replay of the saved step1. Exactly three additional updates produce POINT2, POINT3 and POINT4. The endpoint is POINT4, regardless of intermediate rank or loss. All outcomes are retained; there is no winning-iterate selection, early rank-based stopping, budget scan or automatic additional run.

Each update retains the accepted global-worst plus all-pair margin objective, eta174, shrink-only single-step L2cap1, original1392-dimensional pilot write support, full181frames, g0/R44 and all174valid paths. Each gradient is newly computed at the current terminal through the real color-forward channel and the same split native encoder/decoder VJPs. Saved step1 gradients are not reused. Public margins remain template-derived, never retuned against the new output. The existing pure method, decoder/encoder adapter and blind reader files are unchanged.

New step L2 sums are at most3 in ideal arithmetic. No extra cumulative projection or clamp is added; actual displacement relative to the original marked terminal and saved step1, per-step changes, sum of step norms, sum of squared step norms and floating-point addition residuals are reported. Thus four steps do not share a total cap of1. Zero supported gradient is a recorded no-op and the fixed sequence continues; nonfinite gradients and execution errors remain failures. No-op is not convergence or acceptance.

## Observation and reference accounting

POINT1–POINT3's gradient encoder forward also supplies that point's independently persisted received latent and blind readout. POINT4 uses the ordinary native no-gradient encoder. All four points go through fresh native decode, RGB8 quantization, actual raw444 and reopened RGB24. Raw blind reads persist before truth reporting. Each point retains both keys and both message comparisons, all R44 positions, all raw costs and catalog exclusions. The main denominator is4observations/8path reads/8payload reads/16message comparisons/1392costs/31320catalog slots/29928structural exclusions.

The prior run's three historical normalized/six raw control references are hash-checked again. Its actual step1 normalized/two raw files and both BEFORE/AFTER full RGB files are also verified. POINT1 replay differences are reported without silently replacing the baseline or substituting an earlier successful point. Comparisons within the new sequence use the actual fresh POINT1. Existing references are not new searches or independent FPR observations.

Quality is measured from actual RGB readbacks against (a) the original marked terminal, (b) the saved step1, and (c) the preceding new point. This isolates accumulated and incremental changes but does not certify overall quality relative to unmarked OFF. All four full terminals, normalized tensors, raw RGB/YUV, pilot gradients and applied updates are saved under a new output root. No old artifact is modified.

## Execution and resources

Fixed logical calls:4native decodes,3gradient decodes,3gradient encodes,1native encode,3encoder VJPs,3decoder VJPs,4RGB-to444 and4inverse conversions,8path and8payload reads. Encoder and decoder each perform138forward and138recomputed chunks across their three phases. Transformer calls remain0. Every gradient phase creates and releases its own existing bounded disk spool. There is no new GPU-name or free80GiB preflight gate; actual runtime failures are retained. The previous13.3minute gradient pair suggests about40minutes for three pairs plus other overhead; no timing guarantee.

The method question is whether the observed positive one-step controllability extends to a useful all-candidate positive margin without losing payload or acceptable incremental quality. A positive POINT4 result is still a same-source terminal-development result. Connecting the objective to MULTI generation, normal tail and MP4, unknown phase/snippet synchronization and independent confirmation remain separate subsequent work.

## Local validation and release

Local tests exercise the complete fixed loop with a fake channel/VJP, fresh terminal inputs, fixed final endpoint even after early ideal recovery, all raw/quality/denominator records, zero-gradient continuation, gradient failure retention, parent identities and notebook bindings. They do not execute pretrained weights, actual codecs or a new model experiment. Publish source first, then bind and publish the immutable notebook for the user to Run all.
