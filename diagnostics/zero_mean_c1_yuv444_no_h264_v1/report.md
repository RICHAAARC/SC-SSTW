# YUV444 no-H264 local candidate

The local candidate is ready for main-session review. It is unpublished; the notebook has `SOURCE_SHA = None`. No real conversion, model execution, commit, or push was performed.

## Implementation

- Saved Q8 from layered run `20260930T165530863126Z` enters two materialized rawvideo processes: RGB24 -> yuv444p -> RGB24. Each file is exactly 88,965,120 bytes (181 x 512 x 320 x 3); six new raw files total 533,790,720 bytes. Commands retain automatic matrix/range/scaler settings and the inherited `left` input label. A format/option rejection remains a recorded conversion failure.
- Same FP32 Wan VAE, full 181 frames, fixed g0/R44, original reader and payload diagnostic: three new normalized tensors, six new readouts, 1044 valid costs. No Transformer, VAE decode, H264 encode, or MP4 decode.
- Three reference layers contain nine normalized tensors and 18 saved raw records, 27 references in total. Layered inputs and the separately fixed 420 run `20261001T092411269871Z` have independent roots and result identities. Reference failure retains the full roster and does not erase new evidence.
- The original FULL.phase0 opponent remains the fixed comparison (positive CORRECT index34). New and old layers show their own winners, full top sets, and true-minus-winner alongside this unchanged fixed opponent. No row44 removal or method/budget/threshold change.
- Existing 420 and inherited files have no tracked diff. The 444 adapter/runner are controlled copies to keep earlier files immutable and avoid runtime global substitution. The reader, numerical reporting, opponent loading, worker cleanup, and recovery functions are AST-identical to their proven 420 versions.

## Validation

Five focused CPU mock/static tests passed in 10.06 seconds: fixed command/format/budgets; full fake conversion/VAE workflow with exactly six new infer calls, 27 references, blind/posthoc separation and reference-only loss; failed second conversion with retained first file and other arms; short raw output rejection; reproducible unbound notebook and both read-only input-root guards. These tests execute no real conversion or model.

All 27 existing local reference files and both parent results match their configured hashes. Notebook: six cells, five AST-valid code cells, exact Drive mount first cell, empty outputs, source unbound. Detailed file hashes and validation are in `local_validation.json`; test output is in `pytest_local.txt`.

Main-session independent read-only review: PASS on the same six core file hashes (`/home/richar/projects/Video-WM/diagnostics/zero-mean-yuv444-main-review-20261001/review.json`). No core file changed after that review.

## Interpretation ceiling

This control tests whether the observed ranking failure persists without chroma subsampling. Recovery would support a role for 420-related processing differences; it would not establish pure chroma subsampling as the sole cause in the original MP4 chain. This is one saved source with dependent arms, without FPR, quality, or generalization claims.
