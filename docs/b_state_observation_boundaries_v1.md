# Saved observation boundaries, fixed e01/e02

This adopted diagnostic uses the completed three-group run as input. It compares e01/e02 × P0/PM at the fixed CROP M window. It is post-truth diagnosis: the private source origin identifies the same 40 coordinates at every layer. Neither P0 nor private truth becomes an input to the blind receiver.

## Run the published notebook

Open [the Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_state_observation_boundaries_v1.ipynb) and choose Run all. The first cell only mounts Drive. The next cell downloads the available source from `dev/local-joint-state-payload-v1` and locates this diagnostic entry point; no source ZIP upload or parameter edit is needed. Optional `SOURCE_DIRECTORY` or `SOURCE_ZIP` accepts an editable local copy. No Git checkout, manifest, hash, or dependency-version equality is required. Unsafe ZIP paths and actual download/import failures are errors.

Input defaults to `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`. Output defaults to `/content/drive/MyDrive/Video-WM/B-State-Observation-Boundaries-V1/fixed-e01-e02-v1`. They must be separate. Keep that fixed output for later Run all: any execution artifact makes it read-only before source download, dependency setup, or model loading. Do not delete or rename an attempt to retry it.

The setup reuses the successful three-group environment repair: import probes install missing packages, including ftfy; the actual `FqnToConfig`/`torchao.quantization` incompatibility removes optional torchao once, then probes again in a fresh process. It does not pin or replace a working dependency set. VAE loading uses FP32, posterior mode and the existing shared Drive cache selection. An actual load failure stops the attempt without switching sources.

CLI: `python -m experiments.wan_state_clock.b_state_observation_boundaries_v1_run --input ORIGINAL --output NEW_OUTPUT`. `--help` and read-only recovery need only the Python standard library. The source ZIP includes the actual local import closure and this notebook, without original results, tensors, caches or private inputs.

## Fixed observations and budget

| Layer | Input | New top-level encode calls |
|---|---|---:|
| N | Saved native Q | 0 |
| A | Saved full float RGB, all 181 source frames | 4 E181 |
| B | Same float RGB, fixed 41-frame source slice | 4 E41 |
| C | Saved uint8 RGB8, same 41 frames, divided by 255 once | 4 E41 |
| D | Saved CROP/M true-phase final codec2 Q | 0 |

A/B use `full/float_rgb.pt` exactly as saved: already clipped, not quantized, with no additional clamp. C uses `full/rgb8.pt`; it does not quantize float again. e01 uses phase2, s19, source slice `[72:113]`; e02 uses phase1, s16, `[60:101]`. Protocol source origins and each P0/PM CROP plan's M start53 and true member geometry are checked. The real writer secret is read from protocol; key00 is the model label.

Maximum12 new top-level encodes =4E181+8E41, 1052 encoded input frames. Eight saved and twelve new Q40 give20 fixed layer/case rows and800 coordinates. Failure or interruption spends the attempt and stops subsequent physical calls; untouched rows remain NOT_ENTERED. Inputs, model load and per-layer receipts are persisted before entering their work. A hard interruption can leave STARTED/attempt1, which read-only recovery preserves over a stale ledger. It never resumes or fills gaps. A model-load failure also spends the attempt. Encoder/decoder internal forward-block counts, when observable, are recorded separately from actual top-level encode starts/returns/failures.

There is no generation, DiT, gradient, decode, codec, new media, background fitting, calibration or phase/window/strength search. The backend shares the existing reencode normalization and resets VAE caches around every call.

## Measurements and interpretation

For i=0..9 and ROI=0..3, flatten observed-row first, then ROI. N/A use `roi_q[s-1+i, ROI, s-1+i]`; B/C/D use `roi_q[i, ROI, s-1+i]`. A/B/C compute only these predeclared projections using the unchanged state directions/signs.

Using the saved key00 background, x=L^-1(q-mu), u=L^-1(ones): signed alpha=(x·u)/(u·u), residual=x-alpha*u, and S=max(0,cos(x,u)). Signed alpha is not the production nonnegative alpha estimate. Tiny cosine rounding is bounded; invalid and zero-energy scores are retained as unresolved and stop further physical work. Raw P0→PM delta is compared with native delta using direction cosine and mean(delta)/mean(native delta); zero norms or zero native means remain explicit unresolved metric fields. Whitened delta statistics supplement, not replace, these raw-Q measures.

`result.json`, `evaluation.json`, twenty `layers/` slots, `receipts/`, `physical_counts.json`, and the fixed plan retain failures and denominators. The existing tau is a reference line only. The descriptive first-loss field requires e02 to cross below it and remain below while e01 remains above from that boundary onward; mixed paths remain mixed. It is not a significance test or a scientific PASS.

N→A measures the original decode-to-full-reencode path jointly. A→B measures full-history versus reset/short-window boundary jointly. B→C isolates saved 8-bit quantization in this fixed encoder comparison. C→D combines codec1, fixture copy and codec2, including their encoding context; C and D use the same E41 reset context. This finite diagnostic cannot isolate the two codec passes or establish population-wide causality.

After these observations, the method decision must use the measured boundary, residual and paired-direction evidence to recommend a concrete next candidate. It does not preselect a winner. Increasing gamma remains a possible candidate; native q1 is not a physical limit. The current write is terminal, so there is no later sampling tail that can dilute it. This implementation and local CPU checks alone do not establish real VAE feasibility or a scientific result.
