# Fixed-direction channel attribution

Select a GPU runtime and **Run all**. Uses the fixed BASELINE, COMPOSITE and WORST_ONLY rasters from 20261002T131059238475Z and the same terminal steps and raw420 observations from 20261002T053509124852Z. All input locations are prefilled. The two arms remain matched to their original objectives; neither endpoint is selected or updated.

Only two native encoder VJPs are computed, sequentially at the identical saved baseline raw420 received RGB. COMPOSITE uses the original global hinge plus mean of all 173 pair hinges; WORST_ONLY retains the full-family global hinge. Each full normalized replay is compared with the saved baseline. Cotangents are saved in 16-frame chunks (at most about 32 MB each) so later download and independent dot-product verification are practical. Existing causal checkpoint storage and dependency repair are reused, with no GPU-name or exact-environment gate.

For each matched arm, report five signed loss-change quantities:

1. Saved terminal gradient dotted with actual saved terminal displacement, including float32 step rounding.
2. New baseline raw420 RGB cotangent dotted with saved clamped FLOAT RGB displacement.
3. The same cotangent dotted with saved RGB8/255 displacement.
4. The same cotangent dotted with actual saved raw420 received RGB displacement.
5. Actual objective change, recomputed from saved raw420 normalized tensors over all 173 wrong classes.

A negative prediction means objective decrease. Consecutive differences form a telescoping decomposition of prediction error. The first difference also includes cross-run gradient replay differences; the last includes finite encoder response and piecewise loss effects. All intermediate cotangents belong to the raw420 baseline, not to FLOAT/RGB8 objectives. This is fixed-direction first-order attribution, not proof of a unique physical cause or a new gradient method. Native FLOAT includes the unchanged [0,1] clamp.

The existing large FLOAT files are read directly from mounted Drive. Their quantization to the saved RGB8 rasters and equality to the old channel inputs are checked on CPU and reported. No raster is regenerated. No new decode, decoder VJP, writer step, color conversion, receiver search or payload read is performed. Three old raw420 observations, six old blind readouts and twelve message evaluations remain references. Native encoder replay mismatches are reported, without discarding either fixed arm. Failed and missing rows stay in the two-arm denominator.

Static/CPU tests cover arithmetic, saved-input plumbing, fixed denominators, replay reporting, chunk persistence and failure handling. The two full pretrained encoder VJPs are left to user execution. This diagnostic does not establish MP4 robustness, unknown-phase synchronization, path-aligned payload closure, population FPR or generation-time trajectory embedding.

Results: MyDrive/Video-WM/Zero-Mean-Channel-Direction-V1/<timestamp>/fixed_reference/result.json.
