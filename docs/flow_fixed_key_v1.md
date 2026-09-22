# Fixed-key Flow watermark V1

This isolated development candidate detects one key-derived marker. Four-bit attribution remains preserved on main and is not used by this entrypoint. Its prior implementation and the complete `integrated_payload_v1_20260921T163637378085Z` audit are retained.

## Frozen method

The template equals `state_clock.codebook(key)["codes"][0]`: all 11 windows and 1760 keyed spatial supports, including the legacy directions, sync, polarity and message0 state schedule. The fixed development key is `WanProjection-first-validation-key-v1`. The carrier and state-code files are byte-identical to successful Uniform-Tanh source `3dd3d0ec7c69dd67fcd6e7acf94ae0f3434ba084`. This is provenance, not validation of the new objective or existence receiver.

The writer minimizes `-mean(c_fixed*tanh(projection))` at temperature 1 on the detached CPU current clean leaf. This single-template objective is new; it is not the old A-minus-B objective. All 11 windows carry the same fixed marker and supply synchronization/existence evidence; no 3-pilot/8-data split, message maximization or ECC is used.

Fresh prompt/seed/noise passes through the native prefix to step44. OFF, SINGLE46 and MULTI44_46 branch from that state. SINGLE uses full `R*=0.042943312697648145`; MULTI uses `R*/2` at each of 44 and 46, recomputing the second direction on its controlled live history. Steps47–49 are free. Budget means sum of immediate same-history native response RMS, not equal energy or terminal displacement.

For each of the fixed4284 clock paths, physical four-group or actual three-group projection is summed before a single clip to[-1,1]. Window marker means use160 spatial supports; the path matched term is weighted by observed components(640/full,480/partial3). Fewer than3 groups contributes no matched weight and observer innovation1. Zero usable windows makes the path INVALID. The observer normalizes q direction and averages innovation over11 windows. `score=matched−0.05*innovation−0.002*event`. Thus the full final score is not uniformly component-weighted or a calibrated likelihood. One fixed template is searched; no hidden user-message selection occurs. Ties prefer no event, near identity scale/offset, then deterministic phase/boundary fields.

All four receiver phases must complete for a formal view. Nonfinite observations are INVALID. Each view receives its own decision using the common threshold. The source statistic is the maximum over all three fixed views and their complete time-search family. Two independent calibration OFF source maxima plus1e-6 freeze the boundary. Missing calibration views/phases invalidate calibration; evaluation still runs but remains UNCALIBRATED. Calibration OFF cases are not held-out false positives. Key and full canonical protocol bindings prevent threshold reuse under another definition.

## Fixed development run

2calibration cases×OFF plus2evaluation cases×(OFF,SINGLE46,MULTI44_46): 4fresh cases,8arms,24saved views and96VAE encodes. Views are FULL, DELETE90 and SPEED5_4. No view is skipped because FULL fails. Existing development prompt/seed contents are reused, but videos/terminal/observations are generated afresh; this is not new-content generalization. Evaluation IDs are `eval_p2_s2` and `eval_p3_s3`, with the old content mapping retained in the manifest.

Calls: generation4, transformer448, native scheduler224, shadow/probe/backward6 each, decode8, saved/read MP4s24 each, encode96. All attempts, completions, failures and missing slots remain in the result.

Per-view blind best and fixed diagnostic paths retain11-window signed evidence, q/norm, observed groups/components, innovation and chosen clock path. Terminal identity statistics use the same clipped/observer score as receiver identity. Fixed diagnostic paths never enter the detector maximum or calibration. FULL reports nominal identity agreement; DELETE90 excludes event window5 only from alignment metrics, never from score; SPEED5_4 reports nominal scale and group-level allocation agreement, not exact frame synchronization. Reference-eligible windows are a fixed denominator; missing best windows remain failures.

## Entry points

- User Run-all notebook: `notebooks/flow_fixed_key_v1_colab.ipynb`.
- Fixed runner: `python -m experiments.wan_state_clock.flow_fixed_key_run --output NEW_RUN_PATH`.
- Reusable generation: `python -m runtime.wan.fixed_key_cli generate --prompt TEXT --seed N --key KEY --protocol runtime/wan/fixed_key_protocol.json --output VIDEO.mp4`.
- Independent receive: `python -m runtime.wan.fixed_key_cli receive --input-mp4 VIDEO.mp4 --key KEY --protocol runtime/wan/fixed_key_protocol.json --calibration CALIBRATION.json --output RESULT.json`.

The receiver accepts only video/key/protocol/calibration; writer evidence and truth labels never enter detection. Experiment labels and reference-path comparisons are post-search reports.

The notebook uses an independent first two-line Drive mount and the successful N5 installation argv without new dependency ceilings or hardware-model restrictions. It creates a new timestamped Drive directory before installation, retains setup/pip/fresh-process version logs there even on installation failure, and runs a source-pinned fresh process. Output: `/content/drive/MyDrive/Video-WM/Flow-Fixed-Key-V1/flow_fixed_key_v1_<UTC timestamp>`.

## Evidence boundary

Local11 CPU/stub checks passed. This candidate has not been run on GPU/model/Colab by the implementation agent. Historical successful code and the completed four-bit failure diagnosis do not establish this fixed-key candidate's existence detection, lowFPR, quality, key generalization or robustness. Actual Run all execution remains for the user. No wrong-key experiment or additional threshold has been added.
