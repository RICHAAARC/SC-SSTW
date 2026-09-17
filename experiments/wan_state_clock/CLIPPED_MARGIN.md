# Saved-prefix clipped-margin comparison

Source and notebook publication is authorized; the user runs the notebook.
The assistant has not executed real models, GPU, Colab or Drive experiments.
This is a method candidate, not a confirmed bug fix.

The sole new writing objective is
`mean(clamp(c_wrong*p,-1,1)-clamp(c_correct*p,-1,1))`.
It is exactly the negative nominal full-support receiver matched margin.
Same-code supports cancel; saturated supports have zero gradient, including
wrong-sign saturated supports. Native clamp uses the interior subgradient at
exact +/-1; central finite differences at these kinks are not that derivative.
Zero/nonfinite coefficient gradients retain failed/missing rows with no fallback.
This objective does not align the observer or max-over-clock blind decision.

The original hinge remains the default of velocity_direction_run; a private
restored context selects this isolated new objective and saved prefix. The new
entry reuses zero_path preflight and full UniPC restoration, original codebook,
R and scalar responses. New prompt embeddings are reconstructed; original model
revision/embedding identity is not established by saved artifacts. It runs two
new zero-gradient tails and six rho tails per case, with original checkpoint,
44/45/46 control, steps through 49, q and budget arithmetic. No prefix, old hinge,
old terminal write, response probe or old media is rerun.

Fixed inputs: velocity_calibration_20260917T011448482341Z and
velocity_zero_path_20260917T092208999384Z. New output must be separate from both.
All four dev cases and rho .1/.3/1 A/B remain: 24 terminal rows per method,
including missing/failed rows. The existing unbounded worst/mean selector and
at most two media candidates are unchanged. Media decodes new terminals only;
quality compares to original OFF.mp4; original TERMINAL controls and their
quality/readout records remain historical references. Zero terminal tensor
comparisons against old OFF and Z0 are reported explicitly. Nonidentical zeros
mean matched-control equivalence is not demonstrated; no automatic baseline rerun.
Selection and comparison expose MATCHED / NONMATCHING / UNVERIFIED for saved
zero equality per case, plus paired_method_comparison_valid. A nonmatching or
unverified result wraps the unchanged selector result in PROVISIONAL_*_HISTORICAL_CONTROLS;
raw_selector_result retains its exact decision. This is reporting, not a new
selection threshold. Effective source/output/log metadata is new; the original
config is preserved separately, with generation/model/key unchanged.
No holdout stage exists. Original NO_SELECTION is copied into comparison output,
never overwritten. Differing candidate lists retain all 24 report rows:
NOT_RUN_ORIGINAL / NOT_SELECTED_FOR_MEDIA / missing or failed remain distinct.

CLI, for a later explicitly authorized run:
`python -m experiments.wan_state_clock.clipped_margin_run --input-run OLD --zero-run Z0 --output NEW --stage develop`
Then the same command with `--stage media`; `--stage report` only reads saved
artifacts and writes comparison.json. There is no implicit all/model entry.
Per-case child processes free model/graph ownership between cases.

Fixed development compute upper plan: 384 ordinary Transformer forwards,
192 live scheduler steps, 192 detached shadow steps, 8 backward calls; outer and
inner checkpoint replays retain their existing independent accounting.
New media upper bound: 16 decode/save and 64 VAE encode calls. No GPU feasibility,
readout, quality, or scientific success follows from CPU validation.

Published source: `8b55bce72e661d0705370daff6b4d59d8c0ed57a`.
`notebooks/clipped_margin_colab.ipynb` binds this immutable source and Run all
executes develop, media, then report with the two fixed Drive inputs above.
