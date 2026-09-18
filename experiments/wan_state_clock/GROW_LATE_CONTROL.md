# Fixed late-window terminal candidate

Engineering implementation only; no real model run or media result is asserted.
The sole method change from the original fixed candidate is moving its 20 local
control steps from indices 10–29 to 30–49. Carrier, 16-bit messages, amplitude
0.5, eta 0.1, CFG 5 and native 50-step UniPC are unchanged. The original runtime
and manifest remain untouched. No terminal gradient, new loss, observer, scan,
or automatic media stage is added.

Run explicitly with `python -m experiments.wan_state_clock.grow_late_control_run
--output NEW_DIRECTORY`. Four development cases each retain OFF/A/B, including
failures. All eight marked terminal messages must have 16/16 correct bits and
zero erasures for `media_eligible`; this is only permission to consider media
validation, and never runs VAE/MP4 automatically or asserts scientific success.

Full budget: 1200 Transformer calls, 600 live scheduler steps, 160 local
gradients, 160 independent native OFF shadow steps. Each shadow uses identical
pre-step state/history and the uncontrolled velocity. Its output is diagnostic
only, never fed to the live solver. Control-induced displacement is compared
against this shadow, separately from the whole transition. Cumulative RMS is
the sum of individual step RMS values, not an accumulated displacement norm.

Last-step state, predicted clean before/after control, actual next state and
terminal tensors/readouts/losses are preserved. A nonpositive controlled sigma
fails; it is never silently skipped. Native final lower-order behavior can make
the terminal equal the final controlled clean prediction: recovery therefore
does not by itself prove earlier guidance persisted.

The same eta and 20 controlled steps do not imply an equal actual control
budget. If late-window recovery improves alongside increased cumulative control,
the improvement cannot be attributed to timing alone. Even an 8/8 result only
establishes development-set terminal eligibility; MP4 quality and independent
validation remain unmeasured.

Historical comparison is explicitly NOT_ESTABLISHED until source parameters,
actual schedule, initial tensor hash and environment are checked. Conditioning
and resolved model revisions also matter; missing old metadata prevents a fully
matched causal timing claim. This limits interpretation, not candidate execution.
The earlier 0/8 terminal result is descriptive context only.
If a later causal timing comparison is authorized, the minimal matched extension
would replay the original 10–29 controls for the eight A/B arms using the same
new environment, saved initial latent, conditioning and schedule, reusing OFF.
That extension is neither executed nor automatically scheduled here.

Guidance Watermarking https://arxiv.org/html/2509.22126v2 section 5.5/Table 3
reports late guidance for Sana + VideoSeal. It motivates testing a timing
hypothesis; it is not evidence that this Wan frequency carrier succeeds.
