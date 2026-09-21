# Current integrated method

The adopted historical method remains available through
`experiments.wan_state_clock.run`. The integrated method uses fresh
prompt/seed noise, runs the native Wan trajectory to state 44, and forks OFF,
SINGLE46, and MULTI44_46 paths. SINGLE46 receives the complete fixed native
response budget `0.042943312697648145`; MULTI44_46 receives half at each step.
Every second control recomputes velocity and a temperature-one clean-leaf
gradient from its controlled live state and full scheduler history. Steps
47--49 remain uncontrolled.

The payload is one arbitrary nibble, not an A/B alias. RM(1,3) maps four net
bits to eight data windows with physical hard-window distance four; three keyed
pilot windows use a separate loss term with fixed weight `0.25`. The bounded
decoder guarantees only unique recovery when `2*hard_errors+erasures<4`.
It makes no frame-error or soft-channel claim. The 16-way target uses the mean
of all rivals; with two templates this expression is exactly the historical
`codes[m]-codes[1-m]` tanh objective.

The receiver maps every message to eleven keyed base phases, negating data
states according to its RM word and leaving the three pilots common. It reuses
the legacy fixed-gain `state_clock.observe` update and bounded clock paths, but
its 16-code score is a new receiver and is not described as the previously
validated A/B `local_state` result. The legacy A/B code and tests remain intact.
The receiver accepts only four phase observations and a public codebook; prompt,
truth, original latent, writer evidence, and edit truth are absent. Truth joins
happen after ranking and existence decisions.

The current receiver protocol adds the frozen historical three-group emission:
four groups retain the original calculation; exactly three sum only observed
slot projections and clip once; fewer groups erase the window. Full and partial
blocks are separate, and scores and aggregate evidence use observed-component
weights. This changes the statistic and requires a threshold bound to
`SC-SSTW-Payload-RM13-Partial3-V2` and the receiver key identifier.

Two independent OFF sources are generated and fully received before the
threshold is frozen. Each source statistic is the maximum over all 16 messages,
clock paths, seven saved views, and the fixed three-crop aggregate. Any missing
view makes calibration `UNCALIBRATED`; evaluation cannot return a payload in
that state. The fixed evaluation generation and attack collection still runs so
the denominator and engineering failures remain observable, but all existence
decisions stay `UNCALIBRATED` and return no payload. The two calibration OFF
sources receive the same per-view, crop-aggregate, and source decisions after
threshold construction; they are labeled construction samples and are not
held-out FPR evidence. Two calibration sources give empirical rank resolution
1/3 and do not support a low-FPR claim.

A raw receiver ranking remains persisted even when only some phase encodes are
available. Protocol eligibility is stricter: a view can enter calibration,
aggregation, source maximization, or existence decisions only when the view is
`SCORED` and all four phase rows are `COMPLETE`. Partial raw rankings therefore
cannot freeze a threshold or return a payload.

Reusable entrypoints live in `runtime.wan.integrated_core`, with CLI
`python -m runtime.wan.integrated_cli`. The writer accepts prompt, seed, any
nibble, key, and a public protocol file and produces an MP4 through the same
44/46 control. The independent receiver accepts an MP4, key, explicit protocol,
and optional calibration; it does not read writer terminals, trajectories,
truth, or saved codebooks. The fixed experiment imports these entrypoints while
retaining its `[0x5,0xa]` roster and fixed denominators.
The explicit protocol must exactly equal the canonical public JSON; the same ID
does not authorize changed model, generation, method, or media fields.

The fixed roster is two calibration OFF contents and two evaluation contents
carrying payloads `0x5` and `0xa`. Evaluation generates OFF, SINGLE46, and
MULTI44_46 for each content. Every arm persists FULL, three actual 129-frame
crop MP4s, DELETE90, SPEED5_4, and a second-generation REENCODE. Every saved
view receives four independent VAE phase encodes. The denominator is four fresh
cases, eight arms, 56 saved views, and 224 receiver encodes; failures and
missing rows remain in place.

The three-crop aggregate stores compact evidence for each payload's own best
path. It selects the aggregate payload first, then combines that payload's
matched-path window evidence and runs RM decoding, so different view winners
cannot substitute unrelated hard evidence.

This complete engineering integration is the method accepted for publication
to `main`. It contains no GPU/model result for the new integrated method. CPU/fake
tests establish code, call accounting, lifecycle, blindness, and denominator
behavior only. They do not establish payload recovery after real generation,
quality, generalization, FPR, or paper readiness.

The accepted immutable binding is source S4
`7ad4d9425bfb246b1cd5e4f95aab0c08de521df2` and notebook N4
`909b7ae7a82e58d99704a50ba63c5ccc8ead6f81`. The fixed user-run notebook is
https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/909b7ae7a82e58d99704a50ba63c5ccc8ead6f81/notebooks/integrated_payload_v1_colab.ipynb
and writes under
`/content/drive/MyDrive/Video-WM/SC-SSTW-Core-Integration/integrated_payload_v1_<UTC timestamp>`.
Remote publication and byte verification are reported separately by the release
publisher.
