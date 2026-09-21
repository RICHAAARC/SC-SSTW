# T46 finite-amplitude terminal response selection

This independently named entry preserves the published identity-feedback runtime and notebook. It changes only the sign/skip decision for one fixed T46 tube control. No real model/GPU run has been performed as part of implementation.

## Fixed protocol

Use the two existing development cases `dev_p0_s0` and `dev_p1_s0`, messages A/B, and formal arms OFF, LOCAL_A/B, RESPONSE_A/B: **10 formal videos, 40 receiver encodes, four paired comparisons**. They are development contents, not independent validation. Keep the original state × polarity × sync codebook, 1760 tube blocks, margin 1 objective, native 50-step solver, full MP4/four RGB origins/five receiver modes unchanged. Primary media metric is local_state true-message competitive margin, with unique correct ranking separately recorded.

Run the full OFF reference once, keeping z46, v46 and the complete solver snapshot. Reproduce each message's future LAST49 full-control actual native response R49 in one additional native step. R49 is a conditional mechanism-diagnostic oracle, not an online budget or one deployable parameter shared across examples.

At T46 let d = P_m(z46 − sigma46 v46) − (z46 − sigma46 v46). Normalize by fixed-support RMS to q. One native unit probe from the original history uses v46 − q/sigma46; compute epsilon = R49 / RMS_support(Dunit). Both signs use exactly that epsilon. It is a finite-amplitude clean-control coordinate, not necessarily a small differential perturbation. The allowed formal coordinate h is only −epsilon, 0 or +epsilon, so |h| ≤ epsilon; there is no scan or extra amplification.

Zero direction is explicitly ZERO_DIRECTION with q = 0 and epsilon = 0. Still execute the zero unit probe, both probe tails and formal tails, keeping the fixed calls and outputs. Nonfinite or unmatchable nonzero directions are failures. Oracle-target mismatch is retained as a diagnostic; a valid zero/skip arm has expected actual native response zero.

Each positive/negative probe independently clones the original full T46 history, applies u = ±epsilon q via delta_v = −u/sigma46, runs the real native step46, and freely continues steps47–49 under no_grad. Use the existing terminal latent hinge objective J = 0.5 mean_b relu(1 − c_mb projection_b)^2; J0 is measured from the actual OFF terminal. No terminal gradient, Transformer backward, VAE gradient or receiver truth is involved.

Set tau = 1e−6 max(1, |J0|, |Jplus|, |Jminus|). A side is eligible exactly when J0 − Jside > tau. With no eligible side choose sign0 SKIP_NONIMPROVING; with one choose it; with both choose smaller loss, except a loss difference ≤ tau chooses +. Eligibility is evaluated before the tie preference. Tau is only a numerical decision deadband, not an effect threshold or noise guarantee. Missing/nonfinite probe means INVALID and a retained failed RESPONSE slot, never skip; LOCAL still runs independently when its direction is available.

Record the finite-amplitude secant (Jplus − Jminus)/(2epsilon), or None for epsilon0, and Jplus + Jminus − 2J0. The secant is diagnostic in the unit-clean-control coordinate and does not determine step size; R49 must not replace epsilon in its denominator.

## Formal outputs and evidence

LOCAL formally runs +epsilon. RESPONSE formally runs the selected h. Both start from fresh copies of the original state/history and regenerate their entire tail; a probe terminal is never reused as a formal output. Record formal/probe actual D, original oracle target, expected arm response, terminal fingerprint, terminal objective, and same-control replay difference. A replay match is an engineering determinism check, not independent predictive evidence. Sign0 RESPONSE is still a newly executed formal zero continuation and still goes through media. Correct A/B ranking of a zero/skip video does not establish successful embedding.

All available formal terminals go through decode → H264 MP4 save → actual readback → four RGB origins → VAE encode → blind original reader. No selection/loss gate removes a formal arm from media. Truth joins only after receiver search. Keep fixed 10/40 failures and four RESPONSE-vs-LOCAL rows, distinguishing terminal completeness from media completeness. Report positive/negative/skip/zero/invalid decisions plus formal active-control status. If all four decisions are +, the controller reduces to LOCAL and offers no new control benefit. Active counts never replace the fixed denominator.

Compare actual terminal loss and MP4 local_state margin against both same-case/message OFF and LOCAL. OFF rankings are not calibrated detection/FPR. Two messages are not a payload-capacity experiment. PSNR and temporal residuals against OFF are output-difference diagnostics, not perceptual quality approval; manual content/artifact/motion review remains necessary. No trajectory-success or generalization claim follows from CPU tests or probe selection alone.

## Cost and provenance

Per case: OFF 100 Transformer / 50 native; two LAST oracles 2 native; four signed probe tails 24 Transformer / 16 native; four formal tails 24 Transformer / 16 native; two unit native probes separately counted. Total **148 Transformer / 84 native / 2 unit probes / 0 backward / 5 decode / 5 MP4 save / 20 encode**. Whole fixed run: **296 / 168 / 4 / 0 / 10 / 10 / 40**. Valid skip and zero paths preserve this budget. Failed paths retain attempted/completed counters and missing slots.

Each oracle/unit/probe/formal path records calls separately and synchronized `perf_counter` elapsed seconds during a real run (including work within the recorded path, e.g. diagnostics/persistence). These are actual execution timings only when the user runs the notebook; file timestamps or implementation CPU tests are not GPU runtime estimates. Probe tails and overlapping observations are not independent scientific samples.

Reuse audit: retention.reference and original native continuation preserve full UniPC history; projection_margin/state_clock and the previous terminal_feedback.value define the unchanged carrier/loss. The old identity-feedback candidate moves terminal gradients under an identity approximation; this candidate uses measured finite responses with no gradient. Historical velocity_direction ± diagnostics concern coefficients over steps44–49 and backward without this media experiment; they are not this sign-selection procedure. The old files remain unchanged.

The notebook starts with an independent two-line Drive mount, fetches one published full source SHA, and Run all launches the fixed workflow into a new FlowTubeResponseSelection output directory. It has no mode menu and no automatic execution by the implementing agent.
