# One terminal-feedback update at fixed T46

Isolated branch `dev/flow-tube-terminal-feedback`, based on retention source `fb356d11d3d68b087bc9a31f9e782e085cbcab88`. The published LAST existence detector is not modified. Implementation/CPU tests only: no real model, GPU, Colab run or historical-data rescoring.

## Paper connection and deliberate differences

[Guidance Watermarking §§4.3–4.5](https://arxiv.org/html/2509.22126v2#S4.SS3) evaluates a decoder loss after completing generation; §4.4 adds transformed-image gradients, while §4.5 replaces backward transport through diffusion by an identity approximation and controls guidance magnitude. This candidate borrows terminal evaluation and identity transport only. It uses the existing differentiable latent tube hinge surrogate, not the paper's VAE/image decoder, log-cosine objective, EOT, PCGrad or clipping rule. It is not a reproduction of the paper or evidence for its image-level claims.

## Frozen experiment

Two existing development cases: dev_p0_s0 and dev_p1_s0; both fixed messages. Exactly five media arms per case: OFF, LOCAL_A/B, TERMINAL_FEEDBACK_A/B. No parameter search, repeated update or per-video arm/time selection. Fixed46 was chosen before this candidate's run because the prior retention run exposed both an incorrect T46 message and a negative relative-OFF margin gain, while requiring a short three-step free tail. These are reused development cases, not holdout.

One new OFF generation supplies states and complete UniPC histories. New LAST49 A/B native responses give per-message target R=RMS_support(D49). LAST is an auxiliary future-information oracle, recorded with two native calls per case; it has no video arm and no new media validation. This budget is a mechanism diagnostic, not an online strength rule or equal-perceptual-quality comparison.

From the stored T46 input/CFG velocity, one separate no_grad preview really runs native46 and the free47..49 tail, using a deep copy of all solver history. Both messages share this uncontrolled preview. Its terminal is saved and compared with OFF by numerical_equivalence maxabs with atol/rtol tolerance. The runner separately records support_relative_rms_error = RMS_support(preview−OFF)/max(RMS_support(OFF),1e-30); that diagnostic does not define the equivalence pass rule. The OFF terminal is not substituted for preview computation. Formal controlled updates each start from a fresh copy of the original T46 history, never preview-mutated history.

## Objective and transport

The fixed tube basis and state×polarity×sync signs are unchanged. For b=1..1760, p_b(z)=d_b·block_b(z), L_m(z)=0.5 mean_b relu(1-c_mb p_b(z))². Its loss matches half the existing projection_record loss. Autograd operates only on a detached float64 terminal leaf; the returned gradient g is float32. No Transformer, solver tail, VAE or blind receiver graph is retained or differentiated. CPU tests compare both the tensor layout and analytical coefficients against the original NumPy tube carrier.

LOCAL uses the existing full projection residual P(clean46)−clean46. TERMINAL_FEEDBACK uses −g. Normalize either direction to unit support RMS before a single same-history native probe; zero/nonfinite directions remain failures. For unit direction q, set delta_v=−q/sigma46 and measure its actual native response D_unit. Apply u=(R/RMS_support(D_unit))*q, update velocity v−u/sigma46 once, then continue native47..49 freely. Thus the feedback negative gradient corresponds to a positive g/sigma velocity correction before the scalar normalization/budget factors. All actual u, delta_v, D, raw/unit norms, scales, matching errors and tensors are recorded. Float32 response matching is measured, not assumed exact.

The main first-order prediction is −g·D_actual under the explicit approximation delta_terminal≈delta_next_state. The auxiliary −g·u assumes a clean-control displacement and is labelled separately; u is not D. Neither prediction proves real terminal improvement. Report actual L(OFF)−L(formal terminal), then compare L(LOCAL)−L(FEEDBACK) for the same case/message. These two approximations and the true outcome must not share a label. CPU linear-model native-tail tests require actual objective descent, not just a normalized gradient.

## Costs and evidence boundary

Per case: OFF100 Transformer/50 native; LAST oracle2 native; preview6 Transformer/4 native; four formal arms24 Transformer/16 native plus4 unit native probes; two terminal-leaf backward operations. Totals:130 Transformer,72 native (including4 preview),4 separately counted unit probes,2 leaf backward,5 VAE decode/MP4 saves,20 receiver encodes. Whole experiment:260/144/8/4/10/10/40 respectively. calls_by_path distinguishes preview, oracle and each formal arm. No wall-time or hardware model is inferred from timestamps.

All ten actual media arms use the original complete MP4 save/readback, four real VAE phases and five blind reader modes. No terminal-loss screen chooses which videos proceed. Failures retain ten video/forty observation slots. Terminal and media pair completeness are separate, with four fixed FEEDBACK-vs-LOCAL case/message rows. Main MP4 comparison is local_state correct-minus-other margin and gain over LOCAL/OFF; other modes are diagnostic. Truth joins only after blind read. OFF ranking is not FPR, and this protocol adds no existence threshold.

Quality remains same-case OFF RGB PSNR and temporal residual diagnostics plus human inspection of content, artifacts and motion. It is not a perceptual pass. Positive terminal prediction alone is not trajectory success; even positive true terminal loss improvement requires complete MP4 evidence. No independent generalization or capacity result is claimed.

Notebook `flow_tube_terminal_feedback_colab.ipynb` has the independent two-line Drive mount and a fixed Run all workflow. Publication binds an immutable source SHA. Output is a new FlowTubeTerminalFeedback run, with source ZIP, initial/prompt/negative tensors, OFF nodes/histories, preview, gradients, control tensors, all formal terminals, media and readout records. No prior input tensors are silently reused.
