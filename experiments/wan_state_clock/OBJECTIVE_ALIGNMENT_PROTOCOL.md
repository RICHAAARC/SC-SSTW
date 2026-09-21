# Frozen terminal objective alignment protocol

Frozen before any candidate calculation. Source baseline: 6922c80a65dfd7480de8760332e0546def681a8f, response-selection method source 9fdfac97126fa681c42423089eb342ce85b676a7. Existing run: flow_tube_response_selection_20260921T013844126172Z.

## Roster and single budget rule

Two existing development cases dev_p0_s0/dev_p1_s0; each OFF, LOCAL_A and LOCAL_B saved terminal latent; both target messages A/B for each start. Fixed 6 saved terminal inputs ×2 target messages =12 paired conditions, each one hinge update and one tanh update =24 updated latents. Classify OFF (4 pairs), same-target LOCAL (4 pairs), cross-target LOCAL (4 pairs) separately. They are correlated diagnostics, not twelve independent videos. Keep zero/nonfinite/missing failures and all fixed slots; no alternative candidates or per-example selection.

For case c and target m, B(c,m)=RMS_support(original LOCAL_m terminal − original OFF terminal). Use the same B for both objectives at every start with that case/target. Multiplier is exactly1; there is no scan, clipping, line search or fallback. B is a conditional budget derived from an existing final-latent residual, not a native T46 D budget or deployable universal parameter.

## Objectives and update

Keep the existing 1760 tube blocks, directions d_b, state×polarity×sync code c_m,b and support z[:,:,1:45]. Let p_b=d_b dot block_b(z).

Old Lhinge,m=0.5 mean_b relu(1−c_m,b p_b)^2.
Only new objective Ltanh,m=−mean_b tanh(p_b)(c_m,b−c_other,b), with temperature exactly1.
Compute a detached terminal-leaf gradient in float64. Let g be that objective's gradient; u=−B g/RMS_support(g), with support outside1:45 zero. Cast u to float32 and add once to the original float32 terminal latent; measure the actual rounded delta, RMS and peak. Zero/nonfinite gradients or invalid budget retain failed slots, never pick another objective/direction/scale. No model, solver, VAE or receiver gradient.

## Predeclared evidence

Primary outcome is the unchanged hard-clipped nominal competitive gap Gclip=mean clip(p,−1,1)c_target − mean clip(p,−1,1)c_other, particularly the four same-target LOCAL conditions. Record both message scores, gap before/after, gain, positive/zero/negative gain and correct/tied/wrong nominal ranking separately. No new success/PASS threshold or posterior best subset. Report paired new-minus-old hard-gap gain including negative cases. Tanh's own improvement is not primary success.

Also report old hinge for both messages, fixed tanh message scores and gap, actual support/global RMS and peak, raw gradient norm, requested budget and rounding error, control direction cosine, and gradient energy fractions on critical blocks (codes differ) versus noncritical blocks (codes equal), interior/saturated/boundary blocks. Saturation uses abs(p)>1, interior<1, boundary==1; additionally distinguish target-correct versus target-wrong saturated critical blocks. Gradient shares use squared L2 energy, not block count. These are explanations only, not third optimization candidates.

No g0-only blind search and no synthetic RGB phases. Existing actual MP4 receiver margins/rankings, when locally available from the same run, are copied as unchanged references only; newly updated terminal latents have no MP4/receiver evidence. No existence/FPR, early trajectory transmission, media robustness, capacity or generalization conclusion follows.

## Historical reuse and provenance

Historical source8b55bce72e661d0705370daff6b4d59d8c0ed57a already implements exact negative nominal hard-clipped competition in main/tube_state/clipped_margin.py: mean(clamp(c_other p)−clamp(c_target p)). This equals negative Gclip. Thus competition itself is not new. The current difference is only fixed tanh smoothing plus same-terminal, same-budget paired diagnosis. Same-code contributions cancel in both competitive forms; old clamp has zero gradient beyond abs(p)>1 while tanh can give finite smoothly decaying gradients. The older experiment differentiates44/45/46 solver coefficients and uses its historical strength roster; this diagnostic does not rerun or directly adjudicate that experiment.

Before reading tensors for candidate calculation, verify each of six tensor SHA256s against its case generation.json, check config hash and fixed source identity. Record source/analysis/protocol hashes and all failures. CPU implementation tests do not constitute new research samples. No GPU, model, VAE, new book, notebook publication or push is authorized in this stage.
