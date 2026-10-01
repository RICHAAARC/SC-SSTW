# Saved-terminal channel margin control V1 — local proposal

This is one terminal controllability diagnostic, not a new completed trajectory writer. The fixed source is OVERLAP_MULTI from 20260930T113335113338Z, full terminal SHA256 5c96d514454cd1b23e700f0072b4a40c073e8a038269962743904fb6189c5164. The writer saved the full [1,16,46,40,64] FP32 terminal separately from its small projected sidecars. No generation replay is needed. The YUV444 reference is 20261001T110124499275Z, with all OFF/PAYLOAD_MULTI/OVERLAP_MULTI controls retained as existing observations.

## Fixed single candidate

Keep the 8-dimensional state code, four zero-mean spatial bases, overlapping ages, original 1392-dimensional write support, all R44 receiver coordinates, all 174 valid paths and their exact template equivalence. The detector is unchanged and receives only normalized received data, key and full received availability. It does not receive the original terminal, target time sequence, payload truth, update loss or source-to-video correspondence. This diagnostic itself fixes phase0; it does not claim unknown-phase synchronization.

The writer knows its intended zero-edit time sequence. Write q for the received projection, mu0 for that template, and mui for each distinct wrong template. M=1408. Define Delta_i=J_i-J_0=mean((mui-mu0)*(mui+mu0-2*q)); shared observation energy cancels. Define the public template separation d_i=mean((mui-mu0)^2). The proposed writer loss is:

    global = relu(0.5*min_i d_i - min_i Delta_i)
    local = mean_i relu(0.5*d_i - Delta_i)
    loss = global + local

There are 173 wrong classes with full R44 support. Equal-template coordinates cancel in each local contrast, leaving that candidate's differing time/block/age positions; the common receiver denominator remains 1408. This covers all valid starts, repeats and skips, not selected historical winners. All targets come from public templates, with no data-fitted margin or detection threshold. Half-template separation and the sum of the two terms are explicit new proposal semantics, not a previously validated loss.

Take exactly one projected descent step: project d(loss)/d(terminal) onto the original active pilot basis; delta=-174*projected_gradient, shrink only if its full tensor L2 exceeds1. No scan or retry and no minimum norm amplification. Although the numerical eta/cap and support are inherited, the objective and terminal location changed; this is not a matched generation budget or the same physical effect as conditional-clean control. Preserve all positions including terminal row45; detection uses all original regular1..44.

## Actual channel and derivative

1. Load the complete saved writer terminal; native FP32 VAE decode, normal clamp and RGB8 quantization.
2. Execute the exact two existing FFmpeg processes into real YUV444 and reopened RGB24 files. Do not approximate the forward color conversion.
3. Encode those actual RGB bytes with the frozen native causal VAE, posterior mode and original normalization. Read/persist the unchanged receiver outputs before joining truth. Backpropagate the writer loss to the received RGB.
4. Release encoder graph/storage. Replay native decode from the same full terminal and use the received RGB cotangent. The RGB8/color roundtrip uses an identity straight-through derivative only. This is a direction proxy, not the derivative of discrete quantization. Native VAE encode/decode are differentiable and causal cache history is preserved.
5. Project/cap once, save the changed terminal, then independently execute decode→RGB8→real YUV444→RGB24→native VAE encode again. Compare actual Delta, all-candidate rank, payload and full-frame quality against the freshly measured BEFORE observation. Baseline difference from the historical received tensor is reported separately.

The native encoder/decoder are temporarily wrapped on that VAE instance for functional cache checkpointing. A cache tensor is passed as a differentiable checkpoint boundary, not detached. Each phase is run separately, and existing chunked disk storage is reused. There is no blanket80GiB-free preflight and no GPU-model gate: actual writes respect the existing80GiB per-phase limit and2GiB free margin. Full-size resource feasibility remains unverified; a failed phase retains all earlier evidence and does not automatically reduce resolution, frames or gradients. CPU is supported but much slower. No Transformer is loaded.

## Fixed accounting and interpretation

New: BEFORE and AFTER, each two keys, four path reads, four repeated-payload reads, eight message comparisons,696 valid costs,15660 catalog slots and14964 structural exclusions. Retained references:3 normalized tensors and6 raw reads, including all prior controls. They are not new independent negatives.

Planned logical operations:2 native no-gradient decodes,1 checkpoint decode plus its VJP,1 checkpoint encode plus its VJP,1 native encode; two real color roundtrips (four FFmpeg processes). Each checkpoint phase uses46 forward and46 recomputed causal chunks. One update, no Transformer, no diffusion replay, no H264. Count attempts/completions separately and retain missing slots. New raw/latent/update files are separate from all source media.

Improvement is reported numerically; execution completion and positive Delta are not a quality or science PASS. A candidate can improve loss but worsen real Delta, payload or quality; retain it as a failed direction. If real-channel Delta improves at acceptable cost, only then propose connecting the same objective to MULTI generation and evaluating its actual tail/MP4 output. This diagnostic must not be relabeled generation-time watermarking. H264 robustness, unknown phase, unknown snippets and independent confirmation remain separate work.

## Delivery status

Local proposal, not pushed and not run with pretrained weights. CPU tests cover all-family margin arithmetic and finite-difference gradients, the original pilot support/cap, native tiny Wan causal checkpoint values/gradients, split-VJP agreement, restoration after errors, raw-before-truth persistence and draft notebook structure. Tiny native Wan tests use random local weights and cannot establish full-model resource feasibility or method success. Published SHA binding and user Run-all are pending.
