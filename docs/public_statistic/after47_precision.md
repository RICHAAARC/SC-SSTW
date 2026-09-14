# Saved after47 precision diagnostic

Input: after47-diagnostic01/after47_state_and_gradient.pt from this project and run08/config.json. Restore the full scheduler object on its original devices; move only saved latent/target/deltas to CUDA. No prefix rebuild. Both tiers retain the same target and the same previously saved perturbations. Snapshot gradient is used as a comparison reference, never to select a new delta.

Mixed tier: two gradient evaluations, then no-grad baseline, +delta and opposite_delta. FP32 arithmetic tier: promote the currently loaded rounded Transformer parameters and prompt embeddings to FP32, disable TF32/autocast, retain FP32 VAE/solver and original saved history values. Evaluate one gradient, then baseline and the same two candidates. This isolates an arithmetic change on the same rounded weights, not an original FP32 checkpoint or a rebuilt FP32 trajectory. The FP32 tier is not a method acceptance candidate.

Metrics: full gradient files; exact/elementwise differences, relative L2, cosine; each g dot the same delta; grad/no-grad baseline, symmetric directional difference and common loss change. Digests record unchanged original scheduler and fixed inputs. BF16 input changes alone do not describe the FP32 solver direct path. Two repeated gradients support only within-attempt repeatability.

36 transformer calls,9 VAE,3 backward; at most360 block and39 chunk replay;0MP4,1800 seconds wall,zero retry. Device capacity and memory are diagnostic only. No GPU model/version whitelist. CPU tiny Wan/UniPC tests cover fixed inputs, counts, weight-value preservation and TF32 restoration; real GPU results and full-precision memory remain unverified until execution.
