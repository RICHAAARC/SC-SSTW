# B paired-transfer diagnostic: adopted user-run recipe

Status: the user adopted the historical ALLBIT objective, a=.10 terminal/CFG-clean amplitude (.02 conditional-clean at CFG5), and the complete budget below on 2026-10-15, and authorized publication on the existing dev/local-joint-state-payload-v1 branch. The local package has passed bounded A2/A3/A5 review. Real GPU execution is performed by the user and had not occurred at publication. Run all directly executes this fixed recipe.

The question is whether a receiver-improving terminal direction survives injection through the existing conditional-clean interface and the real final UniPC update. This is one existing source, one finite-step comparison, not a new multi-step joint method or a full32 prerequisite.

## Source and method

Old B saves terminal tensors only; it does not save a usable native prefix with complete UniPC history. Old BASE is JOINT and rebuilding it would require its posterior provider and VAE encodes. This recipe instead uses the original yellow-sailboat prompt/negative prompt, seed2026100701,181x320x512,50 native steps,CFG5 with explicit ZERO control at all steps. ZERO is a newly generated control endpoint, not historical JOINT BASE or S, and is not a new independent source.

A fresh VAE VJP at this ZERO terminal uses the historical ALLBIT objective
L=.8 mean32(H_.02(m))+.2 H_.01(trueG).
All q, key, message8001a55a, rho.5, original local latent mask and22-state readout remain unchanged. Set d=-P grad(L)/||P grad(L)|| once. Active max ties and zero gradient keep the completed ZERO forward and mark dependent arms missing. No candidate selection, scan, projection, extra optimization or blind-path change.

Adopted a=.10 is a terminal-normalized-latent/CFG-clean coordinate:
- TERMINAL=FP32(z_ZERO+a*d).
- At update48 (zero-based penultimate), conditional-clean operand=FP32(a*d/5)=.02*d.
- Conditional velocity changes by -delta_c/sigma48; CFG velocity changes by -5*delta_c/sigma48.
- NATIVE executes the actual UniPC update48 with original full history, then reevaluates both DiT branches on the changed state and executes the actual last update49.

The cached conditional/unconditional predictions at48 come from the identical ZERO input and pre48 history. There are no extra48 predictions. The corrector, model_outputs, last_sample, this_order, lower_order_nums, sigma/timestep sequence and cursor are all saved/restored, never cold-started. The native paired48 effect and final native displacement are measured against ZERO after48 and ZERO terminal separately. Intermediate conditional-clean, CFG-clean and velocity receipts retain shape, L2, RMS, maximum absolute value and nonzero count, distinguishing nominal and realized FP32 values where recorded. Directional projection, orthogonal L2 and outside-mask L2 are saved by method.displacement only for direct_terminal, paired_step48 and final_native effects. Native displacement is not assumed equal to.10. Historical per-step cap1 and terminal ball1 are not equivalent; neither is introduced as a projection here.

## Cost and residency

Nominal uninterrupted run: one generation preparation (pipeline/transformer and text encoder load, one positive+negative prompt encoding),100 baseline DiT calls +2 final-tail calls=102 ordinary DiT calls. 50+2=52 scheduler updates. No DiT gradient/VJP/recompute, posterior lift, VAE encode or codec.

Transformer is offloaded to CPU for fresh terminal readout. VAE load1 supplies one gradient decode/VJP and TERMINAL ordinary decode; VAE is then released and transformer returns for the native tail. Transformer offloads again and VAE load2 supplies NATIVE ordinary decode. Three VAE decodes total:46 gradient forward chunks +92 ordinary chunks=138 forward chunks and46 nominal checkpoint replay chunks. Completed ZERO gradient RGB is reused, not decoded again. Saved tensor context is used after a resume; generation preparation can repeat on resume and its extra load/prompt preparation is counted.

Three endpoint readers retain264 windows,4224 chips,165 metric values (all32 payload margins/bits and22 correlations/trueG per endpoint). The gradient reader adds88/1408/55 reader computations using the same ZERO physical forward, not an extra decode or independent endpoint. Two quality pairs compare TERMINAL and NATIVE with ZERO, retaining181 frame and180 temporal rows, actual peak neighborhoods and images without a quality threshold.

The successful VAE checkpoint path uses CPU/disk boundary spooling, not full live-graph CPU offload. Prior real measurements for this geometry were roughly64.53GiB temporary spool; this is a planning estimate, not a new-run measurement or gate. Runtime receipts include actual disk writes/reads/peak, transfer buffers, physical forward/replay chunks, persisted storage bytes and attempted/completed/current/cumulative calls. CPU must also hold the offloaded transformer, saved contexts, full RGB and latent/history copies. Runtime memory and duration are unmeasured before a user run. No GPU-name or source/version identity gate.

## Durable execution and outputs

Every completed baseline/tail native step atomically saves complete history plus latent, retaining latest and previous. fork48 saves both DiT branches and full pre48 history; zero_after48 provides the true paired reference. A gradient unit atomically contains input terminal, ZERO RGB, q/readout and raw gradient/direction; unfinished autograd graphs are recomputed only on a later user resume. Immediate commits plus60-second JSON snapshots and independent call/failure receipts retain partial work. No automatic retry in an attempt. A resumed tail can start at49, or materialize its already-completed50 endpoint. A complete task repairs reports/images without loading models or rescoring committed readers. Missing committed evidence is reported, not silently regenerated. Same-Drive Run all recovery is CPU/fake tested; real cross-Colab recovery remains untested.

Outputs include result.json/comparison.json, all fixed endpoint rows, native/ snapshots and transport.json, units/gradient.pt, terminal tensors, raw.json before truth scoring, q.pt/metrics.json/readout.json, quality pairs/images, calls/,failures/,checkpoints/,progress.json. ZERO float RGB is inside units/gradient.pt, avoiding a duplicate large artifact. If VJP fails or is interrupted, its retained forward artifact explicitly says FORWARD_ONLY_VJP_FAILED or FORWARD_ONLY_VJP_INTERRUPTED and gradient_complete=false. Physical call counters retain attempted, completed, failed, interrupted_completion_unknown and running separately, both per attempt and cumulatively.

A durable task resumes using its saved effective method/source parameters; a newly supplied config is recorded as requested_config and does not relabel the saved context. The fixed original prompt, negative prompt, token limit, seed, geometry and Wan family are checked as source/method semantics. Missing quality images are redrawn from saved RGB using existing quality.json without recomputing quality scores; missing committed quality metrics are reported without replacement scoring.

Chunk counters retain attempted/completed chunk quantities. Their difference alone does not imply failure: the enclosing decode receipt is authoritative for forward/ordinary outcomes, and the same-attempt decoder_vjp receipt for replay outcomes. Physical call aggregates separately retain known failure and completion-unknown interruption.

Three bounded outcomes use the signed change in this one fixed loss:
1. TERMINAL improves but NATIVE does not: reject simple transport for this construction/point.
2. Both improve: limited support for this construction only.
3. TERMINAL finite step does not improve: cannot attribute failure to the tail.
Missing/undefined evidence remains unresolved; near-zero changes are descriptive and require numerical interpretation, not an invented significance/PASS threshold. Full32, robustness, generalization and joint multi-step success are not inferred.

## Entry points

Run the standalone module with --config experiments/wan_state_clock/configs/local_joint_paired_transfer_v1.json --output /chosen/new-or-resume/task.
Notebook: notebooks/local_joint_paired_transfer_v1_colab.ipynb.
Builder: scripts/build_local_joint_paired_transfer_v1_notebook.py.
Companion: notebooks/local_joint_paired_transfer_v1_portable_source.zip.

The notebook loads the companion ZIP directly from the editable dev/local-joint-state-payload-v1 branch of RICHAAARC/SC-SSTW. No hashes, expected digests, manifests, fixed source reference or exact dependency version gates are used. Publication is confined to these ten paired-transfer files on that dev branch; it does not merge main or execute the GPU experiment.

Open in Colab: https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/local_joint_paired_transfer_v1_colab.ipynb

Run all in the user's GPU runtime. The fixed Drive task directory supports later Run-all resume; return the whole task directory, including failed and missing evidence. Actual Wan behavior, resource use and scientific results remain unmeasured until that user run. Earlier review records saying the recipe was pending adoption describe the earlier local-preparation stage.
