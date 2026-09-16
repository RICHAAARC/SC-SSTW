# True terminal velocity-coefficient direction diagnostic

Implementation base: `5736501151a0e454857e4e503e7008b6b6b27467` in the existing
`dev/flow-tube-state` worktree. The notebook checks out published runtime source
`ff474c712e7d3b5a5874a24c61f77e94d9bfc59c`;
publication does not constitute a real model run.
The prior five-condition `flow_run` entry and published notebook are unchanged.
`prepare_generation` adds `load_vae=False`; its existing default remains True.
The new entry obtains the same prompt embeddings and one 0–43 prefix directly,
so it does not assume an old latent/scheduler snapshot includes embeddings.

## Parameterization and true terminal loss

At indices 44,45,46 only, `v_ctrl = v_CFG.float() + scatter(a_i,b*d_b)`.
There are 3×1760=5280 coefficients, initially zero. Blocks are the original
non-overlapping T,C,H,W flattened supports, with original directions/state codes.
The model CFG computation retains its original BF16 arithmetic. Its completed
velocity and injection are FP32. Coordinates are normalized latent/scheduler
coordinates; no VAE std/mean enters the injection. There is no `-u/sigma` control.

`F(a)` runs all six real Transformer/CFG/UniPC steps 44–49, including history
and the last three uncontrolled steps. The zero prefix is detached; the live
tail sample, converted model_outputs, corrected last_sample and all other
history tensors retain their graph. Independent runs start from detached
copies of the same pre-intervention state. Detached shadow snapshots never
replace live state and preserve Diffusers FrozenDict configuration types.

The loss is `mean_b relu(1-c_b*dot(B_b(F(a)),d_b))^2`. Torch uses the original
block order and float64 projection accumulation while retaining gradients to
FP32 coefficients. Both messages use separate `a=0` AD passes. The direction
is the coefficient Euclidean negative gradient `q=-g/||g||2`, stored in FP32
and checked for unit norm. A zero/nonfinite gradient leaves missing direction
rows; it never produces a guessed fallback direction.

## Single predetermined probe amplitude

ZERO_A's actual terminal is OFF. Existing terminal writes for messages A/B fix
one common `R=max_m RMS_support(terminal_write_m(OFF)-OFF)`. No extra OFF tail,
VAE or MP4 is needed. Both writes must be available to define R.

Let `h_i` be the actual UniPC step coefficient multiplying the *current* velocity
at fixed sample/history. Native non-thresholded UniPC with no external `solver_p`
is coordinatewise affine. A scalarized snapshot preserves the real sigmas,
step_index, lower_order_nums, this_order, timestep_list, model_outputs presence,
and last_sample/corrector presence. Subtracting its actual `step(v=0)` from
`step(v=1)` yields `h_i`, including corrector and predictor effects. Zero-valued
scalar history is valid because only affine offsets depend on history values.
Six scalar scheduler probes obtain three coefficients; they are not Transformer
calls. CPU checks compare them with full-tensor random same-history differences,
including a subsequent live history containing non-leaf gradient tensors.

With `N=1760*1024` support coordinates:

```
s_i  = sqrt(sum_b q_i,b^2 * ||d_b||^2 / N)
kU_i = abs(sigma_i) * s_i
kD_i = abs(h_i) * s_i
epsilon_cap = min_i(R/(3*kU_i), R/(3*kD_i),
                    R/sum_i(kU_i), R/sum_i(kD_i))
epsilon_probe = rho * (1-0.001) * epsilon_cap, rho = 0.1
```

Zero factors impose no constraint. The cap is a budget bound, not a derivative
scale. `rho=0.1` is a fixed engineering probing convention, with no guarantee of
locality or optimality; 0.001 reserves floating-point room. Each message has one
q and one epsilon, with exactly opposite FP32 coefficient arrays. There is no
scan, line search, clipping or repeated shrinkage. Actual coefficient norms,
antisymmetry and multiplication-rounding error are recorded.

At each active step, FP32 scatter produces requested_delta_v. The actual
FP32 addition produces `effective_delta_v=(v+requested_delta_v)-v`. Both tensors,
addition-rounding RMS and the fraction of nonzero requested coordinates swallowed
by addition are recorded. `U=-sigma*effective_delta_v`
is only the frozen-sample/velocity predicted-endpoint *budget quantity*, not
the true final effect. `D=real_step(v+delta_v)-shadow_step(v)` uses the same
controlled history/sample/base velocity. A second detached shadow checks the
controlled step, while the real step retains its graph. U and D each obey
support RMS ≤R/3 per step and sum of RMS ≤R. The code records actual quantities,
their cumulative sums, global/support RMS, and `D-h_i*effective_delta_v` residual.
Budget excess is flagged and retained, never independently clipped for +/-.
Numerical affine-check tolerances are recorded as engineering diagnostics.
Opposite coefficient arrays do not imply exactly opposite effective velocity
increments on the two different controlled trajectories; no such symmetry is
assumed or claimed. The cap is derived from the requested direction; actual
effective increments and their budgets are measured afterwards without retry.

## Checkpoint and execution costs

The checkpoint strategy is nested and non-reentrant: the existing outer pure
Transformer checkpoint remains, while the official Wan
`enable_gradient_checkpointing` hook checkpoints each internal block. Neither
checkpoint encloses CFG, coefficient injection, loss, or a solver update. First
step model inputs are independent of a; the remaining ten Transformer calls per
zero pass have gradient-bearing inputs. Wan 0.39/0.40 activates the internal
path on `torch.is_grad_enabled() and self.gradient_checkpointing`, including eval
with frozen parameters. Flags and the prior checkpoint function are restored
on ordinary return, early-stop replay, and failure. Other runtime entrypoints
retain their existing defaults.

The outer checkpoint prevents block boundary tensors from accumulating across
ten calls. During backward it reconstructs the current invocation's block
boundaries; inner checkpoints replay individual blocks. Those one-call boundaries,
one block's activations, model weights, and the live solver graph still consume
memory. This is not a guarantee that the full model fits L4. CPU weak references
verify that original block inputs do not survive ten completed forwards and
that replay inputs are released after backward or an injected failure. Scalar
counters and formatted traceback strings retain no graph or exception objects.

The older `Public-Statistic-Control` Transformer implementation informed use of
the official block hook only; no VAE cache/offload code, allocator quota, GPU gate,
or historical replay hard limit is imported. The previous user's run completed
the two zero forwards but both outer replays OOMed before a completed backward;
this repair still requires a new user-run GPU measurement.


| Operation | Fixed complete plan |
|---|---:|
| Shared prefix Transformer forwards | 88 |
| Six real tails Transformer forwards | 72 |
| Backward passes | 2 |
| Outer pure-Transformer replay invocations | Up to 20, separately attempted/completed; may early-stop |
| Block forward inside gradient-bearing original calls | 20 × actual model block count across both zeros |
| Block forward while outer replay reconstructs boundaries | Up to 20 × block count; separately attempted/completed |
| Inner block backward replay | Up to 20 × block count; separately attempted/completed; may early-stop |
| Live scheduler steps | 44+36=80 |
| Detached full-tensor budget shadow steps | 36 |
| Scalar response probe steps | 6 |
| VAE encode/decode/backward and MP4 | 0 |

There are exactly six rows: ZERO_A, ZERO_B, PLUS_A, MINUS_A, PLUS_B, MINUS_B.
No tail failure suppresses another valid tail; missing directions stay explicit.
Block units are never added to Transformer invocation totals. Block counters are
accumulated in memory and persisted at outer-call/stage/failure boundaries.
Each row keeps counts, elapsed time, process high-water RSS, stage CUDA peaks,
actual endpoint projections/loss, correct-code signed absolute projection gain,
competitor margins, per-step U/D and final OFF difference. Raw tensors and
effective config/source are persisted for interpretation. Resource snapshots at
forward completion, before/after backward, failure (while traceback is live),
condition-finally cleanup, and after condition return distinguish current CUDA
allocated/reserved bytes from stage peaks. The final return sample occurs after
the exception function frame exits. Full formatted tracebacks preserve the failing
operator without retaining exception objects. Two zero forward
results define only this pair's observed baseline difference. Missing zero
evidence stays null, never synthetic zero.

The report compares `g dot q` with `(L(+eps*q)-L(-eps*q))/(2*eps)`, plus both
loss changes, projection gains and the observed zero floor. BF16 AD is not a
derivative of the discrete rounded sampler. Sign agreement is local evidence,
not a scientific PASS, guaranteed writing mechanism, curvature statement,
saved-video result or payload/receiver extension. Weak/below-floor/nonlinear
results remain uncertain; no automatic follow-up search is run.

## SHA-pinned user-run handoff

`notebooks/velocity_direction_colab.ipynb` fetches the full immutable source SHA
`ff474c712e7d3b5a5874a24c61f77e94d9bfc59c` from GitHub and verifies its detached checkout. Open this notebook in Colab and Run all; the first cell is
the exact independent two-line Drive mount, without force_remount. It creates a
fresh source directory. Result files are saved in
`/content/drive/MyDrive/Video-WM/VelocityDirection/<unique-run-id>/`.
The source ZIP and launcher log are sibling files in `VelocityDirection/`, named
`<unique-run-id>.source.zip` and `<unique-run-id>.launcher.log`, respectively.
They can therefore preserve source/log evidence even if result-directory
creation or runner setup fails. This layout matches the existing launcher.
The effective configuration and `result.json` also record these source/log
paths, the checked-out source directory and the actual CLI result/config paths.
The launcher reads the tracked `configs/generate_replication.json` and applies
the same direction role and `VelocityDirection` output parent, recording the
actual source commit and repository URL. It does not depend on a tracked
`velocity_direction.json`. The source ZIP is produced by `git archive` from
the pinned commit; the effective configuration is saved separately by the runner.
The notebook delivery commit is separate from this runtime source commit.

Rebuild with `python scripts/build_velocity_direction_notebook.py` and check
with `python -m pytest tests/test_velocity_direction.py -q -k notebook`. Method tests use
CPU torch, real UniPC and a deterministic fake Transformer; no pretrained model,
GPU, Colab or Drive execution has occurred. Source/notebook publication is separate from real model/GPU execution.

## Checkpoint repair validation ceiling

`tests/test_velocity_checkpointing.py` uses randomly initialized small real Wan
Transformers and real UniPC. It checks the complete six-step production tail,
all 5280 coefficients and the original terminal loss in FP32 and BF16 (with native
FP32 parameter exceptions): native, outer-only, and nested endpoints/gradients
agree. Solver update counts remain six after backward; all three coefficient
rows carry gradients. These are CPU architecture checks, not pretrained outputs.

A separate ten-call tiny-Wan saved-tensor inventory excludes model weights and
uses storage identity deduplication. Outer-only and nested retain the same
6,120 logical bytes / 5,220 unique storage bytes after forward; block-only retains
65,280 logical / 53,760 unique bytes. These measurements describe this CPU fixture,
not live allocator peaks or predictions for the user's GPU. C++ weak-storage
handles (including aliases with different tensor wrappers) observe block-input
storage peaks of 1,024 bytes in forward and 3,072 bytes in replay for a separate
three-block tiny fixture; both inventories are empty after backward/failure.
These observations cover block inputs, not all internal activations. Weak-reference checks
also cover replay-boundary release, exception restoration, and a subsequent
independent gradient call. Runner failure injection retains all six conditions,
missing directions, full traceback strings and post-return resource snapshots.

Run the bounded validation with
`python -m pytest tests/test_velocity_checkpointing.py tests/test_velocity_direction.py -q`.
No full model, GPU, Colab or Drive experiment was executed for this repair.

Source inspection: [Wan v0.40.0 block activation](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/transformer_wan.py#L634)
and [PyTorch v2.11.0 nested checkpoint semantics](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/checkpoint.py#L563).
