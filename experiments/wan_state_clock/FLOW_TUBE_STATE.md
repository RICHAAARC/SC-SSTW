# Flow tube-state local candidate

Base: `3f0a5fafa7c2aa56fbc69bed17649accf49ae152`; branch `dev/flow-tube-state`.
Published runtime source: `fca6f1f4a447da8f3b725425f0db960541cb6a74`.
This is an engineering implementation, not a GPU result.

The original state-clock codebook, normalized coordinates, 1760 support blocks,
receiver and synchronization remain unchanged. At indices 44/45/46 of 50,
the writer predicts `z0hat = sample - sigma * CFG_velocity`, using the actual
flow-prediction UniPC sigma, and requests
`u_b = relu(1-c_b*p_b)*c_b*d_b/(3*dot(d_b,d_b))`.
The velocity correction is `-u/sigma`. No transformer or VAE backward occurs.
Steps 47/48/49 continue normally, retaining the controlled scheduler history.

The new continuation computes CFG with the original model-output dtype, then
converts the completed velocity to FP32 for every scheduler call, including the
shared prefix and OFF. This avoids a BF16-to-FP32 arithmetic change only in the
controlled branch, which invalidates affine shadow-step scaling. The result
records both dtypes. The historical `generate_terminal_latent` default remains
unchanged. This OFF is the matched new-run control, not an assertion of bitwise
equivalence to a prior BF16-scheduler run.

OFF finishes first. The existing terminal write produces terminal A/B and fixes
`R=max(RMS_support(terminal_A-OFF), RMS_support(terminal_B-OFF))` before either
Flow starts. A/B share R. Both predicted correction u and actual same-history
one-step D are limited to R/3 per active step and sum of RMS at most R. Two
deep-copied shadow schedulers reuse the already computed velocity; only the
final scaled velocity updates the live scheduler. Numerical affine error and
roundoff tolerance are recorded, alongside requested/applied tensors and both
support/global RMS. Net terminal offsets are separate from cumulative budgets.

Five rows are always retained: OFF, TERMINAL_A, TERMINAL_B, FLOW_A, FLOW_B.
The full plan is 124 Transformer forwards, 62 real plus 12 shadow scheduler
steps, five VAE decodes and normal H264 CRF18/yuv420p saves, and 20 receiver
VAE encodes. Each saved MP4 supplies its four actual receiver origins; no
latent-cropping substitute is used. Truth is joined only after blind read.
Each Flow and terminal message must be uniquely correct under all five existing
reader methods. OFF only reports rankings; it cannot establish FPR.

Precodec and saved quality include RGB MSE/PSNR, temporal difference energy
ratio, and residual temporal MSE `mean(((X[t+1]-O[t+1])-(X[t]-O[t]))**2)`.
The fixed first-round saved quality allowance is 1.5 times the corresponding
terminal MSE and residual temporal MSE; it is not a perceptual threshold.
Projection/loss are diagnostic only. Failures/OOM remain engineering failures,
with attempted/completed counts; a completed negative readout/quality result
is retained without changing parameters. CPU/GPU peak memory and elapsed time
are reported for real user execution, not inferred from tensor sizes.

Local notebook: `notebooks/flow_tube_state_colab.ipynb`. Its first code cell is
the exact independent Drive mount. Run all fetches the complete immutable
runtime source SHA above from `https://github.com/RICHAAARC/SC-SSTW.git`, checks
out that commit, and launches the fixed five-condition run, persisting that
source archive, progress and results. There is no mode menu or branch-tip
source resolution. The earlier self-contained local notebook remains available
in the runtime source commit as a historical backup; the current notebook is
the formal published entry. Publication does not imply a real run, model
download, GPU execution or Drive write by the implementation agent.

Regenerate the notebook after any source change:
`python scripts/build_flow_notebook.py`.
Run the necessary CPU/fake/static checks:
`python -m pytest tests/test_flow_tube_state.py -q`.
No model download is needed for these checks; they use installed CPU torch and
the real Diffusers UniPC plus a fake model/media adapter for failure handling.

Old G1 used index-5 intermediate-VAE observer gradients and a short continuation;
its eight saved MP4s failed trajectory fit and quality. This candidate revises
the objective and propagation approximation; it does not erase that failure.
Terminal evidence does not prove Flow propagation. Even a successful real run
would support only this content/seed's initial feasibility.
