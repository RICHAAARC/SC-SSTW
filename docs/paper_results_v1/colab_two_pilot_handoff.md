# Fixed two-pilot Colab handoff

[`paper_results_v1_two_pilot_colab.ipynb`](../../notebooks/paper_results_v1_two_pilot_colab.ipynb)
is the single-file Colab handoff for the adopted `pilot_01` and `pilot_02`
engineering confirmation. Its first code cell is only the two-line Drive mount.
Run all has no mode selector and attempts the two pilots once in the fixed order:

1. extract and content-verify the embedded no-`.git` source closure;
2. write the complete ten-case fixed plan and the two-pilot execution scope;
3. fetch exact baseline source commits and named checkpoint objects, retaining
   each preparation failure independently;
4. bind actual local paths and downloaded file digests into one immutable
   effective config and initialize the complete RunStore;
5. prepare the recorded main environment, isolated baseline dependencies, and
   fixed Hugging Face revisions;
6. run `generate`, `decode`, `framewise`, both baseline embeds, the shared codec,
   quality, both edit-aware baseline extracts, receiver synchronization, and
   receiver reads in separate processes for each pilot;
7. run report-only `evaluate` regardless of phase failures and write a compact
   handoff summary.

The eight confirmation cases stay in the manifest and full left-joined report
as planned rows. The notebook never invokes them and labels them
`PLANNED_NOT_EXECUTED_BY_THIS_NOTEBOOK`. Pilot results remain excluded from the
confirmation cohort. A failed phase is not retried in the same run directory;
all later phases are still attempted once so that missing dependencies and
blocked artifacts remain visible in the fixed rows. VideoSeal and RivaGAN
preparation failures do not stop the main method phases.

## Source and model identities

The notebook does not fetch the unpublished project branch. It embeds
`experiments/paper_results_v1`, `runtime/wan`, and `main/tube_state` with a
deterministic ZIP digest and per-file SHA-256 manifest. This content manifest,
not the contextual build commit, is the execution source identity.

The fixed external identities are:

- VideoSeal source commit
  `870ca7fb33578b90f14c602016b6c2788096226e`, official
  `videoseal_1.0.yaml`, and named checkpoint object
  `https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth`;
- DAI-Lab RivaGAN source commit
  `efffa72a4ca46d4d5051f6970c96424c2cdab441` and the explicitly disclosed
  Peachypie98 community 32-bit checkpoint at repository commit
  `4d9928350509f808b6b57d48d2f958aef811d332`;
- Wan revision `0fad780a534b6463e45facd96134c9f345acfa5b` and
  framewise VAE revision `31f26fdeee1355a5c34592e401dd41e45d25a493`.

The notebook computes and records actual card/checkpoint SHA-256 values after
download. It does not invent an unpublished checkpoint digest. Baseline source
and weights being present is only preparation evidence; successful model/API
compatibility requires the later embed/extract phase receipts. VideoSeal's
official dependency set and RivaGAN's legacy dependency pins are not installed
over the successful main stack. Each baseline gets a separate
`--system-site-packages` virtual environment, with only its required interface
packages added. The obsolete RivaGAN requirements file is never installed.

Historical main-chain real evidence belongs to run
`20261008T003135066923Z/fixed_reference` and source
`ac111d0fed253767651929d115c343fe1636c525`. It closed the trajectory, M05,
MP4, blind correction, and Wan read within that old run. The current published
conditional notebook is bound to a different CPU/fake source, so neither the
old run nor that notebook proves this new handoff executed. The archived
VideoSeal helper and baseline preparation code also have no matching real-run
receipt. The notebook records these boundaries in `handoff_summary.json`.

## Fixed work and storage

For two pilots, the configured attempt contains:

- four 50-step trajectories: 200 scheduler steps and 400 conditional plus
  unconditional transformer forwards;
- two shared framewise writer encodes and four framewise decodes;
- twelve full-video codec roundtrips across the four main arms and two external
  baselines;
- 72 receiver synchronization encodes, 320 logical Wan reads, and at most 240
  physical Wan receiver encodes. The last value is an upper bound before alias
  reuse, not a measured call count;
- 320 receiver rows, 36 baseline rows, 36 adopted comparison rows, 14 quality
  rows, and 22 cost rows in the attempted two-pilot scope.

The complete ten-case plan remains 690 artifacts, 1,600 receiver rows, 180
baseline rows, 180 comparison rows, 70 quality rows, and 110 cost rows.

One `181 x 320 x 512 x 3` RGB8 raster is 88,965,120 bytes. Six PRE/POST arms
for two pilots therefore require 2,135,162,880 bytes, about 1.99 GiB, before
MP4 files, terminal/shared latents, native-soft NPZ sidecars, logs, and
filesystem overhead. The fixed Wan snapshot objects total about 28.93 GB and
the selected framewise safetensors object is about 334.64 MB. Cache and output
space must cover those fixed bytes plus the unmeasured extras. CUDA is required
by the adopted config, but no GPU model, peak VRAM, or total wall time was
recorded in the historical evidence or measured in this delivery. The main
loader follows the successful sequence: prompt text encoder on GPU, release it,
then transformer; it does not claim a new model-offload path.

The historical environment recorded Torch 2.11.0+cu130, diffusers 0.39.0,
Transformers 4.57.6, and NumPy 2.1.3. Its `pip check` returned nonzero because
of retained Gradio/Hub and Jedi conflicts while the runner returned zero.
Accordingly the notebook saves `pip check` as a diagnostic rather than treating
any nonzero result as an automatic scientific or runtime failure.

## Files to return

Return the entire unique Drive directory. The minimum review set is:

- `handoff_summary.json`, `effective_config.json`, `execution_scope.json`,
  `portable_source_receipt.json`, `baseline_setup_receipts.json`, and
  `environment_setup_receipt.json`;
- `execution.log`, `stage_receipts.json`, and `pilot_phase_attempts.json`;
- `run_state/run_state.json`, `run_state/evaluation_report.json`, all five CSV
  tables, and the source-level comparison summary;
- complete `run_state/artifacts/pilot_01` and `pilot_02` trees, including every
  lossless native `.npz` sidecar.

This delivery was checked with standard-library notebook parsing, Python AST
compilation, embedded-file digests, no-`.git` extraction, and a CPU-only plan
expansion. It was not executed in Colab and did not load a model, weight, source
video, VAE, codec, GPU, or Drive output.
