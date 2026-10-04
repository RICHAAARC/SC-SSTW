# SC-SSTW

**main is the sole authoritative release branch.** It contains one complete,
independently copyable reference flow whose limited real mechanism has been
verified: native Wan generation-time MULTI writing → real VAE / saved MP4 →
independent blind 32-bit repeated-payload reading.

This is not completion of the full research proposal. Local time-state
synchronization, time-dependent segment payload, aggregation and calibrated
rejection remain development work outside main. See
[current scope](docs/current_method.md) and [fixed real evidence](docs/grow_video_reference_v1.md).

## Run from a clone or a ZIP

External model weights and standard Python/FFmpeg dependencies are not bundled.
Use a compatible Torch/Torchvision installation, then:

```bash
python -m pip install -r experiments/wan_state_clock/requirements-grow-video-reference.txt
python -m experiments.wan_state_clock.grow_video_reference_run --output /absolute/new-result-directory
```

Install FFmpeg and ffprobe through your normal system package manager if absent.
Run from the extracted project root; choose a fresh output directory.
The default fixed source produces OFF/MULTI/LAST, 3 MP4s, 24 blind readouts and
48 post-read evaluations. There is no runtime dependency on a parent workspace,
dev branch, archive or Git installation. CPU is supported but slow for real Wan.

The [Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/grow_video_reference_v1_colab.ipynb)
mounts Drive and checks out an immutable published source SHA. It writes a fresh
timestamped result and preserves failed/missing rows.

## Source identity outside Git

The committed release_manifest.json identifies the complete runtime source,
config and dependency declaration by SHA-256. GitHub's source ZIP includes it.
A no-Git result records its observed content identity and manifest match; its
source_sha is null. A real checkout additionally records its actual Git commit
and source status. Local modifications are reported rather than called an
official source match. A missing manifest remains explicitly unversioned.

The manifest is a reproducibility record, not a cryptographic signature or a
scientific acceptance threshold. Maintainers regenerate it with
`python scripts/build_release_manifest.py` after runtime changes.

## Validate the standalone release

```bash
python -m pip install -r requirements-dev.txt
python governance/tools/run_validation_profile.py release
```

In addition to installed runtime libraries, these checks use NumPy, pytest and
nbformat. The release profile copies this tree to a temporary directory without
.git and runs the current reference tests with isolated imports. The CPU fixture
covers native steps, local writing, actual FFmpeg save/read, fixture-VAE
interfaces, truth-free reception, evaluation and failure retention.
It does not download weights or count as another real Wan experiment.

## Historical reproduction

Unverified combinations and obsolete executables are absent from the current
release tree. Their immutable source and notebook links remain in the
[historical entry index](docs/historical_entries.md). Branches and Git history
are preserved. Successful results from different historical branches are not
combined into a completed synchronized watermark claim.
