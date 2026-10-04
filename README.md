# SC-SSTW

Research on video watermarks with local spatiotemporal carriers, generation-time
control and blind temporal synchronization.

## Current evidence

The independent **GROW video reference** demonstrates real native Wan MULTI
control and blind 32-bit repeated-payload recovery from a saved MP4 on one fixed
source. It is a runnable reference baseline, not a completed video time-sync
method. See [construction and verified run](docs/grow_video_reference_v1.md).

The remaining method chain is local time-state survival through VAE/media,
blind path recovery, time-dependent segment payload, segment/sequence
aggregation, and appropriate rejection. OLD8 and newer local-state candidates
remain development work; they are not the verified default in main.
See [current method status](docs/current_method.md) and the bounded
[development-route index](docs/development_evidence_index.md).

## Reference entrypoint

```bash
python -m experiments.wan_state_clock.grow_video_reference_run \
  --output /content/drive/MyDrive/Video-WM/GROW-Video-Reference-V1/my-fixed-run
```

Use a fresh output directory. The fixed OFF/MULTI/LAST run retains 3 MP4s,
24 blind readouts and 48 post-read evaluations. Its [Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/grow_video_reference_v1_colab.ipynb)
checks out an immutable source SHA. Historical notebook links remain unchanged.

## Retained early engineering entrypoints

The four-bit / 44-46 [integrated payload entrypoint](docs/integrated_payload_v1.md)
and legacy terminal state-clock `experiments.wan_state_clock.run` remain
available for reproduction. Their engineering integration and tests are not
evidence that the current method has completed real-video synchronization.
The original integrated notebook remains [SHA-pinned](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/7cfe2e73807486700b6d62728fdb6cfccddbaf11/notebooks/integrated_payload_v1_colab.ipynb).

## Local validation and optional runtime

Core checks use `requirements-dev.txt`. The GROW CPU fixture tests additionally
use Torch and the Diffusers/Wan APIs from
`experiments/wan_state_clock/requirements-grow-video-reference.txt`; Torch
should use a compatible existing installation. FFmpeg/ffprobe support fixture
media checks and real saved-media execution.

```bash
python -m pytest -q
python governance/tools/run_validation_profile.py release
```

These are engineering checks. They neither download model weights nor execute
a real generation experiment. Running the reference notebook loads the fixed
Wan model and writes a new result; no hardware model or exact Python-version
gate is imposed. Result completeness, payload recovery, quality and temporal
synchronization are separate conclusions.
