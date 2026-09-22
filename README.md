# SC-SSTW

This development branch contains the [fixed-key Flow V1 candidate](docs/flow_fixed_key_v1.md), a non-attribution existence experiment awaiting user GPU execution. The four-bit implementation remains preserved on `main`; this branch does not merge or replace it.

The formal integrated core adopted for `main` combines fresh Wan generation,
a four-bit payload,
the successful fixed-budget 44/46 trajectory controller, persisted attacks,
four-phase VAE receive, blind synchronization, independent OFF calibration,
existence rejection, and payload recovery. Historical development branches
remain reproducibility records and are not imported by this tree.

## Status

At the original four-bit handoff, engineering integration was accepted for `main` and real execution was pending. The subsequent user run `integrated_payload_v1_20260921T163637378085Z` completed with partial recovery: 2/4 marked sources and 8/28 marked views. These four-bit results do not validate the new fixed-key candidate.
The older two-message Uniform-Tanh run is positive development evidence for its
exact writer and receiver only. It does not validate the integrated core's 16-code
payload, pilots, receiver, calibration, attacks, or new content roster.

## Dependencies and local checks

Core CPU tests require `numpy`; development checks additionally require
`pytest` and `nbformat`:

```bash
python -m pip install -r requirements-dev.txt
python governance/tools/run_validation_profile.py method
python governance/tools/run_validation_profile.py notebook
python governance/tools/run_validation_profile.py governance
python governance/tools/run_validation_profile.py release
```

`requirements-wan-runtime.txt` declares, but does not install or load during
these checks, the optional real Wan runtime packages used by the successful
replication workflow (including NumPy and its tokenizer/safety helpers). FFmpeg
and ffprobe are external runtime dependencies for persisted RGB24/H.264 media.

## Integrated experiment and notebook

The integrated CLI is:

```bash
python -m experiments.wan_state_clock.integrated_payload_run \
  --config experiments/wan_state_clock/configs/integrated_payload_v1.json \
  --output /content/drive/MyDrive/Video-WM/integrated_payload_v1_run
```

The original `experiments.wan_state_clock.run` remains available as the exact
legacy terminal state-clock entrypoint. The fixed Colab notebook pins source S4
`7ad4d9425bfb246b1cd5e4f95aab0c08de521df2` at notebook N5
`7cfe2e73807486700b6d62728fdb6cfccddbaf11`:

https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/7cfe2e73807486700b6d62728fdb6cfccddbaf11/notebooks/integrated_payload_v1_colab.ipynb

Its output directory is
`/content/drive/MyDrive/Video-WM/SC-SSTW-Core-Integration/integrated_payload_v1_<UTC timestamp>`.
The publisher separately records and verifies the final remote publication.
