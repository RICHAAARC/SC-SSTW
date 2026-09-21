# SC-SSTW

This isolated candidate integrates fresh Wan generation, a four-bit payload,
the successful fixed-budget 44/46 trajectory controller, persisted attacks,
four-phase VAE receive, blind synchronization, independent OFF calibration,
existence rejection, and payload recovery. Historical development branches
remain reproducibility records and are not imported by this tree.

## Status

Implementation status is `integrated_candidate_cpu_validated_pending_gpu`.
The older two-message Uniform-Tanh run is positive development evidence for its
exact writer and receiver only. It does not validate this candidate's 16-code
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
legacy terminal state-clock entrypoint. The integrated Colab notebook is built
after the source commit and pins that immutable source SHA.
