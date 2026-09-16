# SC-SSTW

`main` is the sole authoritative implementation of the Wan terminal
projection/state-clock method. Historical development branches remain only to
reproduce their own records; they are not parallel authoritative
implementations and are not imported by this tree.

## Status

Implementation status is `formal_main_published`. Evidence status is separate:
recorded real Wan runs remain fixed-condition method diagnostics; publishing
this source does not establish
FPR, generalization, paper readiness, or Flow-time writing.

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

## Formal experiment and notebook

The formal entrypoint is `experiments.wan_state_clock.run`. The release notebook
is bound in a separate follow-up commit to this source commit, so its immutable
source SHA is visible and auditable. Both notebook and release checks use the
explicit `notebook_binding_kind` policy selection.
