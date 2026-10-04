# Historical entrypoints outside the current release

The current main tree contains only the verified limited GROW chain and its
required shared components. No branch was deleted and no Git history was
rewritten. The entries below remain exactly reproducible from immutable
source [3bee8f6d4cd55784f69c766cf9397774848bba73](https://github.com/RICHAAARC/SC-SSTW/tree/3bee8f6d4cd55784f69c766cf9397774848bba73).

## Removed current-tree files

- [docs/integrated_payload_delivery.md](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/docs/integrated_payload_delivery.md)
- [docs/integrated_payload_v1.md](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/docs/integrated_payload_v1.md)
- [experiments/wan_state_clock/configs/fixed_terminal_validation.json](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/experiments/wan_state_clock/configs/fixed_terminal_validation.json)
- [experiments/wan_state_clock/configs/generate_replication.json](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/experiments/wan_state_clock/configs/generate_replication.json)
- [experiments/wan_state_clock/configs/integrated_payload_v1.json](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/experiments/wan_state_clock/configs/integrated_payload_v1.json)
- [experiments/wan_state_clock/integrated_payload_run.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/experiments/wan_state_clock/integrated_payload_run.py)
- [experiments/wan_state_clock/run.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/experiments/wan_state_clock/run.py)
- [main/tube_state/payload_codec.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/main/tube_state/payload_codec.py)
- [main/tube_state/projection_margin.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/main/tube_state/projection_margin.py)
- [main/tube_state/state_clock.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/main/tube_state/state_clock.py)
- [notebooks/integrated_payload_v1_colab.ipynb](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/notebooks/integrated_payload_v1_colab.ipynb)
- [notebooks/wan_state_clock_colab.ipynb](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/notebooks/wan_state_clock_colab.ipynb)
- [runtime/wan/integrated_cli.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/runtime/wan/integrated_cli.py)
- [runtime/wan/integrated_core.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/runtime/wan/integrated_core.py)
- [runtime/wan/integrated_payload_protocol.json](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/runtime/wan/integrated_payload_protocol.json)
- [runtime/wan/payload_control.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/runtime/wan/payload_control.py)
- [runtime/wan/quality.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/runtime/wan/quality.py)
- [scripts/build_integrated_payload_notebook.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/scripts/build_integrated_payload_notebook.py)
- [tests/test_integrated_notebook.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/tests/test_integrated_notebook.py)
- [tests/test_integrated_payload.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/tests/test_integrated_payload.py)
- [tests/test_state_clock.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/tests/test_state_clock.py)
- [tests/test_wan_projection.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/tests/test_wan_projection.py)
- [tests/test_candidate_isolation.py](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/tests/test_candidate_isolation.py)
- [requirements-wan-runtime.txt](https://github.com/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/requirements-wan-runtime.txt)

## Why these are outside main

The four-bit RM/44–46 integrated combination has engineering tests, but its
complete real mechanism has not been established. Its entrypoints, protocol,
configs, notebook builder and dedicated tests are therefore outside the
current release. The terminal state-clock/projection modules have historical
carrier-specific evidence, but are neither required by the GROW chain nor a
validation of current generated-video temporal synchronization. They stay
available through the immutable source above rather than as another main API.

The shared generation module no longer exposes its old terminal-only wrapper.
The trajectory module retains only fingerprints, native_step and scheduler
validation; the 44/46 reference, prefix and speculative-copy helpers were
removed. The GROW method and media adapter semantics remain unchanged.

Historical notebooks: [four-bit integration](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/notebooks/integrated_payload_v1_colab.ipynb)
and [terminal state-clock](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/3bee8f6d4cd55784f69c766cf9397774848bba73/notebooks/wan_state_clock_colab.ipynb).
Their own immutable source bindings remain historical records, not current
release endorsements or full-proposal completion claims.
