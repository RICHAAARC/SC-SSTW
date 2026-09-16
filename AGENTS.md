# SC-SSTW main contract

Read `.codex/project_contract.md` before changing main. Keep
`main/tube_state` independent of runtime, experiments, and governance.
`runtime/wan` is a shared adapter and must not import
`experiments.wan_state_clock`. Detection must not receive writer evidence or
truth labels; truth joins are reporting-only after search.

Preserve fixed denominators, missing observations, and failures in persisted
results. Record code/config/seed/call counts/output paths for a real run.
Local PASS is an engineering implementation check, not a scientific PASS.
