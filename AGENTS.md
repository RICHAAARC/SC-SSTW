# SC-SSTW main release contract

Read .codex/project_contract.md before changing main. main is the sole release
branch; unclosed candidate combinations stay outside its current executable tree.
Keep main/tube_state independent of runtime, experiments and governance.
runtime/wan must not import experiments. Blind reception must not receive
writer states or truth. Preserve the fixed GROW denominator and all failures.

The complete limited GROW chain must work from a standalone no-.git copy.
Record actual source content/manifest identity; never fabricate a Git SHA.
CPU/fixture checks verify release engineering, not full research completion.
