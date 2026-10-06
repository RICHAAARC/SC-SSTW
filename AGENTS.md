# SC-SSTW main release contract

Read .codex/project_contract.md before changing main. Main is the sole release
branch. Keep the original GROW chain and the narrow M05 / global alignment /
fixed single-deletion whitelist; no full development-tree merge.
main/tube_state must not import runtime, experiments or governance. Runtime must
not import experiments/governance. Stable receiver extraction must preserve
vote ordering, strict zero and Counter ties; GROW keeps its own 46-time reader.
Blind selection uses received input/public protocol/key, never truth or writer
state. Retain all fixed failed/alias rows; semantic truth parsing follows seals.
Complete CLIs must work without notebook/scripts and from a no-.git source copy;
record real manifest content identity, never invent a Git SHA or inherit parent
Git. Model execution/Colab and Git publication require the current user scope.
CPU/fixture and release checks are engineering evidence, not scientific PASS.
