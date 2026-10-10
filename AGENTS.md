# SC-SSTW main release contract

Read .codex/project_contract.md before changing main. Main is the sole release
branch. Keep the original GROW chain and the narrow M05 / global alignment /
fixed single-deletion / conditional joint V1 whitelist; no full development-tree merge.
main/tube_state must not import runtime, experiments or governance. Runtime must
not import experiments/governance. Stable receiver extraction must preserve
vote ordering, strict zero and Counter ties; GROW keeps its own 46-time reader.
Blind selection uses received input/public protocol/key, never truth or writer
state. Retain all fixed failed/alias rows; raw observations are saved before
semantic truth is read.
Complete CLIs must work without notebook/scripts and from an editable no-.git
source copy. A source URL or directory is enough; no source manifest is required.
Model execution/Colab and Git publication require the current user scope.
CPU/fixture and release checks are engineering evidence, not scientific PASS.

Project-wide rule: immutability and strong reproducibility are not project
requirements. Do not introduce B64 source packaging, expected package/source/
config hashes, source-SHA verification, dirty-Git checks, exact-version checks,
manifest requirements or identity-comparison workflows as execution admission.
Missing or changed identity/provenance must never block usable code. Use ordinary
source downloads and editable refs/directories; a brief URL/path note is enough.
Remove such governance instead of adding warning/strict/override modes. Preserve
safe extraction, actual import/load errors, input semantics, blind truth
isolation, fixed denominators and complete failure records.
