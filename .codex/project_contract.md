# SC-SSTW public mechanism contract

Main is the public release branch. Retain the existing complete GROW chain,
adopted M05/global alignment/fixed single-deletion/conditional joint V1, and the
confirmed public multistep segmented payload. Keep historical and ongoing A/B
synchronization methods on their development branches; no full dev-tree merge.

Public segmented payload: native 50 steps, control indices25..49, effective
latent times1..44 in four11-time segments, eight bits/segment, net32. Physical
mask still includes guard times0/45 with first/last segment targets; guards are
not read votes. Preserve alpha.5, coordinates, local real/ortho FFT mean-MSE,
eta5520, conditional Flow update, FP32 CFG5 and native scheduler history.
Existing repeated-payload R44/R22 time-major bit::8 extraction stays unchanged,
as do existing saved-input181/177/89 geometries and the fixed conditional-joint
source roster.
Readout given a caller-supplied time map is conditional: no path estimator,
detection/authentication, missing-frame recovery or edited blind-sync claim.

Do not replace existing generation/VAE/GROW mathematics wholesale. Core has no
runtime/experiments/governance dependency; runtime has no outer-layer import.
Preserve original strict-zero and Counter encounter-tie rules. GROW's own all46
reader remains separate. Synthetic context frames cannot vote for missing cells.

Blind estimators use received input, key and public protocol only. Oracle maps
and true messages belong to conditional evaluation, after blind output saving
when both are compared. Existing blind payload seals precede oracle-map semantic
parsing, and true-message evaluation follows final oracle sealing. Preserve
fixed failures, aliases, missing support and uncertainty;
no retries filling old failed slots and no scientific PASS from engineering.

Current user AGENTS supersedes historical identity/binding contracts. Source,
config, environment and data identity are optional records, never admission
requirements. No B64/hash/manifest/Git-clean/fixed-version/source-binding gates
or strict/override modes. Keep real file/format/numeric and information-isolation
checks. Public CLIs work from ordinary source copies without .git. Notebooks
load available source; synchronize a changed notebook with its builder.
Run only checks covering the actual change. No model/media/GPU/Colab execution
or Git publication beyond current authorization. Old evidence retains its
original source, denominator and claim ceiling; integration is not a new run.
