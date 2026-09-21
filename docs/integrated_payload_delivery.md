# Integrated Payload V1 delivery card

- Candidate branch: `dev/sc-sstw-core-integration`
- Source S2: pending freeze
- Notebook N2: pending source binding
- Review state: A2/A3 repair applied; fresh same-version review pending

## Delivered change

The candidate integrates fresh Wan generation, four-bit RM payload and explicit
pilots, fixed-budget SINGLE46/MULTI44_46 control, seven persisted views per arm,
four-phase blind receive, fixed-gain state search, independent OFF calibration,
per-view and crop/source decisions, and unknown rejection. Protocol eligibility
requires all four receiver phases even when a partial raw ranking exists.

Calibration freezes or records `UNCALIBRATED` before evaluation. A failed
calibration does not remove the fixed evaluation rows; it prevents every
existence decision from returning `DETECTED` or a payload. Calibration OFF rows
are labeled threshold-construction samples, not held-out FPR evidence.

The subprocess path preserves stage logs, generate/media exit codes, parent
launch failures, fixed slots, live tee output, and cumulative progress.

## Validation

Pending final S2/N2 test run. The fixed denominator remains 4 fresh cases,
8 arms, 56 saved views, and 224 receiver encodes.

## Pending external evidence

GPU, Wan model, Colab, Drive, remote publication, real attack recovery,
quality, generalization, and FPR evaluation were not executed. The notebook can
run only after its pinned source commit is published to the configured remote.
