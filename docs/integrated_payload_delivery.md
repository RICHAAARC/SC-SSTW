# Integrated Payload V1 delivery card

- Candidate branch: `dev/sc-sstw-core-integration`
- Source S2: `b914c7387d22cb722f4f37e0d0f0592a0ef0ba08`
- Notebook N2: the source-binding commit containing this card; exact SHA is
  reported in the freeze handoff because a commit cannot embed its own SHA
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

The affected integrated test file passed 14/14 after the final progress/status
change. The complete project suite passed 28/28 during the repair cycle. The
fixed denominator remains 4 fresh cases, 8 arms, 56 saved views, and 224
receiver encodes. Final notebook pin/reproducibility checks run at N2.

## Pending external evidence

GPU, Wan model, Colab, Drive, remote publication, real attack recovery,
quality, generalization, and FPR evaluation were not executed. The notebook can
run only after its pinned source commit is published to the configured remote.
