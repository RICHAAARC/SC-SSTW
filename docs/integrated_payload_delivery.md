# Integrated Payload V1 delivery card

- Candidate branch: `dev/sc-sstw-core-integration`
- Source S2: `b914c7387d22cb722f4f37e0d0f0592a0ef0ba08`
- Notebook N2: `5c28961a927d2479048eea9a659eda3d435a8d78`
- Review state: A2, A3, A4, and A5 passed N2 with no remaining blocker

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
receiver encodes. The source-pin notebook and affected integrated checks passed
16/16 after binding N2. This documentation-only final commit does not claim a
new run of the complete 28-test suite.

A2 closed four-phase eligibility, preservation of both stage exit codes, and
the calibration-source decision shape; its two directed checks passed. A3
confirmed that a real `g0` raw `SCORED` result becomes formal `INVALID` when the
other phases are absent, and checked successful, spawn-failure, and
missing-result fixed slots plus real subprocess stdout/stderr tee behavior;
seven directed checks passed. A4 found no unresolved synthesis disagreement.
A5 independently checked the fresh prefix-44 path, live histories at 44/46,
blind receiver signature and truth-only-after-decision reporting, four-phase
eligibility, no payload under `UNCALIBRATED`, the 4/8/56/224 denominator,
failure retention, and the source-bound Run-all notebook; all milestone checks
passed.

## Pending external evidence

GPU, Wan model, Colab, Drive, remote publication, real attack recovery,
quality, generalization, and FPR evaluation were not executed. The notebook can
run only after S2 is published to the configured remote. The A5 result supports
handoff to the upper-level final `main` audit only; it is not scientific
evidence. Nothing in this candidate was pushed, and authoritative `main` was
not modified.
