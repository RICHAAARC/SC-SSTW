# Paper Results V1 validation

Validation is limited to CPU/static reporting behavior. No model, GPU, Colab,
Drive, media encode/decode, generation, or new scientific experiment ran.

## Focused tests

Command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1.py
```

Result: `6 passed in 0.09s`.

The cases cover retained failed/missing/excluded/unsupported slots, complete
manifest denominators, unevaluable planned pairs, duplicate slot and input
conflicts, unplanned saved observations, conditional-joint schema adaptation,
blind/truth separation, the static 44-slot historical manifest, and CLI use
from a copied source directory with no `.git`.

## Historical saved-result import

Input:

`/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/result.json`

Command:

```bash
python3 -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/historical_conditional_joint.manifest.json \
  --result historical_joint=/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/result.json \
  --output-dir docs/paper_results_v1/historical_import_example
```

Observed import: 44/44 logical slots were present, all 22 correct-key K0 slots
had zero final bit errors for the fixed `OKOK` message, and eight predeclared
non-FULL RAW/SYNC pairs were evaluable. Every one had RAW=0 and SYNC=0, so each
is `NO_BER_GAIN_OBSERVED`. The FULL181 pair is retained as an excluded geometry
control. The report distinguishes one seen source label, 44 logical slots, and
32 unique physical key reads. The saved call receipts separately record 21
blind Wan receiver encodes. All 15 available quality/call fields were imported;
the planned wall-clock cost remains `MISSING`, producing
`COMPLETE_WITH_RETAINED_ISSUES` without changing the 44-slot result denominator.
