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

Result: `11 passed in 0.08s`.

The cases cover retained failed/missing/excluded/unsupported slots, complete
manifest denominators, unevaluable planned pairs, duplicate slot and input
conflicts, unplanned saved observations, conditional-joint schema adaptation,
blind/truth separation, the static 44-slot historical manifest, and CLI use
from a copied source directory with no `.git`. They also exercise the strict
integer-32 manifest and saved-result contract, explicit key labels, pair
identity mismatches, malformed result objects and rows, non-finite JSON input,
and report writing after those failures are retained.

## Same-version review repairs

Independent A2/A3 review requested a narrow engineering repair pass. The
reporter now binds V1 to integer 32 in the manifest and saved
`planned_final_bits`, requires manifest `key_label`, and prevents RAW/SYNC
comparisons across different result, view, input, or key identities. Malformed
top-level data, payload/posthoc containers, individual rows, and non-finite JSON
are retained as failed or conflicting evidence rather than aborting report
generation. Recovery summaries separately expose full fixed, eligible, and
evaluable bit denominators. Unplanned observations are the union of saved
payload and posthoc locators, with both sides' presence and status recorded.
Physical counts are labeled as unique physical key-read identities rather than
media counts or receiver call counts.

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
32 unique physical key-read identities. The whole run contains nine logical
views over eight physical input files; the eight primary synchronization cases
use seven of those inputs. The saved call receipts separately record 21 blind
Wan receiver encodes. All 15 available quality/call fields were imported;
the planned wall-clock cost remains `MISSING`, producing
`COMPLETE_WITH_RETAINED_ISSUES` without changing the 44-slot result denominator.
