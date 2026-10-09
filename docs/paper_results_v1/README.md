# Paper Results V1

This package turns an explicit predeclared slot manifest plus saved experiment
outputs into deterministic JSON, CSV, and Markdown reports. It is an outer
evaluation layer. It does not import or change generation, receiver, runtime,
or core mathematics.

## Run

From a repository root or a source copy with no `.git` directory:

```bash
python -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/manifest.template.json \
  --result RESULT_ID=/absolute/path/to/result.json \
  --output-dir /absolute/new/output/directory
```

The template intentionally has an empty source roster. Before a real run, the
user must adopt the evaluation sources, arm definitions, run budget, and any
decision thresholds. Each planned payload slot then names a `result_id` and an
exact saved-output locator. Saved observations fill those rows by left join;
they never define or enlarge the denominator.

The runnable example is explicitly synthetic:

```bash
python -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/synthetic_fixture.manifest.json \
  --result fixture=experiments/paper_results_v1/synthetic_fixture.result.json \
  --output-dir /tmp/paper-results-v1-synthetic
```

`historical_conditional_joint.manifest.json` is a static 44-slot import map for
the already-audited conditional-joint development result. It was declared from
the frozen protocol layout, not generated from observed successful rows. It
retains 40 blind and 4 oracle rows and separates correct-key, wrong-key, and
oracle summaries. The historical run has one seen development source, eight
logical primary synchronization cases, seven physical input files, 21 blind
receiver encodes, and 32 physical key reads. Those counts are different objects
and must not be pooled as independent sources.

The checked example outputs are under `historical_import_example/`. Its result
binding checks the original source SHA, content identity, and config identity;
the generated report also records the imported file path and byte SHA-256.

## Outputs and interpretation

- `report.json` contains the manifest denominator, blind rows, truth rows,
  conditional recovery summaries, predeclared pairs, measurements, input
  identities, and any unplanned saved observations.
- `blind_rows.csv` excludes posthoc truth and message-recovery fields.
- `truth_rows.csv` contains postseal recovery evaluation.
- `paired_sync_effect.csv` includes every predeclared pair, including incomplete
  ones. A RAW=0/SYNC=0 pair is `NO_BER_GAIN_OBSERVED`.
- `measurements.csv` retains missing quality and cost fields.
- `report.md` is a compact human-readable rendering of the same records.

The six slot states are `OBSERVED`, `FAILED`, `MISSING`, `EXCLUDED`,
`UNSUPPORTED`, and `CONFLICT`. The full fixed denominator is every manifest
slot, including excluded and unsupported rows. Conditional tables separately
show the full fixed, eligible, and evaluable denominators. Failed, missing, and
conflicting rows remain in the full and eligible denominators and cannot become
success; excluded and unsupported rows remain in the full denominator with
their reasons. A pair-level exclusion, such as FULL181, does not remove its
payload rows from the 44-slot denominator.

`global_00` / FULL181 has only one public offset and is a geometry/payload
control. The historical manifest retains its rows but excludes that comparison
from the eight-case synchronization pair denominator. Historical P0 already
contains generation payload, so P1/P0 and M05/P0 quality rows are not an OFF
watermark distortion measurement.

The historical 32-bit rows recover the fixed configured `OKOK` message. They
do not establish arbitrary 32-bit message capacity. A future message or
identity roster remains pending; this package does not randomize or replace the
payload when importing old evidence.

## Pending paired-control proposal

One narrow protocol is documented for user review, without adopting or running
it. For each future unseen source, keep the source/noise/seed paired and produce
`OFF_NATIVE` and `PAYLOAD_NATIVE` (P0). From P0, separately produce
`PAYLOAD_FRAMEWISE_RECON` (P1) and `PAYLOAD_FRAMEWISE_M05`. Every POST artifact
used in a comparison would traverse the same adopted codec, while PRE and POST
remain distinct. This would make full-hybrid versus OFF the total distortion,
P1 minus P0 the extra framewise-VAE component, and M05 minus P1 the M05
increment. The source count, roster, execution budget, codec, and numerical
criteria are all pending user adoption. Historical P0 cannot fill OFF, and its
POST references do not establish this future same-codec comparison.
Implementation stops at this pending boundary.
## External baseline inventory status

The local archive contains historical RivaGAN, VideoSeal, and framewise HiDDeN
adapters plus 7,000 score records per method. This first implementation does
not import those JSONL records into the main 32-bit recovery table. The raw
record decisions remain `pending_threshold_calibration`, while a separate
historical aggregation table contains thresholded summaries; neither artifact
has been rebound to the new protocol or re-audited here.

The inventory locations are:

- `SyncTube.zip::SyncTube/configs/baselines/external_{rivagan,videoseal}_source.json`
- `SyncTube.zip::SyncTube/experiments/baseline_comparison_gate/{rivagan,videoseal}_adapter.py`
- `SyncTube_results.zip::SyncTube_results/numeric_conclusion_data/source_evidence/baseline_comparison_gate/external_*/records/baseline_formal_score_records.jsonl`
- the sibling `tables/baseline_comparison_table.csv` files

RivaGAN uses the community `Peachypie98/RivaGAN` 32-bit checkpoint and averages
per-frame bit correctness rather than producing the same sequence-level 32-bit
recovery event; its source pin is
`efffa72a4ca46d4d5051f6970c96424c2cdab441`. VideoSeal, pinned at
`e00b98c7ca77eb1fb5b9b68260e7c6c8fc207a84`, expands its message to the
model-native length (commonly 256) while historical scoring compares an
expected prefix, so a 32-bit/native-capacity mapping requires a user decision.
The archived source adapter metadata says `formal`, while the score records say
`real_smoke_adapter`; exact same-version binding is therefore not established.
The HiDDeN record path has a known 30 predicted bits versus a recorded
`compared_length=32` inconsistency and stays a fallback inventory item. No
wrapper is copied and no historical TPR/FPR is promoted into this report. This
stage is an inventory only; an external-record adapter waits for the adopted
capacity, roster, codec, and score semantics.

The remaining user decisions for a new evaluation are the source roster and
size, VideoSeal capacity/redundancy semantics, the concrete OFF and
quality/resource comparison budget, and any additional attacks or numerical
thresholds. No publication notebook is generated by this package.
External baseline integration stops at this inventory boundary.
