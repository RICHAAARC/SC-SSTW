# B1 DCT2 implementation handoff

2026-10-24, B-THREE-IMPLEMENT-01. Implemented in `dev/b-local-identity-v1`,
starting from the common `ac8929a` base. Current authorization adopts COMMON r1,
INTEGRATED_PROPOSAL c0 and B1-LOCAL-DCT2-r0. The older design's
PROPOSED_NOT_ADOPTED/readonly statements are historical; this component follows
the adopted c0, without implementing or running the older five-arm proposal.

## Files and ownership

- `main/tube_state/b_local_identity_dct_v1.py`: fixed DCT basis/layout, writer
  target/one local backward, shared projection, separate regional bit caches.
- `runtime/wan/b_local_identity_dct_v1.py`: original native50/CFG5 seam with
  DCT2 identity at k25..49. Does not load a model or call the state writer/media.
- `tests/test_b_local_identity_dct_v1.py`: small CPU tensors and fake trajectory
  dispatch. No old observations, real model, VAE, codec or GPU work.

No shared `__init__`, old source, common interface, runner, notebook, builder,
other group's file or historical untracked material is modified. No common
source change is needed from B1: the integrator can import these new modules.

## Writer APIs

```python
from main.tube_state import b_local_identity_dct_v1 as dct
target = dct.build_target(normalized_fp32_terminal_shape_tensor, key, bits32)
delta, receipt = dct.local_delta(conditional_clean_fp32, target)
```

The writer tensors have shape `[1,16,46,40,64]`; target is
`[2,4,46,16]` in canonical mode order. Both guards are written. There are 5888
targets, beta=1/sqrt(2), eta=368 and one loss/backward, so a unit DCT direction
receives `-(q-T)/8`. Delta is exactly zero outside the two direct support
regions/ch0..3. Subsequent generation/decoding need not preserve that support.
Each receipt separates regional/guard loss, actual delta scale, one backward,
two region projection batches and 5888 coefficients. No FFT/IFFT is called.

```python
from runtime.wan.b_local_identity_dct_v1 import run_trajectory
terminal_cpu_fp32, receipt = run_trajectory(
    pipe, initial, scheduler, prompt, negative, dtype, key, bits32, count, record)
```

This matches the original segmented positional signature. It reuses only that
module's conditional/unconditional forward helper, plus the original native
scheduler helper. It does not call its FFT `local_delta` or `guided_velocity`.
The fresh native history is run for all50 steps, with25 DCT2 gradients. An
entered failed gradient has its begin counter and propagates the actual error;
there is no retry. The common runner must retain its failed/not-entered slots.
Terminal176-state construction and all media work remain the total integrator's
responsibility; this runtime returns the actual P0 terminal without rewriting it.

## Shared projection and thin reader APIs

```python
# One actual E41 projection, shared by both keys and both Rc templates.
p = dct.project(z_e41_fp32, ('R0', 'R1', 'Rc'), range(1, 11))
q0 = p['coefficients']['R0']             # CPU tensor [4,10,16]
r0 = dct.read_coefficients(q0, key, 'R0', range(1, 11), key_id='key00')
rc0 = dct.read_coefficients(
    p['coefficients']['Rc'], key, 'R0', range(1, 11),
    observed_region_id='Rc', key_id='key00')
# For eval native P0/PM only: guards excluded, no Rc/native FFT branch.
pn = dct.project(z_native_fp32, ('R0', 'R1'), range(1, 45))
```

`project` also accepts the equivalent keyword form
`project(z, native=False, include_control=True)`. It validates actual FP32,
shape and finite inputs, then projects both regions once, optionally Rc. An
error is raised to the physical-call owner; no zero observation is substituted.
Coefficient tensors remain unkeyed in canonical `(u,v)` order. Keyed permutation
only takes place in `read_coefficients`, so no second physical projection is
needed. The basis has one cached CPU construction; actual device/dtype copies
are ordinary computation, not identity checks.

The thin reader returns a region record with `local_cells`, canonical
`coefficients`, `signed_votes`, `zero_coefficient_mask`, `planned_bits`,
`returned_bits` and status/cause information. Each cell has8 `bits`, each with
`local_bit`, `value`, `ones`, `count=8`, `returned_count`, `tie`,
`zero_coefficients`, `coefficient_refs`, `status` and `cause_set`.
Coefficients are ordered channel0..3 then keyed j=b,b+8 for every bit.
Counter first-encounter ties and strict>0/zero-to0 are retained. There is no
cross-region pooling, bit-based reference choice, source group or source clock.

The thin reader intentionally leaves `evidence_id=None`: total runtime owns
physical view/row/region IDs. It can attach those IDs and place the records at
the shared address `cache[evidence_id][key_id][str(local_bit)]`.
Every `coefficient_refs` entry gives channel, canonical mode index, (u,v) and
keyed index. Region records explicitly distinguish observed and carrier region.
Rc requires `observed_region_id='Rc'`, is `diagnostic_only`, and cannot be
relabeled as a third production replica. Do not place Rc in production cache.

An optional complete wrapper `decode(p, key, view_id=..., key_id=...)` returns
separate `regions['R0'/'R1']` and `control_diagnostics['R0'/'R1']` dictionaries.
Its physical ID spelling is `view_id/row{r}/{actual_region}`; if used, the common
planner must use the same IDs. IDs never contain key/candidate/bit; Rc templates
share one Rc ID. Using the thin reader instead lets the common adapter keep its
own matching ID format without changing B1 mathematics.

## Failure and scope contract

`unavailable_projection(native=..., include_control=..., status=..., cause=...)`
constructs the complete planned projection with missing coefficients. Supported
unavailable statuses are FAILED, NOT_ENTERED, UNSUPPORTED, INTERRUPTED,
UNRESOLVED. `decode` keeps those statuses on all corresponding local bit slots.
For a directly missing regional cache, all that region's slots remain null;
for a nonfinite coefficient row, all8 bits of that row remain null. Other
regions/rows retain their returned values, without changing required replicas.

Unavailable bit `count=8` is the fixed planned denominator, not evidence that
eight votes returned: `returned_count=0`, value/ones/tie/zero count are null.
Bad raw rows serialize as null, never NaN or fabricated zero. Available bits
have status AVAILABLE and returned_count8. Region/top status INCOMPLETE is a
summary; per-bit status/cause_set retains the actual failed/not-entered reason.
Rc failure is recorded in its diagnostic record, not silently substituted into
or counted as loss of a production region.

Physical RGB input/context and nominal/source support mapping stay with the
common view/B2/B3 adapters. This module extracts local latent regions after
full E41 encoding; it does not claim autonomous RGB patch decoding. State C,
reference freeze, CDC all-replica reads, calibration, path enumeration, source
relations, source/arm/private truth, post analysis and one-shot attempt ownership
are deliberately absent from its receiver signatures.

## Engineering verification

Using the existing CPU interpreter at
`diagnostics/trajectory-main-integration-20261007/validation-venv/bin/python`
under the project root, the15 component tests pass. They check double-precision
basis orthogonality, FP32 projection/target round trip, guard exclusion, the
actual autodiff1/8 scale and zero outside support, fresh-leaf isolation,
regional independence, zero/ties, key/template projection reuse, failure slots,
JSON null serialization, CFG arithmetic and fake50-step dispatch/failure.

The initial run had13 passes and two fixture issues: an overly tight FP32
round-trip tolerance and a missing `predict_x0` property on the fake scheduler.
The tolerance is now8 FP32 eps for synthesis plus projection; no method value,
reader threshold or carrier constant changed. The fake scheduler now has the
actual interface property. The full15-test rerun passed; a later region-iterator
handling correction is checked with its affected projection-reuse test.

This is CPU/unit and fake dispatch evidence only. No actual Wan/DiT, VAE, MP4,
GPU, Colab/Drive or old scientific replay was run. Real media survival, temporal
state interaction, quality, independent source attribution and scientific
success remain untested. Existing B1 design and its separate five-arm diagnostic
are not executed or silently folded into the c0 budget.
