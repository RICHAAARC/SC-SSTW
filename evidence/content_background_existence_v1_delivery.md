# Content-Background-Existence-V1 delivery card

Status: source-reviewed engineering candidate, statically and CPU validated. The fixed new-source GPU experiment has not been run, and `main` was not changed.

## Method and implementation

- `main/tube_state/content_background_existence.py` is pure receiver code. Each view performs the correct-key ORIGINAL blind search once and the same complete 4,284-path search for each of 16 frozen raw SHA-256 wrong keys. It reports every raw wrong-key score, mean, population standard deviation (`ddof=0`), `Z`, and per-key attempted/scored path counts.
- `Z = (S_correct - mean16) / (population_std16 + 1e-6)`. Missing or malformed phases, non-finite values, any missing/invalid key search, or `std <= 1e-6` produce `INVALID`. The fixed key count, epsilon, and path family are not selectable.
- `experiments/wan_state_clock/content_background_existence_run.py` is the independent fixed Run-all runner. Four calibration OFF sources complete first and persist one threshold per receiver before the four evaluation sources begin. Evaluation uses OFF, legacy SINGLE46, and legacy MULTI44_46 with FULL, DELETE90, and SPEED5_4: 8 sources, 16 physical arms, 48 views, and 192 phase encodes.
- ORIGINAL reuses the candidate receiver's correct-key detection; it is not searched a second time. C1 and C2 remain separate frozen receivers. Paired evaluation OFF is reporting-only and never enters deployment calibration.
- The runner reuses the successful generation, VAE, media, offload, and subprocess paths. The only shared runtime change adds a scheduler-44 fingerprint and checks that each `generate_key_terminals` branch leaves the source scheduler snapshot unchanged. Writer controls, steps 44/46, `R*`, and call budgets are unchanged.
- `scripts/build_content_background_existence_notebook.py` produces the immutable-SHA notebook only after source commit `S` exists. It preserves the baseline unbounded `transformers` install and environment version receipt, exact torch 2.11.0 cu128 and diffusers 0.40.0 setup, two-line Drive mount cell, CUDA availability check without a GPU-model gate, retained nonzero-run results, and direct fixed Run all.

The frozen new-source roster is `diagnostics/parallel-method-development-20260923/receiver_new_source_roster.json`, SHA-256 `4347827e8f6bb5cd8cae76d86241249d159eef45408e890bced40e5aad2d2bf3`. The canonical config copies its eight case IDs, prompts, seeds, arm/view roster, and fixed denominators.

## CPU saved-tensor development result

The read-only replay used the prior 96 persisted phase tensors. Their receipt SHA-256 is `0e3a7b2e5dfbb64e598366992b422753bb27a94e6ec36886d7588e1507a5a8cf`; 96/96 file hashes matched.

- 24/24 views were `SCORED`; the correct-key ORIGINAL score/path reproduced 24/24 existing records.
- 408/408 fixed key searches attempted exactly 1,747,872 paths, with no retained failures.
- Four workers took 936.67 seconds wall time; summed view time was 3301.50 seconds, range 93.60–170.46 seconds.
- The two-source same-batch development threshold was `-0.8699531680108334`. This is not the formal four-source new-run threshold.

Evaluation results on the old development sources were:

| Receiver | OFF views detected | Marked views detected | OFF source-arms detected | Marked source-arms detected |
|---|---:|---:|---:|---:|
| CONTENT_BACKGROUND_Z | 3/6 | 12/12 | 1/2 | 4/4 |
| ORIGINAL | 5/6 | 12/12 | 2/2 | 4/4 |
| C1_MATCHED_CONFIRM | 0/6 | 4/12 | 0/2 | 2/4 |
| C2_STATE_CONFIRM | 0/6 | 4/12 | 0/2 | 2/4 |

All three p3 OFF views remain detected by the normalized candidate. Therefore the candidate reduces the same-batch ORIGINAL OFF count but does not separate p3 OFF; its 12/12 marked detections cannot be reported as method success. S and Z use different units, so their raw magnitudes are not compared. Same-source marked-minus-OFF rows and decision overlaps are descriptive only and do not define a threshold or causal background-removal claim.

The full local result remains at `evidence/content_background_existence_v1_development/result.json` and per-view files beneath that directory. The compact record is `evidence/content_background_existence_v1_development_summary.json`. The roughly 32.7 MB raw evidence directory should remain local unless the main audit explicitly selects it for publication.

## Version binding and validation

The CPU replay executed receiver SHA-256 `3d503b1411cfa79cc483ff7b6f76e47c02d2ed684a49d35a886b27a7fb1fbc14`, replay SHA-256 `8c4868716bbe125ecb34c9b58dff25f163d0122dc8a85545f8cd502f4e53ee63`, and spec SHA-256 `8755b1c4626206bafc7c804d359d7d3137da7f758533ab69dd55aa8b3d9eab6f`. The final receiver SHA-256 is `d743a82a60927d4d48aae6645aad6c39463b6a3d3fe80f264505cf28b67c4e68`; it only preserves the successful correct-key record and actual count fields when another fixed key makes the overall row INVALID. All executed rows were SCORED, and an independent 24-view recomputation matched mean, population std, and Z below `1e-12`, so the fixed replay was not repeated.

Validation completed before the final two A3 mechanical repairs: the full CPU/static suite passed 77 tests, the fixed-key runtime suite passed 17 tests, and the candidate plus paired regression set passed 18 tests. After the two repairs, the affected candidate tests passed 11/11; builder output parsed as Python through those tests, direct builder execution succeeded, and `git diff --check` passed. No GPU, model, new-source generation, VAE execution, or notebook experiment was run.

Review state: A2 and A3 returned PASS. A3's two mechanical findings were repaired: the extra `transformers` major-version gate was removed and `runtime/wan/payload_control.py` was added to the recorded source hashes. A4 synthesis accepted the engineering candidate and its development-only evidence boundary. A5 milestone audit returned PASS with no blocker.

## Remaining publication sequence

After final A3/A4/A5 and main-audit approval, create local source commit `S` from the precise approved source/evidence file set. Push only after the main audit authorizes it. Then run the builder with immutable `S`, statically verify the unique notebook, and create notebook-only commit `N`. The user performs the fixed GPU/Colab run.
