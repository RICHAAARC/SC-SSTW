# HISTORICAL_CONDITIONAL_JOINT_SEEN_SOURCE_IMPORT

Report status: `COMPLETE_WITH_RETAINED_ISSUES`

Method: **SC-SSTW current-main conditional joint V1 historical import**

Finite public offset/phase and single-deletion receiver over one already-seen development source. Generation uses full-spatial-FFT repeated 32-bit payload control at steps 25..49 with pilot_gradient=0; terminal framewise M05 is a separate later write.

Evidence ceiling: Historical same-source development evidence only. This import does not establish independent-source generalization, FPR, synchronization necessity, payload-off quality, or scientific PASS.

## Comparability description

These fields describe the planned comparison. They do not adopt a matching rule.

| Field | Value |
| --- | --- |
| message_length_bits | `32` |
| redundancy | `Same 32 payload bits are repeated over the full 40x64 latent FFT carrier and 46 latent times; this is descriptive, not a cross-method matched rate.` |
| codec | `Existing saved pre/post media path from the frozen conditional-joint run; no new codec execution or cross-method codec matching.` |
| auxiliary_inputs | `Public key and protocol plus terminal framewise M05 for HYBRID. P0 already contains generation payload and is not OFF.` |
| matching_rule_status | `DESCRIPTIVE_ONLY_NOT_ADOPTED` |

## Fixed denominator

| Item | Count |
| --- | ---: |
| slots | 44 |
| measurements | 16 |
| pairs | 9 |
| independent_source_labels | 1 |
| observed_unique_physical_key_read_identities | 32 |
| slots `OBSERVED` | 44 |
| slots `FAILED` | 0 |
| slots `MISSING` | 0 |
| slots `EXCLUDED` | 0 |
| slots `UNSUPPORTED` | 0 |
| slots `CONFLICT` | 0 |

## Result bindings

| Result | Load | Saved status | Source SHA | Content identity | Config identity | Path |
| --- | --- | --- | --- | --- | --- | --- |
| historical_joint | LOADED | COMPLETE | `ac111d0fed253767651929d115c343fe1636c525` | `01c720a60e572454f46f01938b1311e5e91e9e67071ab9fbf4bcc0b451524376` | `c7661974d625b115c4b814e04ffd064ebc2a6f38cc3a97f934de2d08ecde6ba6` | `/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/result.json` |

Logical slots, unique physical key-read identities, input media, actual calls, and independent source labels are separate counts.

## Conditional 32-bit recovery

| Source | Arm | Receiver | Key label | Key role | Physical key-read IDs | Exact / full fixed | Exact / eligible | Exact / evaluable | Bit errors / evaluable bits | States |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| historical_seen_dev_source_ac111d0 | HYBRID | GLOBAL | K0 | CORRECT_KEY | 8 | 9 / 9 | 9 / 9 | 9 / 9 | 0 / 288 | `{"OBSERVED":9}` |
| historical_seen_dev_source_ac111d0 | HYBRID | GLOBAL | K1 | WRONG_KEY | 8 | 0 / 9 | 0 / 9 | 0 / 9 | 101 / 288 | `{"OBSERVED":9}` |
| historical_seen_dev_source_ac111d0 | HYBRID | ORACLE | K0 | CORRECT_KEY | 2 | 2 / 2 | 2 / 2 | 2 / 2 | 0 / 64 | `{"OBSERVED":2}` |
| historical_seen_dev_source_ac111d0 | HYBRID | ORACLE | K1 | WRONG_KEY | 2 | 0 / 2 | 0 / 2 | 0 / 2 | 22 / 64 | `{"OBSERVED":2}` |
| historical_seen_dev_source_ac111d0 | HYBRID | PATH | K0 | CORRECT_KEY | 2 | 2 / 2 | 2 / 2 | 2 / 2 | 0 / 64 | `{"OBSERVED":2}` |
| historical_seen_dev_source_ac111d0 | HYBRID | PATH | K1 | WRONG_KEY | 2 | 0 / 2 | 0 / 2 | 0 / 2 | 22 / 64 | `{"OBSERVED":2}` |
| historical_seen_dev_source_ac111d0 | HYBRID | RAW | K0 | CORRECT_KEY | 8 | 9 / 9 | 9 / 9 | 9 / 9 | 0 / 288 | `{"OBSERVED":9}` |
| historical_seen_dev_source_ac111d0 | HYBRID | RAW | K1 | WRONG_KEY | 8 | 0 / 9 | 0 / 9 | 0 / 9 | 101 / 288 | `{"OBSERVED":9}` |

## Planned RAW/SYNC pairs

| Pair | State | RAW errors | SYNC errors | SYNC - RAW | Interpretation / reason |
| --- | --- | ---: | ---: | ---: | --- |
| historical/global_00/K0/raw-v-global-geometry-only | EXCLUDED |  |  |  | FULL181 has one public offset and is a geometry/payload control, not a synchronization-success denominator. |
| historical/global_01/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/global_02/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/global_03/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/global_04/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/global_05/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/global_06/K0/raw-v-global | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/path_00/K0/raw-v-path | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |
| historical/path_01/K0/raw-v-path | EVALUABLE | 0 | 0 | 0 | NO_BER_GAIN_OBSERVED |

A RAW=0 and SYNC=0 pair is reported as `NO_BER_GAIN_OBSERVED`. It is not counted as a synchronization benefit.

## Quality and cost measurements

| Measurement | Source | Arm | Kind | State | Value | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| historical/PRE/P1_P0/rgb_psnr_db | historical_seen_dev_source_ac111d0 | PAYLOAD_ONLY_VS_NATIVE_PAYLOAD_SOURCE | QUALITY | OBSERVED | `32.67273254463354` |  |
| historical/PRE/M05_P1/rgb_psnr_db | historical_seen_dev_source_ac111d0 | M05_INCREMENT_OVER_PAYLOAD | QUALITY | OBSERVED | `38.351690984813004` |  |
| historical/PRE/M05_P0/rgb_psnr_db | historical_seen_dev_source_ac111d0 | HYBRID_VS_NATIVE_PAYLOAD_SOURCE | QUALITY | OBSERVED | `31.96813159452364` |  |
| historical/POST/P1_P0/rgb_psnr_db | historical_seen_dev_source_ac111d0 | PAYLOAD_ONLY_VS_NATIVE_PAYLOAD_SOURCE | QUALITY | OBSERVED | `32.209429017008695` |  |
| historical/POST/M05_P1/rgb_psnr_db | historical_seen_dev_source_ac111d0 | M05_INCREMENT_OVER_PAYLOAD | QUALITY | OBSERVED | `37.26074024298309` |  |
| historical/POST/M05_P0/rgb_psnr_db | historical_seen_dev_source_ac111d0 | HYBRID_VS_NATIVE_PAYLOAD_SOURCE | QUALITY | OBSERVED | `31.623100525939805` |  |
| historical/PRE/M05_P1/residual_roughness | historical_seen_dev_source_ac111d0 | M05_INCREMENT_OVER_PAYLOAD | QUALITY | OBSERVED | `1.8453359780495995` |  |
| historical/POST/M05_P1/residual_roughness | historical_seen_dev_source_ac111d0 | M05_INCREMENT_OVER_PAYLOAD | QUALITY | OBSERVED | `1.5124304928273704` |  |
| historical/cost/generation | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `true` |  |
| historical/cost/framewise_receiver_encode | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":8,"completed":8}` |  |
| historical/cost/sync_score | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":18,"completed":18}` |  |
| historical/cost/blind_wan_receiver_encode | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":21,"completed":21}` |  |
| historical/cost/blind_payload_read | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":30,"completed":30}` |  |
| historical/cost/oracle_payload_read | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":2,"completed":2}` |  |
| historical/cost/quality_compute | historical_seen_dev_source_ac111d0 | HYBRID | COST | OBSERVED | `{"attempted":6,"completed":6}` |  |
| historical/cost/wall_seconds | historical_seen_dev_source_ac111d0 | HYBRID | COST | MISSING | `` | planned measurement locator absent from result |

## Retained unplanned observations

None.

Descriptive fixed-denominator report only. Correct-key, wrong-key, and oracle rows remain separate. Missing or failed slots are not successes. A zero-error RAW/SYNC pair is NO_BER_GAIN_OBSERVED, not evidence of synchronization benefit.
