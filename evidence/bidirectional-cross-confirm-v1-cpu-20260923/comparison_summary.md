# Bidirectional-Cross-Confirm-V1 CPU comparison

The full CPU replay passed 24/24 views and exact A→B full-field equivalence in 24/24. The joint observation+matching-codebook partition-role swap passed one full 4,284-path check.

All counts below use the fixed old evaluation batch. OFF and marked results are shown together; calibration samples are excluded.

| Receiver | OFF views | marked views | OFF source-arms | marked source-arms |
|---|---:|---:|---:|---:|
| ORIGINAL | 5/6 | 12/12 | 2/2 | 4/4 |
| C1_MATCHED_CONFIRM | 0/6 | 4/12 | 0/2 | 2/4 |
| C2_STATE_CONFIRM | 0/6 | 4/12 | 0/2 | 2/4 |
| BIDIRECTIONAL_C2_MEAN | 0/6 | 6/12 | 0/2 | 2/4 |

The candidate adds two marked view detections, both p2 SPEED5_4. Its blind-minus-reference gaps are -0.019112 and -0.022730, versus -0.011674 and -0.018261 for single-direction C2. The gains therefore do not come from numerically narrowing the blind/reference gap; they come from the changed statistic and its own calibration threshold on this development batch.

For p3, single-direction C2 fixed references pass 0/6. The bidirectional fixed-reference mean passes 6/6, while bidirectional blind scores pass 0/6. This supports a remaining blind-selection loss under the new statistic on this reused batch; it does not establish independent generalization.

Evidence ceiling: old same-batch CPU development diagnosis only. No model, VAE, GPU, media, or new-source execution occurred; GPU resources remain unmeasured.
