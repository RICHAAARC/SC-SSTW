# Fixed continuation: total update 4 to total update 7

Click **Run all**. The source is pinned below. Start from the audited POINT4 terminal of run 20261001T150038391655Z, with Delta=-4.3296350339845635e-7 and true rank2. Local POINT1 replays total-step4; POINT2, POINT3 and POINT4 are total-steps5,6,7. The fixed endpoint is local POINT4 / total-step7, even if an earlier point wins. Exactly three fresh-gradient updates; no best-point selection, early stopping, scan or automatic extension.

Unchanged: eta174, per-update shrink-only L2cap1, original1392 pilot coefficients, existing global-worst plus all-pair loss and public template margins, g0/R44, all174 valid paths and both keys. Each point uses native VAE decode, RGB8, materialized raw YUV444, reopened RGB24 and native VAE encode. Each update recomputes both VJPs. Identity STE is a direction estimate; actual channel readback determines scores. Paths88,90,83 remain ordinary members of the full candidate set; no candidate is removed and no initial position is imposed on the reader.

Four observations / three updates / eight path reads / eight payload reads / sixteen message comparisons /1392 valid costs. Existing negative controls remain historical references. Every point reports top catalog indices, Delta, rank, global target and shortfall, remaining active local hinges, payload and quality. Payload is a separate repeated-bit diagnostic; exact bits do not establish synchronized decoding.

Quality is measured against the original already-marked RGB, the saved total-step4 RGB, and the preceding new point. It is not quality relative to OFF. Per-update cap is1; this batch adds at most3 to the sum of step norms in ideal arithmetic. Prior sum0.10940240292144192 is carried forward. Actual displacement from the original terminal and from this batch start are recorded separately; no new cumulative clipping.

The completed preceding three-update run spent38.54minutes in gradient phases, with19.35GiB peak allocated CUDA memory and64.53GiB peak temporary disk. Expect roughly40minutes plus model loading, media and Drive persistence; no runtime guarantee or GPU-name gate. Current-Python setup and fresh child process follow the successful run. Gradient phases clean their temporary storage separately.

Local CPU/static checks do not execute pretrained weights. This notebook leaves real execution to the user. This remains a one-source terminal-development diagnostic: no Transformer replay, generation-time trajectory embedding, H264/MP4 robustness, unknown-phase synchronization, calibrated FPR or scientific PASS is claimed.


Source and immutable notebook are published separately. Original loop implementation and all pure method/runtime/reader files remain unchanged. Parent result SHA256: d5e4f5440255a561cfe6aca007d1de32ce89baff354262f93745d088f5cbe46c. Parent terminal SHA256: d1ac45610a2fc20eaf84044785fa7c102b363344557e4bce8bacf26cec490e86.
