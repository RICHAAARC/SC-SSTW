# Integrated Payload V1 delivery card

- Candidate branch: `dev/sc-sstw-core-integration`
- Canonical-protocol source S4: `7ad4d9425bfb246b1cd5e4f95aab0c08de521df2`
- Source-bound notebook N4: `909b7ae7a82e58d99704a50ba63c5ccc8ead6f81`
- Current review state: A2 and A3 passed D4 with no remaining blocker; the upper-level reviewer accepted D5 and authorized `main` publication
- Pre-repair R1--R3 source S3: `8da1687c05a86bb320bcee67bb084bcf020068a7`
- Previous N2 review: A2/A3/A4/A5 passed
  `5c28961a927d2479048eea9a659eda3d435a8d78`; that approval predates S3 and
  its A5 result does not approve the new Partial3 receiver or public API

## R1--R3 repair

R1 adds receiver protocol `SC-SSTW-Payload-RM13-Partial3-V2`. Complete
four-group emissions keep the original numerical path. Exactly-three-group
windows project only observed temporal components and clip once, without zero
fill, `4/3`, or energy rescaling; fewer groups are erased. Full blocks, partial
blocks, and observed components are reported separately. Scores, `q`, hard
evidence, aggregation weights, and calibration use the same component support.
Four completed phase encodes remain a separate formal view-eligibility rule.

R2 adds the reusable `runtime.wan.integrated_core` API and
`runtime.wan.integrated_cli`. A user can generate an MP4 from prompt, seed, any
payload `0..15`, key, and the standalone protocol, or receive an MP4 from only
its path, key, explicit protocol, and optional calibration. The fixed `[5,10]`
experiment imports the same public 44/46 writer and four-phase receiver. The
receiver rebuilds its codebook and has no writer-terminal, trajectory, truth,
or saved-codebook input. Missing or mismatched protocol/key calibration retains
ranking but returns `UNCALIBRATED` and no payload.
S4 closes the remaining R2 protocol-identity gap: public generation and receive
now require the entire canonical JSON, including every method, model,
generation, and media field. Keeping the same protocol ID while changing or
omitting a field is rejected before model loading. The fixed runner checks its
overlapping model, generation, payload, and control fields against that same
canonical document.

R3 stores hard-window evidence on every payload's best path. Crop aggregation
ranks a payload, combines only that payload's matched-path evidence across the
three received crop files, then runs RM decoding. View-global winners cannot
supply evidence for a different payload.

The earlier phase eligibility, retained generate/media exits and logs, live
stdout tee, predeclared failure slots, calibration-source decision shape, and
`UNCALIBRATED` evaluation fallback remain. The fixed denominator and call plan
remain 4 cases, 8 arms, 56 saved views, 224 receiver encodes, 448 transformer
forwards, 224 scheduler steps, and 6 each of zero shadows, unit probes, and CPU
clean-leaf backwards.

## Validation

- R1 structural regression enumerated all 1,428 `scale=[5,4]` paths for the
  145-frame SPEED5_4 geometry; every path has all eight data windows usable with
  three or four groups. The nominal path has group counts
  `[3,4,3,3,3,4,3,3]`. A controlled synthetic observation passed the complete
  blind search and recovered payload 13. A separate check proves the four-group
  projection remains exactly the original calculation.
- R2 CPU stubs called the real public orchestration without a count callback,
  passed payload 13 into the real control boundary, and exercised independent
  MP4 receive with no writer artifacts. A one-phase failure kept raw `SCORED`
  ranking but produced formal `INVALID` and no payload.
- R3's counterexample gives the three views different global payload winners
  `(1,2,3)` while payload 13 wins the aggregate; the decoder recovers 13 only
  from payload 13's per-view path evidence.
- The affected integrated file passed 19/19. The one authorized complete CPU
  suite passed 33/33. After pinning S3, notebook plus public-entry checks passed
  4/4. These are the D3 validation results and were not rerun for S4.
- S4's directed set passed 12/12: seven same-ID value mutations and one missing
  field were rejected before VAE loading; canonical public writer/receiver,
  omitted-count generation, and fixed-runner protocol checks remained valid.
  After binding S4, the notebook checks passed 2/2. These are
  CPU/synthetic/stub engineering checks only.
- A2 confirmed that `payload_codec` did not change from D3 to D4, so its R1/R3
  pass remains applicable. It also confirmed canonical equality and fixed-runner
  consistency without rerunning tests.
- A3 enumerated all 21 canonical leaf fields: every single-field value change
  and an extra field were rejected. Its same-ID wrong-CRF plus frozen-calibration
  counterexample reached zero VAE, read, and decide calls and could not return
  `DETECTED`; canonical prompt/seed input and the fixed runner remained valid.
  Ten directed pytest checks passed, and the notebook's first mount cell and S4
  source pin were correct.

## Pending evidence and scope

No GPU, Wan model, Colab, Drive, or remote run was executed. There is no real
attack-recovery, quality, FPR, or generalization result for S4. The historical
two-message 4/4 result applies only to its exact older method. The previous A5
approval likewise does not extend to the new Partial3 receiver or public API.

Upper-level review task `01a0c2b5-515d-7c23-b01f-a3221ed6de2a` accepted D5,
including the A2/A3 R1--R3 repairs, and explicitly authorized publication of the
complete integrated core to `main`. This publication commit contains the
accepted S4/N4 binding; final `main` and remote hashes and GitHub byte
verification belong to the publisher's release handback. At D5 no push or
authoritative `main` modification had occurred, so this card does not claim the
remote is already published.

Fixed Colab URL:
https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/integrated_payload_v1_colab.ipynb

Fixed output root:
`/content/drive/MyDrive/Video-WM/SC-SSTW-Core-Integration/integrated_payload_v1_<UTC timestamp>`.
