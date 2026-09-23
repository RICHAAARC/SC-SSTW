# Content-Background-Existence-V1 frozen method spec

This candidate normalizes the correct-key ORIGINAL blind-search score against a fixed background of sixteen wrong-key scores. It is an existence receiver only. It receives the four persisted phase tensors and no writer evidence, truth label, paired OFF sample, or fixed-path oracle.

For every saved view, run the unmodified ORIGINAL search over all 4,284 `state_clock.clock_paths()` once with the configured correct key and once with each wrong key below. Every key uses the same complete path family and ranking rule. Let its finite winning score be `S_key`. The candidate statistic is

`Z = (S_correct - mean(S_wrong[0:16])) / (population_std(S_wrong[0:16]) + 1e-6)`.

The wrong keys are the raw 32-byte SHA-256 digests of UTF-8 `SC-SSTW-Content-Background-Existence-V1/wrong/NN`, where `NN` is exactly `00` through `15`. Their frozen hexadecimal encodings are:

| NN | SHA-256 digest hex |
|---|---|
| 00 | `9dca8e4c40959bbf126f524fcba265cc336b4303226e4f87cbd2dcffe2b375c5` |
| 01 | `8d0c3ccd4ccee1ce2e877d28dcad2f3fc67ee0105f800cfe8c0b31e4f984d6ec` |
| 02 | `daf439c90c825b7921273ed89af57ccb75915cde7433705a043f4cf7d58db16f` |
| 03 | `363db89fe336f8da9042f9bc866e1932d17e3e15a930f2b79ff5a7a11ec913d0` |
| 04 | `e56325da82f9b6a197647a90d211dfd4ef61031390e664c98fca4dd615207e2e` |
| 05 | `e4377efe95d3a36234e0f84e7cd8aaef4619171793f088658a361e7fc697324d` |
| 06 | `cef5abcdd4eb81fb123d96feb3e987750d2eb9c38b61b950da410bcf644410e1` |
| 07 | `5809292bb9e16fba722c2fa8045e65462379681fd7f22df021bdfc5da9d66986` |
| 08 | `25e6de36409a9c8681c670d30a6161d5ca81a2be9dbab09cc9bbe959e96da5a3` |
| 09 | `9f6b39425650a2fe5c934470c97d8849358e161abe0bd68459e91f7d7181402c` |
| 10 | `ec8fb5e649de70e7d5f75e179b1dfca7faec90f02769efa476ced7391595bc70` |
| 11 | `37006db392771e3b67f0a53259bafca760a46c8d868cab2a926a683bb9203a34` |
| 12 | `62c07d122be704ad72ec0129ed6ae13d0c97c70def0c05153210dd3929e1d6ba` |
| 13 | `81641c8441992c011292d5bc9e7bf0218aa01e04e3ef38cd3ba624d516fcab5f` |
| 14 | `65f6f780f2b7a286fe6b5f1a9cf0d63373152c9a991be4bba87f9431bf31b36f` |
| 15 | `a1fff9f728667bfa961469607248be87171acc76659cd854d9ae6338568bf427` |

A view is `INVALID` if the four exact phases are not all present, any phase tensor is malformed or non-finite, any of the seventeen searches is missing/invalid/non-finite, any search does not attempt exactly 4,284 paths, or the population standard deviation is at most `1e-6`. There is no key subset, retry-based substitution, changed key count, sample standard deviation, or changed epsilon.

The development replay denominator is fixed at 24 saved views, 96 phase tensors, 17 full searches per view, and 1,747,872 path evaluations. All rows and failures are retained. The correct-key ORIGINAL result is reused as the ORIGINAL report; it is not searched twice. C1 and C2 are reported separately with their frozen receiver semantics.

For the new-source run, each receiver independently freezes a threshold after all four calibration OFF sources: first take the maximum of FULL, DELETE90, and SPEED5_4 within each source, then take the maximum across the four sources and add `1e-6`. Missing or invalid calibration input yields `UNCALIBRATED`; evaluation is still attempted and retained without constructing a replacement threshold. Paired evaluation OFF is never a deployment input.

The fixed new-source denominator is 8 fresh sources, 16 physical source-arms, 48 saved views, and 192 phase encodes. Calibration is 4 OFF sources; evaluation is 4 sources with OFF, SINGLE46, and MULTI44_46, using FULL, DELETE90, and SPEED5_4. The writer remains the legacy 44/46 `R*` path. This is a same-batch development candidate and fixed user-run validation design, not an independent low-FPR or generalization claim.
