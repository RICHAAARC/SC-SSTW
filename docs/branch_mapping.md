# Branch name mapping

- Current branch: `dev/public-statistic-control`.
- Historical local and remote compatibility tag: `dev/生成端公共关系载体/二维图像统计-持续生成约束`, retained at `891988bfeaedb82a622d9ae3aac7a3e1531d66d5`.
- Existing branch-based notebooks now fetch that exact pre-rename source commit and verify HEAD. This freezes the previous default source; it does not identify the source of earlier experiments.
- Existing SHA-pinned notebooks and experimental code are unchanged. Historical diagnostics remain unchanged. No model experiment was run for this maintenance.

- The old Chinese branch has been removed; the same-name tag preserves the original commit. Historical `git fetch origin NAME` and `git clone --branch NAME` were both verified from fresh temporary repositories against the remote tag. All development branches now use English names.
