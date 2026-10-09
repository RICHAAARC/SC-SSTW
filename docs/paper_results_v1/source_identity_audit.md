# Fixed-roster source identity audit

Status: **all ten adopted prompt/seed candidates had no literal match in the
searched local history, but unseen status remains unproven.** This audit fixes
the evidence label for the adopted roster; it does not replace a candidate or
turn absence from a bounded search into a novelty claim. Machine-readable
metadata is in
`experiments/paper_results_v1/source_identity_audit.json`.

The ten complete prompt/seed pairs are unique (10/10), and their seeds do not
overlap the two known development seeds `2026100701` and `2026092501`. Those
known seeds were checked against
`experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1_preparation.json`
and `experiments/wan_state_clock/configs/grow_video_reference_v1.json`.

## Recorded search

The plain-text pass used `rg -uu -n -o -i -U` with the ten seeds and distinctive
prompt phrases. It enumerated 9,971 candidate files across the following roots:

| Root | Enumerated files |
| --- | ---: |
| alive | 101 |
| archive | 1,392 |
| datasets | 4 |
| diagnostics | 3,293 |
| memory | 6 |
| other | 6 |
| release-candidates | 62 |
| session | 9 |
| worktrees | 5,097 |
| README | 1 |

The included extensions were JSON, JSONL, Markdown, text, YAML, TOML, Python,
notebook, CSV, TSV, and log files. Exit code 1 with empty stderr recorded zero
matches. The current Paper-Results-V1 worktree was excluded to avoid matching
the roster against itself. `.git`, framework and dependency environments,
cache directories, and generated dependency trees were also excluded.

A separate read-only pass scanned 10 zip files: 401 text entries and 74,259,656
decoded bytes. It recorded zero matches, zero errors, zero UTF replacement
characters, and no text entry skipped for exceeding 64 MiB. The archives were
the three SyncTube packages, four public-statistic preparation packages,
`diagnostics/main-portable-release-20261004/published-source.zip`, and the
latent-translation and local-temporal-warp source packages. Exact paths and
entry counts are preserved in the JSON metadata. Only the listed text formats
were decoded; media and weights were not extracted.

## Evidence limit

Each case is therefore labeled
`USER_ADOPTED_LIMITED_LOCAL_NO_MATCH_UNPROVEN`. The scan was neither global nor
an atomic snapshot: concurrent work could add text after enumeration. It did
not inspect Git history objects, nonlocal Drive/Colab data, external session
storage, images, videos, model weights, PDFs, other unlisted binary formats,
nested archives, or dependency contents. Literal matching can also miss a
semantic rewrite, a different seed, or a different description of the same
source. The valid conclusion is `NO_MATCH_IN_SEARCHED_LOCAL_HISTORY`, not
“confirmed unseen.”
