# Spatial zero mean candidate: local implementation

Candidate base7167d443038927ef4395b33f52cc9dc0a2512bbb, dedicated dev/video-trajectory-blind-validation-v1. All files are new; old production method/runtime/notebook and evidence stay byte-identical.

The sole encoding change is the adopted keyed DC-slot replacement. Writer and reader share the new versioned basis. State templates, finite-family mean SSE/ties, pilot active1392/eta174/cap1 and all C1 media/call denominators are unchanged. Full real generation is NOT_EXECUTED. Old all-cap logs are not inherited.

New writer sidecars retain75 full projections independently of blind reads; failures do not erase completed native trajectories. The terminal projection/fingerprint is separately guarded following main review. All model/VAE call counts stay inherited. The existing C1 engineering loop is versioned with explicit imports and no runtime global monkeypatch.

Validation: initial12 passed/1 failed32.91s; only failure was an exact-zero test for FP64 energy subtraction (1.11e-16). Test-only tolerance correction, targeted1 passed/12 deselected1.13s; all13 tests covered across those runs. No broad rerun, B grid, saved-run reranking or real model/media execution. Exact commands and boundary details are in local_validation.json. Callback on/off equality is specifically OVERLAP_MULTI; each synthetic run checks torch CPU RNG, not all possible RNG systems. An execution-neutral evidence_ceiling wording delta subsequently passed load_config static validation without a suite rerun.

Motivation only: prior terminal-formation snapshot c13f244bbc101ab7a2dd8e0d69eed580d27737c6a90da400c24d801c5403e139, tree4474280aae0ea4d8d04dff8ca431579039d82fc9b186823ef9fd61e791ad655e. The current candidate neither reconstructs nor reinterprets old observations under new U.

Notebook is an unpublished draft, SOURCE_SHA=None, independent output directory, no Colab link. No commit/push/merge/GPU/model/VAE/codec/C2 execution. Same-version numerical/runtime review and milestone review remain pending at freeze.
