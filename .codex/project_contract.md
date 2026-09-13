# SC-SSTW Project Contract

## Current authorization

The project is at research_defined. The authorized implementation scope is the
migrated CPU-only synthetic feasibility bridge for affine-invariant public
capture, public-only calibration, and state-constrained synchronization.

## Claim ceiling

Current local Stage 1 authorization additionally permits isolated CPU saved-video
observation/public-candidate engineering under docs/stage1_observation.md. This
does not change method_readiness.yaml or establish any real-video scientific gate.

All migrated interfaces, tests, protocols and runner outputs are synthetic_only.
They do not establish performance on a VAE, DiT, Flow Matching, saved MP4,
attacks, fixed false-positive rate, or a paper claim.

## Boundaries

- main/ contains pure method code and may not import runtime/, experiments/,
  governance/, or paper_artifacts/.
- runtime/ is reserved for separately authorized real-model work.
- experiments/feasibility/ may only run CPU-only constructed-channel probes.
- paper_artifacts/ may only rebuild from future frozen records and manifests.
- outputs/, MP4 files and Drive packages are not committed.

## Validation

The default suite is lightweight and may run only CPU-only synthetic tests.
Real models, GPU work, external services and formal evidence workflows require
explicit current authorization.
