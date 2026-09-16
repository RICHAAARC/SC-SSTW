# RPGov reference boundary

The compact profile, dependency-boundary, notebook, and release-check design
was informed by RPGov commit `59a92fe709a1bf82a4a3958fea5ef43bc67301f5`:
`governance/tools/run_validation_profile.py`,
`governance/policies/validation_profiles.yaml`, and the dependency, notebook,
and release audit modules under `governance/harness/audits/`.

This candidate retains only its four local checks plus the explicit future
published-notebook check. It does not copy RPGov's complete registry, field or
naming controls, stage machine, or release-package machinery.
