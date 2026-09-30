# Video temporal state-path diagnostic V1

This mechanism/config is fixed before scoring the saved MULTI window values.
The existing writer, pilot, payload, generator and evidence remain unchanged.
The already observed run20260930T011044716840Z is CPU development evidence,
not independent confirmation. No model, GPU, VAE, media encoding or Colab runs.

A path has one received phase g and source watermark state s_tau with regular
tau in1..45. Initial b is0 for181 received frames and0..52 for129 frames;
g=(-b)%4 and tau_1=(b+g)//4+1. Use the old primary R45/full or R31/crop;
crop g0 j32 remains excluded. Every later edge permits delta tau in{0,1,2},
with at most one nonunit edge. This is a nominal regular-window model (about
four newly supplied RGB frames), not arbitrary single-RGB-frame edits, phase
changes or a VAE receptive-field model. The finite code never wraps.

The pilot emission is dot(X_i,keyed_PN)*temporal_sign(tau_i), normalized by
sqrt(64*R*sum(X squared)) using the same entire phase support. Summed emission
extends the previous global cosine. There is no edit penalty or advance prior.
No old0.5 or new empirical detection threshold applies. Zero energy is marked
explicitly with score0; nonfinite/missing phase data is incomplete. Absolute
1e-12 is only a numerical equality tolerance. All scores are
UNCALIBRATED_DIAGNOSTIC and accepted_payload=False.

DP states (observed i, source tau, nonunit-edge count) retain every numerical
optimal predecessor and reconstructible starting b. Full enumeration supplies
an independent path catalog check. Each full condition has89 catalog entries:
one zero-edit,44 repeats and44 skips; all44 skips exceed finite source support,
so45 entries are scorable. Each crop has53*(1+30*2)=3233 scorable entries.
Across6 full and24 crop conditions this is78126 catalog rows,264 structural
exclusions,77862 scorable and1278 zero-edit paths. Exclusions remain rows.

Exact (g, emitted sign sequence) groups are observational equivalence classes:
all members have the same score for any pilot observation. Persist class member
IDs/reconstructible parameters, b/event sets, per-observed-index feasible tau
sets and ranges, and zero-edit membership. Preserve top ties between distinct
classes too. Lexicographic (g, full tau sequence, b, event type, event index or0)
is a canonical representation only, never a normal-advance prior or evidence
of uniqueness. Structural ambiguity refuses a unique synchronization claim.

The pure path API receives only public key/protocol and received-length/phase
pilot arrays. It receives no payload, message, arm, filename, source start or
truth. The runner verifies102 saved artifacts' hashes and public schema, then
atomically persists30 blind condition catalogs/classes/DP graphs before any
60 message truth joins. All missing/failed slots remain; oracle cannot complete
a missing search. The previous global baseline remains a separate diagnostic.

Only posthoc reporting compares canonical fitted paths with the known no-edit
full/crop starts, distinguishes higher-score fits from equivalence ambiguity,
and reports payload statistics already saved. No observed anomaly selects or
deletes windows. Repeated payload cannot prove synchronization necessity.
Future real edited-MP4 and time-dependent-payload validation are interfaces
only; neither is implemented or claimed here.
