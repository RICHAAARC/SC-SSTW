"""Post-numeric roles only. No path or payload decoding."""
from collections import Counter
from main.tube_state import b_legacy_migration_v1 as method


def evaluate(observations):
    rows=[];vectors=[]
    for role in method.plan():
        raw=observations[role["row"]]
        expected=method.TARGETS[role["target"]][1] if role["arm"]=="S_ID" and role["key"]=="key00" else None
        vectors.append(dict(**role,expected_identity=expected,members=raw.get("members",[]),
                            status=raw["status"],reused=role["layer"]=="mp4" and role["history"]=="local"))
        for item in raw["identities"]:
            positive=item["identity"]==expected
            member=item["status"]=="CODE_MEMBER"
            rows.append(dict(**role,expected_identity=expected,identity=item["identity"],
                geometric_role="POSITIVE" if positive else "NEGATIVE",status=item["status"],
                s=item.get("s"),rho=item.get("rho"),correct_identity_retained=positive and member,
                false_attribution=not positive and member,phase_certified=False))
    branches={}
    for layer in method.LAYERS:
        for history in method.HISTORIES:
            selected=[r for r in rows if r["layer"]==layer and r["history"]==history]
            branches[layer+"/"+history]=dict(planned_vectors=12,planned_identity_slots=552,
                planned_positive=2,planned_negative=550,
                retained_positive=sum(r["correct_identity_retained"] for r in selected),
                false_attributions=[r for r in selected if r["false_attribution"]],
                status_counts=dict(Counter(r["status"] for r in selected)),
                reused=layer=="mp4" and history=="local")
    comparisons={}
    # Matched identity retention only. This is not payload coverage/BER gain.
    for name,a,b in [("rgb8_history",("rgb8","local"),("rgb8","full")),
                     ("mp4_history",("mp4","local"),("mp4","full")),
                     ("full_codec",("rgb8","full"),("mp4","full")),
                     ("local_codec",("rgb8","local"),("mp4","local"))]:
        left=[r for r in rows if (r["layer"],r["history"])==a and r["geometric_role"]=="POSITIVE"]
        right=[r for r in rows if (r["layer"],r["history"])==b and r["geometric_role"]=="POSITIVE"]
        reused_side="before" if a==("mp4","local") else "after" if b==("mp4","local") else None
        comparisons[name]=dict(cross_run=reused_side is not None,reused_side=reused_side,denominator=2,retained=sum(x["correct_identity_retained"] and y["correct_identity_retained"] for x,y in zip(left,right)),
            lost=sum(x["correct_identity_retained"] and not y["correct_identity_retained"] for x,y in zip(left,right)),
            added=sum(not x["correct_identity_retained"] and y["correct_identity_retained"] for x,y in zip(left,right)),
            rows=[dict(target=x["target"],before=x,after=y) for x,y in zip(left,right)],
            payload_BER="UNMEASURED",scope="fixed clean identity response; not full-video coverage")
    return dict(scope="existing Wan carrier operator control, no equivalent legacy migration/receiver success claim",vectors=vectors,rows=rows,
        planned_vectors=48,planned_identity_slots=2208,planned_positive=8,planned_negative=2200,
        new_vectors=36,reused_vectors=12,status_counts=dict(Counter(r["status"] for r in rows)),
        branches=branches,comparisons=comparisons,path_error="NOT_ESTIMATED_NO_PATH",
        full_video_correct_coverage="UNMEASURED",false_payload_coverage="UNMEASURED_NO_PAYLOAD_OUTPUT",
        payload_BER="UNMEASURED_FFT0",cross_run_MP4_history_contrast=True,scientific_pass=None)
