"""Card74 post-numeric geometry only. Never used by the observation functions."""
from collections import Counter


def evaluate(local):
    # This function is entered only after numeric_completion.json is durable.
    specifications = [("A", "clean", 61), ("B", "clean", 60), ("C", "clean", 62),
                      ("D", "clean", 109), ("E", "delete", 61), ("F", "delete", 64)]
    rows, roles = [], []
    for arm_index, arm in enumerate(("P0", "S_base", "S_ID")):
        for case_index, (case, view, a) in enumerate(specifications):
            n = arm_index*6+case_index
            source = list(range(181)) if view == "clean" else [i for i in range(181) if i != 63]
            target, context = source[a:a+4], source[a-5:a]
            complete = target == list(range(target[0], target[0]+4)) and (target[0]-1) % 4 == 0
            identity = (target[0]+3)//4 if complete else None
            continuous_context = context == list(range(context[0], context[0]+5))
            for key in ("key00", "key01"):
                raw = local[f"local/{n:02d}/{key}"]["vectors"][0]
                legal = identity if arm == "S_ID" and key == "key00" and complete else None
                role = dict(slot=f"local/{n:02d}/{key}", arm=arm, case=case, key_role="CORRECT" if key == "key00" else "WRONG",
                            source_target=target, source_context=context, target_complete=complete,
                            context_continuous=continuous_context, expected_identity=legal,
                            negative_control=(key == "key01" or arm != "S_ID"))
                roles.append(role)
                for item in raw["identities"]:
                    positive = item["identity"] == legal
                    member = item["status"] == "CODE_MEMBER"
                    rows.append(dict(**role, identity=item["identity"], geometric_role="POSITIVE" if positive else "NEGATIVE",
                                     status=item["status"], s=item["s"], rho=item["rho"], code_member=member,
                                     false_attribution=member and not positive, correct_identity_retained=member and positive,
                                     phase_certified=False))
    positive = [x for x in rows if x["geometric_role"] == "POSITIVE"]
    negative = [x for x in rows if x["geometric_role"] == "NEGATIVE"]
    return dict(scope="fixed local marker feasibility; not phase authentication/gap recovery/full receiver/FPR",
                physical_identity_slots=1656, positive_slots=3, negative_slots=1653,
                rows=rows, roles=roles, status_counts=dict(Counter(x["status"] for x in rows)),
                retained_positive=sum(x["code_member"] for x in positive),
                positive_not_retained=[x for x in positive if not x["code_member"]],
                false_attributions=[x for x in negative if x["code_member"]],
                negative_control_vectors=sum(x["negative_control"] for x in roles),
                context_discontinuity_does_not_remove_target=True,
                shifted_and_seam_can_have_identical_observation=True,
                logical_vectors=42, logical_identity_slots=1932, reused_identity_slots=276,
                logical_G_roles=[dict(arm=arm, key_role="CORRECT" if key=="key00" else "WRONG",
                    case="G", received_start=108, source_target=[109,110,111,112],
                    source_context=[104,105,106,107,108], target_complete=True, context_continuous=True,
                    reuse_of=f"local/{i*6+3:02d}/{key}", new_physical_observation=False)
                    for i,arm in enumerate(("P0","S_base","S_ID")) for key in ("key00","key01")],
                payload_fft_calls=0, full32_payload_effect="UNMEASURED", scientific_pass=None)
