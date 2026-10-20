"""Post-only clean directional contrasts and known-grid payload side effects."""
from main.tube_state import b_legacy_position_margin_v1 as method


def evaluate(sync,payload,truth):
    scores={};contrasts={};payload_rows={};pairs={}
    bits=truth.get("bits") if isinstance(truth,dict) else None
    for name,raw in sync.items():
        valid=raw["status"]=="SCORE_OBSERVED"
        rows=[]
        for i in range(44):
            q=raw["pooled_q"][i][i] if valid else None
            gap=raw["competitor_margins"][i][i] if valid else None
            rows.append(dict(clean_expected_position=i+1,q=q,competitor_margin=gap,
                exact_max_ties=raw["rows"][i].get("exact_max_ties"),
                clean_rank=None if q is None else 1+sum(v>q for v in raw["pooled_q"][i]),
                reliable_identity=None,status=raw["status"]))
        scores[name]=dict(rows=rows,unit_denominator=44,identity_status="UNCALIBRATED")
    for layer in ("native","mp4"):
        for key in method.KEY_IDS:
            p,s=[scores[f"{a}/{layer}/{key}"]["rows"] for a in method.ARMS]
            contrasts[f"{layer}/{key}"]=[dict(position=i+1,P0=p[i]["q"],PM=s[i]["q"],
                increment=None if p[i]["q"] is None or s[i]["q"] is None else s[i]["q"]-p[i]["q"],
                post_only=True) for i in range(44)]
    for name,raw in payload.items():
        decoded=raw["decoded_bits"]
        rows=[dict(bit=i,decoded=b,target=None if bits is None else bits[i],
                   correct=None if bits is None or b is None else b==bits[i]) for i,b in enumerate(decoded)]
        cells=[]
        for cell in raw["local_cells"]:
            for bit in cell["bits"]:
                b=bit["decoded"];i=bit["bit_index"]
                cells.append(dict(time=cell["observed_time"],bit=i,decoded=b,
                                  correct=None if bits is None or b is None else b==bits[i]))
        payload_rows[name]=dict(status=raw["status"],aggregate_denominator=32,rows=rows,
            local_bit_denominator=352,local_bits=cells,canonical_grouping=True,
            local_structural_missing=0,local_unread_bits=sum(x["decoded"] is None for x in cells),
            local_correct=None if bits is None else sum(x["correct"] is True for x in cells),
            local_errors=None if bits is None else sum(x["correct"] is False for x in cells),
            false_coverage="UNMEASURED; no reliable synchronization placement",
            sync_benefit=False,structural_missing=0,unread_bits=sum(x["decoded"] is None for x in rows),
            structure_rule="fixed complete clean181; decoding failure is unread, not deletion",
            attribution_error="UNMEASURED; no reliable identity/path decision")
    for key in method.KEY_IDS:
        p,s=[payload_rows[f"{a}/{key}"]["rows"] for a in method.ARMS]
        common=[i for i in range(32) if p[i]["decoded"] is not None and s[i]["decoded"] is not None]
        localp,locals_=[payload_rows[f"{a}/{key}"]["local_bits"] for a in method.ARMS]
        local_common=[i for i in range(352) if localp[i]["decoded"] is not None and locals_[i]["decoded"] is not None]
        pairs[key]=dict(fixed_denominator=32,common_bit_indices=common,
            local_fixed_denominator=352,local_common_indices=local_common,
            P0_local_common_BER=None if bits is None or not local_common else sum(localp[i]["correct"] is False for i in local_common)/len(local_common),
            PM_local_common_BER=None if bits is None or not local_common else sum(locals_[i]["correct"] is False for i in local_common)/len(local_common),
            local_added_coverage=[i for i in range(352) if localp[i]["decoded"] is None and locals_[i]["decoded"] is not None],
            local_lost_correct_bits=[i for i in range(352) if localp[i]["correct"] is True and locals_[i]["correct"] is False],
            local_lost_readability=[i for i in range(352) if localp[i]["decoded"] is not None and locals_[i]["decoded"] is None],
            P0_common_BER=None if bits is None or not common else sum(not p[i]["correct"] for i in common)/len(common),
            PM_common_BER=None if bits is None or not common else sum(not s[i]["correct"] for i in common)/len(common),
            added_coverage=[i for i in range(32) if p[i]["decoded"] is None and s[i]["decoded"] is not None],
            lost_correct_bits=[i for i in range(32) if p[i]["correct"] is True and s[i]["correct"] is False],
            lost_readability=[i for i in range(32) if p[i]["decoded"] is not None and s[i]["decoded"] is None])
    return dict(status="POST_ONLY",evidence_level="DIRECTION_EVIDENCE_ONLY",scores=scores,directional_contrasts=contrasts,payload=payload_rows,
        payload_pairs=pairs,truth_status=truth.get("status"),identity_status="UNCALIBRATED",
        phase_status="UNCALIBRATED",accepted_identity_count=None,false_attribution=None,
        path_error="UNMEASURED",scientific_pass=None,
        scope="single clean full-history directional transfer and canonical payload side effects; no edited/unknown-phase validation")
