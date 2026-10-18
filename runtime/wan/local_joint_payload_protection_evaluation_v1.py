"""Post-persistence semantic evaluation only; never used by blind selection."""
from collections import Counter

from main.tube_state import local_joint_payload_protection_v1 as m
from main.tube_state import local_payload_secded_39_32_v1 as ecc

AVAILABLE = {"READ", "UNRESOLVED", "CODE_CONSISTENT", "SINGLE_ERROR_RULE_APPLIED",
             "DETECTED_UNCORRECTABLE", "INVALID_SHORTENED_SYNDROME", "INVALID_CODEWORD"}


def bit_row(decoded,target,status,**indices):
    available=status in AVAILABLE
    return dict(**indices,decoded=decoded,target=target,
        status=("READ" if decoded is not None else "UNRESOLVED") if available else status,
        correct=(decoded == target) if available and decoded is not None else None)


def summarize(rows):
    correct=sum(r["correct"] is True for r in rows)
    errors=sum(r["correct"] is False for r in rows)
    erasures=sum(r["status"] == "UNRESOLVED" for r in rows)
    count=correct+errors
    return dict(denominator=len(rows),rows=rows,correct=correct,errors=errors,erasures=erasures,
        unavailable=len(rows)-count-erasures,read_count=count,coverage=count/len(rows) if rows else None,
        ber_on_read=None if not count else errors/count,fixed_error_fraction=errors/len(rows) if rows else None)


def compare(old,new):
    if len(old["rows"]) != len(new["rows"]):
        raise ValueError("comparison denominator mismatch")
    common=[]; changes=[]; added=[]; lost=[]; old_errors=new_errors=0
    for a,b in zip(old["rows"],new["rows"]):
        index={k:a[k] for k in ("position","bit","coordinate") if k in a}
        if a["target"] != b["target"]:
            raise ValueError("comparison truth mismatch")
        x,y=a["decoded"],b["decoded"]
        if a["correct"] is not None and b["correct"] is not None:
            common.append(index);old_errors+=not a["correct"];new_errors+=not b["correct"]
        if x is None and y is not None: added.append(index)
        if x is not None and y is None: lost.append(index)
        if x != y or a["status"] != b["status"]:
            event=("REPAIRED" if a["correct"] is False and b["correct"] is True else
                   "INTRODUCED_ERROR" if a["correct"] is True and b["correct"] is False else
                   "LOST_COVERAGE" if x is not None and y is None else
                   "ADDED_COVERAGE" if x is None and y is not None else "OTHER_CHANGE")
            changes.append(dict(**index,old=x,new=y,target=a["target"],old_status=a["status"],new_status=b["status"],event=event))
    return dict(denominator=old["denominator"],common_read_count=len(common),common_read_indices=common,
        old_common_ber=old_errors/len(common) if common else None,new_common_ber=new_errors/len(common) if common else None,
        added_coverage=added,lost_coverage=lost,repaired_errors=sum(r["event"] == "REPAIRED" for r in changes),
        introduced_errors=sum(r["event"] == "INTRODUCED_ERROR" for r in changes),
        lost_previously_correct=sum(r["old"] == r["target"] and r["new"] is None for r in changes),changes=changes)


def evaluate_local(raw,message,codeword,*,encoded_arm,key_id):
    states=[];symbols=[];data=[];parity=[];post=[];blocks=[];gaps=[];margins=[]
    scores=raw["raw_scores"].get("pooled_scores")
    for k in m.STATES:
        pos=raw["symbols"]["positions"][k]; decoded=raw["ecc_blocks"][k]
        states.append(bit_row(pos["inferred_state"],k,pos["status"],position=k))
        block_raw=[];block_post=[]
        for i,value in enumerate(pos["codeword_bits"]):
            row=bit_row(value,codeword[i],pos["status"],position=k,coordinate=i)
            symbols.append(row);block_raw.append(row)
        for b,i in enumerate(ecc.DATA_COORDINATES):
            data.append(bit_row(pos["data_bits"][b],message[b],pos["status"],position=k,bit=b,coordinate=i))
            row=bit_row(decoded["message"][b],message[b],decoded["status"],position=k,bit=b,coordinate=i)
            post.append(row);block_post.append(row)
        for j,i in enumerate(ecc.PARITY_COORDINATES):
            parity.append(bit_row(pos["parity_bits"][j],codeword[i],pos["status"],position=k,coordinate=i))
        returned=decoded.get("returned_codeword")
        raw_data=pos["data_bits"]
        blocks.append(dict(position=k,decoder_status=decoded["status"],inferred_state=pos["inferred_state"],
            raw_errors=sum(r["correct"] is False for r in block_raw),raw_erasures=sum(r["status"] == "UNRESOLVED" for r in block_raw),
            raw_unavailable=sum(r["status"] not in ("READ","UNRESOLVED") for r in block_raw),
            raw_error_coordinates=[r["coordinate"] for r in block_raw if r["correct"] is False],
            post_message_errors=sum(r["correct"] is False for r in block_post),
            post_message_correct=all(r["correct"] is True for r in block_post),
            rule_applied=decoded.get("rule_applied",False),flipped_coordinate=decoded.get("flipped_coordinate"),
            returned_wrong_codeword=returned is not None and returned != codeword,
            miscorrection=decoded.get("rule_applied",False) and returned is not None and returned != codeword,
            undetected_wrong_codeword=decoded["status"] == "CODE_CONSISTENT" and returned != codeword,
            newly_wrong_data_bits=[b for b in range(32) if raw_data[b] == message[b] and decoded["message"][b] is not None and decoded["message"][b] != message[b]],
            parity_only_fix_without_data_gain=(decoded.get("rule_applied",False) and decoded.get("flipped_coordinate") in ecc.PARITY_COORDINATES and raw_data == message and decoded["message"] == message)))
        # Truth-state geometry is reported separately, never fed to decoder.
        for j in m.STATES:
            if j != k:
                value=None if scores is None else scores[k][k]-scores[k][j]
                gaps.append(dict(position=k,competitor=j,gap=value,target_shortfall=None if value is None else m.GAMMA_S-value))
        for j in range(39):
            if scores is None:
                coordinate=target=margin=None
            elif j < 32:
                b=raw["raw_scores"]["public_permutations"][k][j]
                coordinate=ecc.DATA_COORDINATES[b]
                target=(2*message[b]-1)*raw["raw_scores"]["public_masks"][k][j]
                margin=target*scores[k][22+j]
            else:
                p=raw["raw_scores"]["public_parity_permutations"][k][j-32]
                coordinate=ecc.PARITY_COORDINATES[p]
                target=(2*codeword[coordinate]-1)*raw["raw_scores"]["public_parity_masks"][k][j-32]
                margin=target*scores[k][22+j]
            margins.append(dict(position=k,carrier_slot=j,coordinate=coordinate,margin=margin,
                                target_shortfall=None if margin is None else m.GAMMA_P-margin))
    result=dict(status=raw["status"],scope="coded transmission" if encoded_arm and key_id == "CORRECT" else "reference agreement control, not pooled transmitted BER",
        state=summarize(states),raw_symbols=summarize(symbols),raw_data=summarize(data),raw_parity=summarize(parity),ecc_data=summarize(post),
        raw_aggregate=summarize([bit_row(r["decoded"],message[b],r["status"],bit=b) for b,r in enumerate(raw["raw_aggregate"])]),
        ecc_aggregate=summarize([bit_row(r["decoded"],message[b],r["status"],bit=b) for b,r in enumerate(raw["ecc_aggregate"])]),
        blocks=blocks,decoder_status_counts=dict(Counter(b["decoder_status"] for b in blocks)),
        miscorrections=sum(b["miscorrection"] for b in blocks),undetected_wrong_codewords=sum(b["undetected_wrong_codeword"] for b in blocks),
        parity_only_fixes_without_data_gain=sum(b["parity_only_fix_without_data_gain"] for b in blocks),
        truth_geometry=dict(state_competitor_denominator=462,state_gaps=gaps,symbol_denominator=858,symbol_margins=margins,
            meaning="post-blind truth-state geometry; never substitutes for actual inferred-state bit rows"))
    result["raw_to_ecc_local"]=compare(result["raw_data"],result["ecc_data"])
    result["raw_to_ecc_aggregate"]=compare(result["raw_aggregate"],result["ecc_aggregate"])
    return result


def evaluate_all(observations,writer_encoding):
    message,codeword=writer_encoding["message"],writer_encoding["codeword"]
    result={}
    for slot,observed in observations.items():
        arm,layer,key=slot.split("/")
        raw=observed["legacy"]
        legacy=summarize([bit_row(raw.get("decoded_bits",[None]*32)[b],message[b],raw["status"],bit=b) for b in range(32)])
        legacy["votes"]=raw.get("votes")
        legacy["signed_vote_fractions"]=[None if not v.get("count") else (2*message[b]-1)*(v["ones"]-v["zeros"])/v["count"] for b,v in enumerate(raw.get("votes",[{}]*32))]
        result[slot]=dict(local=evaluate_local(observed["local"],message,codeword,encoded_arm=arm == "J_ECC",key_id=key),legacy=legacy)
    pairs={}
    for layer in m.LAYERS:
        for key in m.KEYS:
            for left,right in (("P","J0"),("P","J_ECC"),("J0","J_ECC")):
                a,b=[result[f"{arm}/{layer}/{key}"] for arm in (left,right)]
                row={field:compare(a["local"][field],b["local"][field]) for field in ("state","raw_data","ecc_data","raw_aggregate","ecc_aggregate")}
                row["legacy"]=compare(a["legacy"],b["legacy"])
                row["legacy_vote_fraction_changes"]=[dict(bit=i,baseline=x,candidate=y,change=None if x is None or y is None else y-x) for i,(x,y) in enumerate(zip(a["legacy"]["signed_vote_fractions"],b["legacy"]["signed_vote_fractions"]))]
                if right == "J_ECC":
                    row["baseline_raw_to_ECC_local"]=compare(a["local"]["raw_data"],b["local"]["ecc_data"])
                    row["baseline_raw_to_ECC_aggregate"]=compare(a["local"]["raw_aggregate"],b["local"]["ecc_aggregate"])
                pairs[f"{left}_to_{right}/{layer}/{key}"]=row
    return result,pairs
