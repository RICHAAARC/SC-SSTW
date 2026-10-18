"""Post-blind clean evaluation; no state/path selection."""
import math
def summarize(rows):
    n=len(rows);correct=sum(r["correct"] is True for r in rows);errors=sum(r["correct"] is False for r in rows)
    erasures=sum(r["status"] in ("MISSING","UNRESOLVED") for r in rows)
    return dict(denominator=n,rows=rows,correct=correct,errors=errors,erasures=erasures,
                unavailable=n-correct-errors-erasures,read_count=correct+errors,
                coverage=(correct+errors)/n,ber_on_read=errors/(correct+errors) if correct+errors else None)

def bit_row(value,target,status,**indices):
    available=status in ("READ","UNRESOLVED","MISSING")
    return dict(**indices,decoded=value,target=target,status="MISSING" if status=="READ" and value is None else status,
                correct=value==target if available and value is not None else None)

def compare(a,b):
    common=[(x,y) for x,y in zip(a["rows"],b["rows"]) if x["correct"] is not None and y["correct"] is not None]
    changes=[dict(bit=x.get("bit"),position=x.get("position"),old=x["decoded"],new=y["decoded"],
                  target=x["target"],old_status=x["status"],new_status=y["status"])
             for x,y in zip(a["rows"],b["rows"]) if x["decoded"]!=y["decoded"] or x["status"]!=y["status"]]
    return dict(denominator=a["denominator"],common_read_count=len(common),
        old_common_ber=sum(not x["correct"] for x,y in common)/len(common) if common else None,
        new_common_ber=sum(not y["correct"] for x,y in common)/len(common) if common else None,
        added_coverage=sum(x["decoded"] is None and y["decoded"] is not None for x,y in zip(a["rows"],b["rows"])),
        lost_coverage=sum(x["decoded"] is not None and y["decoded"] is None for x,y in zip(a["rows"],b["rows"])),
        introduced_errors=sum(x["correct"] is True and y["correct"] is False for x,y in zip(a["rows"],b["rows"])),
        repaired_errors=sum(x["correct"] is False and y["correct"] is True for x,y in zip(a["rows"],b["rows"])),changes=changes)

def evaluate(observations,message):
    if len(message)!=32 or any(type(x) is not int or x not in (0,1) for x in message):raise ValueError("32 evaluation truth bits required")
    results={}
    for slot,o in observations.items():
        s,p=o["state"],o["payload"]
        states=[bit_row(row.get("label"),k,row["status"],position=k) for k,row in enumerate(s["states"])]
        payload=[bit_row(p["decoded_bits"][b],message[b],p["status"],bit=b,segment=b//8) for b in range(32)]
        scores=s.get("pooled_scores")
        competitors=[dict(position=k,competitor=j,gap=None if scores is None else scores[k][k]-scores[k][j],
                          target_shortfall=None if scores is None else 1/math.sqrt(2)-scores[k][k]+scores[k][j])
                     for k in range(22) for j in range(22) if j!=k]
        vote_rows=[b for g in p["segments"] for b in g["bits"]]
        margins=[None if not v.get("count") else (2*message[b]-1)*(v["ones"]-v["zeros"])/v["count"] for b,v in enumerate(vote_rows)]
        results[slot]=dict(state=summarize(states),state_competitors=competitors,state_competitor_denominator=462,
            payload=summarize(payload),segments=p["segments"],signed_vote_fractions=margins,
            structural_missing_bits=0,planned_segments=4,scope="clean public-grid local32; no estimated edit-path benefit")
    pairs={}
    for layer in ("terminal","mp4"):
        for key in ("CORRECT","WRONG"):
            a,b=[results[f"{arm}/{layer}/{key}"] for arm in ("P","S")]
            pairs[f"{layer}/{key}"]=dict(state=compare(a["state"],b["state"]),payload=compare(a["payload"],b["payload"]),
                vote_fraction_changes=[dict(bit=i,baseline=x,candidate=y,change=None if x is None or y is None else y-x)
                                       for i,(x,y) in enumerate(zip(a["signed_vote_fractions"],b["signed_vote_fractions"]))])
    return results,pairs

