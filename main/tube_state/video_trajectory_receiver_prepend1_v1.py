"""Fixed public clip operations and descriptive post-seal comparisons."""
from collections import Counter
OPERATIONS=("IDENTITY","PREPEND1_DROP1","TAIL_REPEAT")

def received_index_map(operation):
    """Map each of 177 output frames to a received-clip index; no source truth."""
    if operation=="IDENTITY":return list(range(177))
    if operation=="PREPEND1_DROP1":return [0]+list(range(176))
    if operation=="TAIL_REPEAT":return list(range(176))+[175]
    raise ValueError("fixed public operation required")

def operation_receipt(operation):
    indices=received_index_map(operation)
    counts=Counter(indices)
    return dict(operation=operation,received_index_map=indices,
        # PREPEND inserts the synthetic copy before the retained original clip[0].
        synthetic_duplicate_output_indices=([0] if operation=="PREPEND1_DROP1" else [176] if operation=="TAIL_REPEAT" else []),
        duplicated_received_indices=[i for i,n in sorted(counts.items()) if n>1],
        dropped_received_indices=[i for i in range(177) if i not in counts])

def summarize(joined):
    """Reporting-only summaries over ALL time-bit rows, preserving ties."""
    if joined["status"]!="EVALUATED_TRUTH":
        return dict(status="FAILED",error=joined.get("error","read unavailable"))
    bits=joined["bit_rows"];rows=joined["time_bit_rows"]
    def summary(values):
        return dict(rows=len(values),decoded_errors=sum(x["bit_error"] for x in values),
            signed_normalized_margin_mean=sum(x["signed_normalized_margin"] for x in values)/len(values),
            negative=sum(x["signed_normalized_margin"]<0 for x in values),
            zero=sum(x["signed_normalized_margin"]==0 for x in values),
            positive=sum(x["signed_normalized_margin"]>0 for x in values),
            ties=sum(x["tie"] for x in values))
    return dict(status="DESCRIPTIVE_POSTSEAL",final=summary(bits),
        by_time=[dict(receiver_latent_index=t,**summary([x for x in rows if x["receiver_latent_index"]==t])) for t in range(1,45)],
        by_channel=[dict(payload_channel=c,final=summary([x for x in bits if x["payload_channel"]==c]),
            time_bits=summary([x for x in rows if x["payload_channel"]==c])) for c in range(4)],
        by_time_channel=[dict(receiver_latent_index=t,payload_channel=c,
            **summary([x for x in rows if x["receiver_latent_index"]==t and x["payload_channel"]==c]))
            for t in range(1,45) for c in range(4)],
        time_semantics="Receiver indices only; local diagnostics do not replace the canonical 1320-vote Counter")

def compare(original,changed):
    """Fixed same-run changed-minus-original reporting; never selects an arm."""
    if any(x["status"]!="EVALUATED_TRUTH" for x in (original,changed)):
        return dict(status="FAILED",error="paired fixed read unavailable; comparison slot retained")
    def delta(a,b):
        return dict(signed_normalized_margin_delta=b["signed_normalized_margin"]-a["signed_normalized_margin"],
            decoded_error_delta=b["bit_error"]-a["bit_error"])
    a={x["bit_index"]:x for x in original["bit_rows"]}
    b={x["bit_index"]:x for x in changed["bit_rows"]}
    ta={(x["receiver_latent_index"],x["bit_index"]):x for x in original["time_bit_rows"]}
    tb={(x["receiver_latent_index"],x["bit_index"]):x for x in changed["time_bit_rows"]}
    time_channel=[]
    for t in range(1,45):
        for c in range(4):
            values=[delta(ta[t,i],tb[t,i]) for i in range(c*8,c*8+8)]
            time_channel.append(dict(receiver_latent_index=t,payload_channel=c,
                signed_normalized_margin_mean_delta=sum(x["signed_normalized_margin_delta"] for x in values)/8,
                decoded_error_delta=sum(x["decoded_error_delta"] for x in values)))
    return dict(status="DESCRIPTIVE_POSTSEAL",direction="changed minus S1_ORIGINAL",
        bit_error_delta=changed["bit_errors"]-original["bit_errors"],
        bit_rows=[dict(bit_index=i,payload_channel=i//8,**delta(a[i],b[i])) for i in range(32)],
        by_time_channel=time_channel,selection_or_retry=False,
        evidence_ceiling="Same receiver index is a diagnostic label, not proof of matched source receptive fields or unique phase causality.")
