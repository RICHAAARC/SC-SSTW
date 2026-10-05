"""Fixed R44 vote evidence; no runtime, experiment, model, or truth inputs."""
from collections import Counter
import numpy as np
R=44
CHANNELS=4
COORDINATES=240
BITS=32
VOTES_PER_READ=42240
TIME_BITS_PER_READ=1408

def _counter(row):
    values=((np.asarray(row,dtype=np.int8)+1)//2).tolist()
    ones=sum(values); count=len(values)
    return dict(decoded=int(Counter(values).most_common(1)[0][0]),ones=ones,
                zeros=count-ones,count=count,margin=abs(2*ones-count),
                normalized_margin=abs(2*ones-count)/count,
                first_vote=values[0],tie=2*ones==count)

def detailed_votes(signed_votes, coordinate_order, zero_mask):
    """[channel, receiver latent time 1..44, original coordinate-list index].

    signed_vote=2*int(FFT.real>0)-1: exact coefficient zero gives -1, not
    erasure. A bit's sequence is time-major, then coords[bit::8], exactly the
    unchanged reader's Counter encounter order. Nominal stride coordinates
    describe receiver indexing, NOT causal receptive fields or source frames.
    """
    votes=np.asarray(signed_votes)
    zeros=np.asarray(zero_mask)
    if votes.shape!=(4,44,240) or not np.isin(votes,[-1,1]).all():
        raise ValueError("fixed finite binary polarity tensor required")
    if zeros.shape!=votes.shape or not np.isin(zeros,[0,1]).all():
        raise ValueError("exact-zero coefficient mask required")
    if np.any(votes[zeros.astype(bool)]!= -1):
        raise ValueError("FFT exact zero must remain the original zero vote")
    coords=[[int(h),int(w)] for h,w in coordinate_order]
    if len(coords)!=240 or len({tuple(c) for c in coords})!=240:
        raise ValueError("original 240 unique frequency positions required")
    bits=[]; times=[]
    for ch in range(4):
        for bit in range(8):
            seq=votes[ch,:,bit::8]
            bits.append(dict(bit_index=ch*8+bit,payload_channel=ch,slot=bit,
                             **_counter(seq.reshape(-1))))
            for t in range(44):
                times.append(dict(receiver_latent_index=t+1,nominal_stride_coordinate=4*(t+1),
                                  bit_index=ch*8+bit,payload_channel=ch,slot=bit,
                                  **_counter(seq[t])))
    return dict(status="READ",R=44,shape=[4,44,240],
        axes=["payload_channel","receiver_latent_index_1_to_44","coordinate_list_index"],
        coordinate_order=coords,receiver_latent_indices=list(range(1,45)),
        signed_votes=votes.astype(np.int8).tolist(),zero_coefficient_mask=zeros.astype(np.int8).tolist(),
        signed_vote_definition="2*int(FFT.real>0)-1; exact zero maps to -1",
        flatten_order="For ch then bit: signed_votes[ch,:,bit::8].reshape(-1), C order",
        time_semantics="Receiver latent indices and nominal stride4 coordinates only; no independent four-frame support or definite source-frame assignment",
        bit_rows=bits,time_bit_rows=times,detailed_vote_count=VOTES_PER_READ,
        time_bit_count=TIME_BITS_PER_READ,truth_inputs=False)

def missing_detail(error):
    return dict(status="FAILED",error=str(error),shape=[4,44,240],R=44,
                planned_detailed_votes=VOTES_PER_READ,actual_detailed_votes=0,
                signed_votes=None,truth_inputs=False,
                time_bit_rows=[dict(receiver_latent_index=t,bit_index=b,
                                    payload_channel=b//8,status="FAILED",error=str(error))
                               for b in range(32) for t in range(1,45)])

def join_truth(detail, expected):
    """Reporting only; callers must durably seal blind evidence first."""
    if len(expected)!=32 or any(x not in (0,1) for x in expected):
        raise ValueError("fixed 32-bit reporting truth required")
    if detail["status"]!="READ":
        return dict(status="FAILED",error=detail.get("error"),
                    bit_rows=[dict(bit_index=i,expected=expected[i],status="FAILED") for i in range(32)],
                    time_bit_rows=[dict(x,status="FAILED") for x in detail["time_bit_rows"]])
    def join(row):
        bit=expected[row["bit_index"]]
        return dict(row,expected=bit,bit_error=int(row["decoded"]!=bit),
                    signed_normalized_margin=(2*bit-1)*(row["ones"]-row["zeros"])/row["count"])
    bits=[join(x) for x in detail["bit_rows"]]
    return dict(status="EVALUATED_TRUTH",bit_rows=bits,
                time_bit_rows=[join(x) for x in detail["time_bit_rows"]],
                bit_errors=sum(x["bit_error"] for x in bits),
                error_bits=[x["bit_index"] for x in bits if x["bit_error"]],
                truth_signed_vote_reconstruction="(2*expected_bit-1)*blind signed_vote, with bit=channel*8+coordinate_list_index%8")

