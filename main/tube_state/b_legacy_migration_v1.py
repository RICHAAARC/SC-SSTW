"""B existing C73 carrier operator control; not an equivalent old writer migration."""
from . import c73_identity_v1 as identity

ARMS=("P0","S_base","S_ID")
LAYERS=("rgb8","mp4")
HISTORIES=("full","local")
KEY_IDS=("key00","key01")
TARGETS=((61,16),(109,28))


def plan():
    return [dict(row=f"row{i:02d}",arm=arm,layer=layer,history=history,target=target,key=key)
            for i,(arm,layer,history,target,key) in enumerate(
                (a,l,h,t,k) for a in ARMS for l in LAYERS for h in HISTORIES
                for t in range(2) for k in KEY_IDS)]


def unavailable(status,latent_time):
    record=identity.unavailable(status,local=True)["vectors"][0]
    record["latent_time"]=latent_time
    return record


def observe_full(latent,book):
    identity.legacy.validate_latent(latent)
    times=[t for a,t in TARGETS]
    roi,pooled=identity.project_times(latent,times,book)
    result=[]
    for i,time in enumerate(times):
        row=identity.classify(pooled[i],book["words"])
        row.update(latent_time=time,roi_scores=roi[i].tolist())
        result.append(row)
    return result


def observe_local(latent,book):
    return identity.observe(latent,book,local=True)["vectors"][0]
