"""Small experiment records interface; no method selection, model or truth parsing."""
from pathlib import Path
import gzip,hashlib,importlib.metadata,json,os
from runtime.wan.provenance import source_identity
def digest(p):
    try:return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:return None  # Optional locating record only.
def read(p):
    p=Path(p);raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,indent=2,allow_nan=False)+"\n").encode()
    if p.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_bytes(raw);os.replace(tmp,p)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def keys(cfg):return (("K0",cfg["key"]),("K1",cfg["wrong_key"]))
def environment(cfg):
    result={}
    for name in (cfg.get("environment_pins") or {}):
        try:result[name]=importlib.metadata.version(name)
        except Exception:result[name]=None
    return result

def public_config(path):
    path=Path(path).resolve();cfg=read(path);cfg["_config_path"]=str(path)
    for field in ("preparation_config","oracle_config","posthoc_config"):
        sibling=path.with_name(path.stem+"_"+field.removesuffix("_config")+".json")
        if field not in cfg and sibling.is_file():cfg[field]=str(sibling)
        if field in cfg:
            value=Path(cfg[field]);cfg[field]=str(value if value.is_absolute() else path.parent/value)
    name=cfg.get("name")
    template=Path(__file__).parent/"configs"/(str(name)+".json")
    if not template.is_file():raise ValueError("unknown released entry configuration")
    fixed=read(template)
    for field in ("fixed_denominator","planned_calls","physical_upper_bounds"):
        if field in fixed and cfg.get(field)!=fixed[field]:raise ValueError("fixed public protocol field: "+field)
    if any(cfg.get("model",{}).get(k)!=fixed["model"].get(k) for k in ("id",)):raise ValueError("fixed Wan receiver model required")
    for key in ("id", "scaled_latent_factor", "method_dtype", "compute_dtype", "batch_frames", "replaces_wan_native_decoder"):
        if key in fixed.get("framewise_model", {}) and cfg.get("framewise_model", {}).get(key) != fixed["framewise_model"][key]:
            raise ValueError("fixed framewise interface field: " + key)
    if name=="video_trajectory_payload_framewise_sync_m05_v1":
        if cfg.get("target_margin")!=0.5 or cfg.get("media")!=dict(fps=8,crf=18,pix_fmt="yuv420p",codec="libx264"):raise ValueError("fixed M05 target and codec required")
        if cfg["source"]["shape"]!=[181,320,512,3]:raise ValueError("M05 fixed FULL181 geometry")
    elif name=="video_trajectory_receiver_estimated_align_v1":
        if sorted(x["frames"] for x in cfg["inputs"].values())!=[89,177,181]:raise ValueError("fixed 181/177/89 roster")
    elif name=="video_trajectory_internal_single_deletion_v1":
        if len(cfg["inputs"])!=2 or any(x["frames"]!=177 for x in cfg["inputs"].values()):raise ValueError("fixed two177 roster")
    else:raise ValueError("unknown released entry configuration")
    for x in cfg.get("inputs",{}).values():
        if x["shape"]!=[x["frames"],320,512,3]:raise ValueError("fixed receiver spatial geometry")
    return cfg

def source_metadata(root,cfg,default_config):
    provenance=source_identity(root)
    return dict(source_sha=provenance["git_commit"],source_clean=(not provenance["git_status"]) if provenance["git_commit"] else None,
        source_provenance=provenance,source_files=provenance["files"],
        config_sha256=digest(cfg.get("_config_path",default_config)))

class Store:
    def save(self):dump(self.output/"result.json",self.data)
    def failure(self,stage,exc):
        self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()
    def call(self,name,fn):
        self.data["calls"][name]["attempted"]+=1;self.save()
        value=fn();self.data["calls"][name]["completed"]+=1;self.save();return value


# Explicit execution fields only: version, source hash and file locators are absent.
DECISION_FIELDS = frozenset(("status", "decoded_bits", "votes", "mode", "oracle", "key_label", "key_id",
    "observation_id", "input_id", "view_id", "protocol", "frames", "received_frames", "R", "operation",
    "physical_read", "alias_of", "planned_final_bits", "planned_votes", "planned_time_bits",
    "planned_detailed_votes", "planned_time_bit_rows", "offset", "phase", "path", "family", "b", "k",
    "H0", "joint", "global", "global_", "reason", "received_index_map", "encode_id", "logical_slots",
    "top_offsets", "top_paths", "best_score", "gap", "sync_accepted", "truth_inputs", "truth_used",
    "full_singleton_geometry_only", "hypothesis", "nominal_source_map", "synthetic_output_indices",
    "dropped_received_indices", "inserted_gap_output_index", "operation_cost", "output_frames",
    "synthetic_outputs", "repeated_occurrences", "dropped_received", "shape", "dtype", "arm", "layer"))

def decision_values(row):
    if isinstance(row, list):return [decision_values(x) for x in row]
    if not isinstance(row, dict):return row
    return {k:(v if k in ("votes","decoded_bits") else decision_values(v)) for k,v in row.items() if k in DECISION_FIELDS
            and not (k=="path" and not isinstance(v,(dict,list)))}

def check_snapshot(path, groups=None, *, truth_inputs=False):
    """Read saved decisions before truth. Optional locating records never participate."""
    value=read(path)
    if not isinstance(value,dict) or value.get("truth_inputs") is not truth_inputs:
        raise ValueError("saved output has wrong blind/oracle information condition")
    for field,current in (groups or {}).items():
        saved=value.get(field)
        if not isinstance(saved,dict) or set(saved)!=set(current):
            raise ValueError("saved fixed slot set differs: "+field)
        for identity,row in current.items():
            if not isinstance(row,dict) or not isinstance(saved[identity],dict):raise ValueError("saved decision row must be an object")
            if decision_values(saved[identity])!=decision_values(row):
                raise ValueError("saved decision differs: "+field+"/"+identity)
    return value

def checked_detail(path,row):
    """Validate saved posthoc tables against their known support and read decision."""
    detail=read(path)
    if not isinstance(detail,dict):raise ValueError("payload detail must be an object")
    for field in ("R","received_frames"):
        if field in row and detail.get(field)!=row[field]:raise ValueError("payload detail support differs: "+field)
    complete=row.get("status")=="READ"
    if (detail.get("status")=="READ")!=complete:
        raise ValueError("payload detail read status differs from saved completion")
    R=detail.get("R")
    if type(R) is not int or R not in (22,44):raise ValueError("payload detail fixed R22/R44 support required")
    times=detail.get("time_bit_rows")
    if not isinstance(times,list) or len(times)!=32*R:raise ValueError("payload detail time-bit support must be complete")
    bits=detail.get("bit_rows")
    if complete and (not isinstance(bits,list) or len(bits)!=32):raise ValueError("payload detail final-bit support must be 32")
    if bits is not None and (not isinstance(bits,list) or len(bits)!=32):raise ValueError("payload detail final-bit support must be 32")
    def index(value,name,limit):
        if type(value) is not int or value not in range(limit):raise ValueError("payload detail invalid "+name)
        return value
    def support(item,time=False):
        if not isinstance(item,dict):raise ValueError("payload bit row must be an object")
        b=index(item.get("bit_index"),"bit index",32)
        if type(item.get("payload_channel")) is not int or item["payload_channel"]!=b//8:raise ValueError("payload bit channel mismatch")
        if "slot" in item and (type(item["slot"]) is not int or item["slot"]!=b%8):raise ValueError("payload bit slot mismatch")
        if time:
            t=index(item.get("receiver_latent_index"),"latent index",R+1)
            if t==0:raise ValueError("payload latent index starts at 1")
            if "nominal_stride_coordinate" in item and item["nominal_stride_coordinate"]!=4*t:raise ValueError("payload stride coordinate mismatch")
        else:t=None
        status=item.get("status")
        if complete and status not in (None,"READ","SCORED"):raise ValueError("successful detail contains failed/unknown bit row")
        if not complete and status in ("READ","SCORED"):raise ValueError("failed detail contains successful bit row")
        return (b,t) if time else b
    bit_map={}
    for item in bits or []:
        b=support(item)
        if b in bit_map:raise ValueError("duplicate final bit index")
        bit_map[b]=item
    time_map={}
    for item in times:
        key=support(item,True)
        if key in time_map:raise ValueError("duplicate time-bit index")
        time_map[key]=item
    if set(time_map)!={(b,t) for b in range(32) for t in range(1,R+1)}:raise ValueError("incomplete time-bit indices")
    if not complete:return detail  # Preserve failure/unknown; never infer a successful read.
    def counts(item,expected):
        values=tuple(item.get(k) for k in ("ones","zeros","count"))
        if any(type(v) is not int or v<0 for v in values):raise ValueError("payload vote counts must be nonnegative integers")
        one,zero,count=values
        if count!=expected or one+zero!=count:raise ValueError("payload vote count/support mismatch")
        decoded=item.get("decoded")
        if type(decoded) is not int or decoded not in (0,1):raise ValueError("payload decoded bit must be binary")
        if one!=zero and decoded!=int(one>zero):raise ValueError("payload decoded bit conflicts with saved counts")
        if "tie" in item and (type(item["tie"]) is not bool or item["tie"]!=(one==zero)):raise ValueError("payload tie/count mismatch")
        if "first_vote" in item:
            if type(item["first_vote"]) is not int or item["first_vote"] not in (0,1):raise ValueError("payload first vote must be binary")
            if one==zero and decoded!=item["first_vote"]:raise ValueError("payload saved tie decision mismatch")
        return values
    for b in range(32):
        final=counts(bit_map[b],30*R)
        temporal=[counts(time_map[b,t],30) for t in range(1,R+1)]
        if tuple(sum(x[i] for x in temporal) for i in range(3))!=final:raise ValueError("payload time/final vote counts differ")
    original=detail.get("original_readout",{})
    if not isinstance(original,dict):raise ValueError("payload original readout must be an object")
    for saved in (row,original):
        decoded=saved.get("decoded_bits")
        if decoded is not None:
            if not isinstance(decoded,list) or len(decoded)!=32 or any(type(v) is not int or v not in (0,1) for v in decoded):raise ValueError("saved decoded support must be 32 binary bits")
            if any(bit_map[b]["decoded"]!=decoded[b] for b in range(32)):raise ValueError("payload detail decision differs: decoded_bits")
        votes=saved.get("votes")
        if votes is not None:
            if not isinstance(votes,list) or len(votes)!=32:raise ValueError("saved vote support must be 32 rows")
            for b,vote in enumerate(votes):
                if not isinstance(vote,dict) or any(type(vote.get(k)) is not int or vote[k]!=bit_map[b][k] for k in ("ones","zeros","count")):
                    raise ValueError("payload detail decision differs: votes")
    return detail
