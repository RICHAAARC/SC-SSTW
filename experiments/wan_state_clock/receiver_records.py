"""Small experiment records interface; no method selection, model or truth parsing."""
from pathlib import Path
import gzip,hashlib,importlib.metadata,json,os
from runtime.wan.provenance import source_identity
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
    for name in cfg["environment_pins"]:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
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
    for field in ("fixed_denominator","planned_calls","physical_upper_bounds","environment_pins","framewise_model"):
        if field in fixed and cfg.get(field)!=fixed[field]:raise ValueError("fixed public protocol field: "+field)
    if any(cfg.get("model",{}).get(k)!=fixed["model"].get(k) for k in ("id","revision")):raise ValueError("fixed Wan receiver model required")
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
