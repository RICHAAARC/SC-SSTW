"""Source identity over the actual isolated import/config closure."""
from pathlib import Path
import ast,hashlib,json,subprocess

def source_identity(root,entry,config_paths):
    root=Path(root).resolve();todo=[entry];seen=set()
    while todo:
        rel=todo.pop();p=root/rel
        if rel in seen or not p.is_file():continue
        seen.add(rel)
        if p.suffix!=".py":continue
        for parent in p.relative_to(root).parents:
            init=(parent/"__init__.py").as_posix()
            if (root/init).is_file() and init not in seen:todo.append(init)
        for n in ast.walk(ast.parse(p.read_text())):
            mods=[]
            if isinstance(n,ast.Import):mods=[x.name for x in n.names]
            elif isinstance(n,ast.ImportFrom) and n.module:mods=[n.module]+[n.module+"."+x.name for x in n.names]
            for module in mods:
                if module.startswith(("main.","runtime.","experiments.")):
                    candidate=module.replace(".","/")+".py"
                    if (root/candidate).is_file() and candidate not in seen:todo.append(candidate)
    files={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in sorted(seen)}
    config_files={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in config_paths}
    content=hashlib.sha256(json.dumps(dict(files=files,config_files=config_files),sort_keys=True,separators=(",",":")).encode()).hexdigest()
    result=dict(kind="content_identity",source_sha=None,source_clean=None,source_files=files,config_files=config_files,content_sha256=content)
    tracked=list(files)+[str(Path(p).resolve().relative_to(root)) for p in config_paths if Path(p).resolve().is_relative_to(root)]
    if (root/".git").exists():
        result.update(kind="git_checkout",source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),source_clean=not bool(subprocess.check_output(["git","status","--porcelain","--",*tracked],cwd=root,text=True).strip()))
    return result
