"""Build the actual fixed P48 notebook and ordinary editable source ZIP."""
import ast
import io
import json
from pathlib import Path
import zipfile
from scripts import build_local_joint_p0_input_vjp_v1_notebook as p0

ROOT=p0.ROOT
NAME="local_joint_p48_joint_control_v1"
CONFIG_PATH="experiments/wan_state_clock/configs/"+NAME+".json"


def source_files():
    # Include import dependencies for portability; no runtime fingerprint chain.
    seeds=list(p0.PORTABLE_FILES)+[CONFIG_PATH,"docs/"+NAME+".md","scripts/build_"+NAME+"_notebook.py",
        "experiments/wan_state_clock/"+NAME+"_run.py","runtime/wan/"+NAME+".py","main/tube_state/"+NAME+".py"]
    found=set(); pending=list(seeds)
    def include(module):
        parts=module.split(".")
        if not parts or parts[0] not in ("main","runtime","experiments","scripts"): return
        for i in range(1,len(parts)+1):
            for rel in ("/".join(parts[:i])+".py","/".join(parts[:i])+"/__init__.py"):
                if (ROOT/rel).is_file() and rel not in found: pending.append(rel)
    while pending:
        name=pending.pop()
        if name in found: continue
        found.add(name)
        if not name.endswith(".py"): continue
        tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names: include(alias.name)
            elif isinstance(node,ast.ImportFrom) and node.module:
                module=node.module
                if node.level:
                    parent=name[:-3].split("/")[:-node.level]
                    module=".".join(parent+[module])
                include(module)
                for alias in node.names: include(module+"."+alias.name)
    return sorted(found)


def portable_archive():
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_files(): archive.writestr(name,(ROOT/name).read_bytes())
    return out.getvalue()


def build_notebook():
    nb=p0.build_notebook(); config=json.loads((ROOT/CONFIG_PATH).read_text()); old=json.loads((ROOT/p0.CONFIG_PATH).read_text())
    for cell in nb["cells"]:
        if cell["cell_type"]!="code": continue
        s="".join(cell["source"]).replace("FIXED_CONFIG = "+repr(old),"FIXED_CONFIG = "+repr(config))
        for a,b in ((p0.NAME,NAME),("Local-Joint-State-Payload-V1-P0-Input-VJP","Local-Joint-State-Payload-V1-P48-Joint-Control"),
            ("resumable-p0-input-vjp-v1","resumable-p48-joint-control-v1"),("Video-WM-P0-","Video-WM-P48-"),
            ("P0_FIXED_TWO_BRANCHES","P48_FIXED_JOINT_CONTROL"),("full-size P0","fixed P48"),("P0 status:","P48 status:"),
            ("Fixed P0 task:","Fixed P48 task:"),("No recoverable P0 result","No recoverable P48 result"),
            ("branches={b: dict(status='MISSING', reason='not_run') for b in ('conditional', 'unconditional')}",
             "endpoints={b: dict(status='MISSING', reason='not_run') for b in ('ZERO', 'ANCHOR', 'CANDIDATE')}"),
            ("branches=result['branches']","endpoints=result['endpoints']"),
            ("from diffusers import WanTransformer3DModel","from diffusers import WanTransformer3DModel, AutoencoderKLWan; import cvxpy, clarabel"),
            ("'safetensors', 'huggingface-hub'","'safetensors', 'huggingface-hub', 'cvxpy', 'clarabel'")):
            s=s.replace(a,b)
        cell["source"]=s.splitlines(keepends=True)
        s=s.replace("from runtime.wan."+NAME+" import needs_model_work","from runtime.wan.local_joint_p48_journal_v1 import needs_model_work")
        s=s.replace("from runtime.wan."+NAME+" import finalize_interrupted","from runtime.wan.local_joint_p48_journal_v1 import finalize_interrupted")
        if "fixed_slots = dict" in s:
            s=s.replace("fixed_slots = dict(status='SETUP_STARTED',", "fixed_slots = dict(status='SETUP_STARTED', objectives={n: dict(status='MISSING') for n in ('S','P','W','Q')},")
        s=s.replace("endpoints=result['endpoints'], counts=result['counts']", "endpoints=result['endpoints'], objectives=result['objectives'], counts=result['counts']")
        cell["source"]=s.splitlines(keepends=True)
    nb["cells"][1]["source"]=[(ROOT/("docs/"+NAME+".md")).read_text()]
    # P48 alone keeps ordinary source bytes beside the durable task, not inside
    # RUN_OUTPUT (whose contents are the physical execution/recovery state).
    nb["cells"][3]=p0.code("""
        import io, urllib.request, urllib.error, zipfile
        SOURCE_CACHE = OUTPUT / 'source_package.zip'
        try:
            downloaded = False
            try:
                with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
                    package = response.read()
                downloaded = True
                source_location = SOURCE_URL
            except Exception as network_error:
                try:
                    package = SOURCE_CACHE.read_bytes()
                except OSError as cache_error:
                    raise RuntimeError('Source download unavailable: ' + repr(network_error)
                                       + '; source cache unreadable: ' + repr(cache_error)) from cache_error
                source_location = str(SOURCE_CACHE)
                print('Network source unavailable; loading saved source ZIP:', repr(network_error))
            WORKSPACE.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(package)) as archive:
                for entry in ('runtime/wan/local_joint_p48_journal_v1.py',
                              'experiments/wan_state_clock/local_joint_p48_joint_control_v1_run.py'):
                    try:
                        archive.read(entry).decode('utf-8')
                    except (KeyError, OSError, ValueError, UnicodeError) as entry_error:
                        raise RuntimeError('Source entry unreadable: ' + entry + ': ' + repr(entry_error)) from entry_error
                for member in archive.infolist():
                    target = (WORKSPACE / member.filename).resolve()
                    if not target.is_relative_to(WORKSPACE.resolve()):
                        raise ValueError('unsafe source ZIP path: ' + member.filename)
                    if member.is_dir(): target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            if downloaded:
                try:
                    temporary = SOURCE_CACHE.with_suffix('.zip.tmp')
                    temporary.write_bytes(package)
                    os.replace(temporary, SOURCE_CACHE)
                except OSError as cache_error:
                    print('Source loaded, but saving its offline cache failed:', repr(cache_error))
            for module_name in list(sys.modules):
                if module_name.split('.', 1)[0] in ('main', 'runtime', 'experiments'):
                    sys.modules.pop(module_name, None)
            importlib.invalidate_caches()
            sys.path.insert(0, str(WORKSPACE))
            print('Source location:', source_location, WORKSPACE)
        except BaseException as exc:
            fail('SOURCE_LOAD', exc)
            raise
    """)
    nb["cells"][-1]=p0.code("""
        result = json.loads((RUN_OUTPUT / 'result.json').read_text())
        print('Status:', result['status'], '; scientific PASS: False')
        print('Expected budget:', result['expected_cost'], '; P0 sunk separately:', result['p0_sunk_cost'])
        print('Actual calls:', result['counts'], 'blocks:', result['block_counts'], 'chunks:', result['chunk_counts'])
        print('Block counting scope:', result['block_count_scope'])
        for name, row in result['endpoints'].items(): print(name, row)
        for name, row in result['objectives'].items(): print(name, row)
        print('SOCP:', result.get('solution')); print('Finite comparison:', result.get('comparison'))
        print('Resources:', result['resources']); print('Failures retained:', result['failures'])
        print('Current attempt:', result.get('current_attempt')); print(result['ceiling'])
        print('Return whole task directory:', OUTPUT)
    """)
    for i,cell in enumerate(nb["cells"]): cell["id"]="p48-joint-control-"+str(i)
    nb["metadata"]["colab"]["name"]=NAME+"_colab.ipynb"
    return nb


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__=="__main__": main()
