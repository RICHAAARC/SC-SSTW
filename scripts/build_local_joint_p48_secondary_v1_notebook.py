"""Build the report-only notebook for the single unresolved fixed CPU attempt."""
import ast
import io
import json
import zipfile
from scripts import build_local_joint_p48_joint_control_v1_notebook as p48

ROOT=p48.ROOT
NAME='local_joint_p48_secondary_v1'
CONFIG_PATH='experiments/wan_state_clock/configs/'+NAME+'.json'
DECISION_PATH='notebooks/'+NAME+'_cpu_solution.json'


def source_files():
    pending=[CONFIG_PATH,DECISION_PATH,'docs/'+NAME+'.md','scripts/build_'+NAME+'_notebook.py',
        'experiments/wan_state_clock/'+NAME+'_run.py','runtime/wan/'+NAME+'.py','main/tube_state/'+NAME+'.py']
    found=set()
    def include(module):
        parts=module.split('.')
        if parts[0] not in ('main','runtime','experiments','scripts'):return
        for i in range(1,len(parts)+1):
            for p in ('/'.join(parts[:i])+'.py','/'.join(parts[:i])+'/__init__.py'):
                if (ROOT/p).is_file() and p not in found:pending.append(p)
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        if path.endswith('.py'):
            for node in ast.walk(ast.parse((ROOT/path).read_text())):
                if isinstance(node,ast.Import):
                    for a in node.names:include(a.name)
                elif isinstance(node,ast.ImportFrom) and node.module:
                    module=node.module
                    if node.level:module='.'.join(path[:-3].split('/')[:-node.level]+[module])
                    include(module)
                    for a in node.names:include(module+'.'+a.name)
    # Imported builders refer to their configs/docs as data; retain their ordinary
    # source dependencies so this builder also works from the editable ZIP.
    found.update(p48.source_files())
    return sorted(found)


def portable_archive():
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in source_files():z.writestr(path,(ROOT/path).read_bytes())
    return out.getvalue()


def build_notebook():
    nb=p48.build_notebook();code=p48.p0.code
    nb['cells'][1]['source']=[(ROOT/('docs/'+NAME+'.md')).read_text()]
    config=json.loads((ROOT/CONFIG_PATH).read_text())
    s=''.join(nb['cells'][2]['source'])
    old=json.loads((ROOT/p48.CONFIG_PATH).read_text())
    s=s.replace('FIXED_CONFIG = '+repr(old),'FIXED_CONFIG = '+repr(config))
    s=s.replace(p48.NAME,NAME)
    s=s.replace("/resumable-p48-joint-control-v1')","/resumable-p48-joint-control-v1/secondary-direct-v1')")
    s=s.replace('Video-WM-P48-','Video-WM-P48-Secondary-')
    s=s.replace('Fixed P48 task:','P48 unresolved secondary report task:')
    nb['cells'][2]=code(s)
    s=''.join(nb['cells'][3]['source']).replace('runtime/wan/local_joint_p48_journal_v1.py','runtime/wan/'+NAME+'.py').replace(p48.NAME,NAME)
    nb['cells'][3]=code(s)
    nb['cells'][4]=code("""
        # The fixed CPU outcome is unresolved. Only CPU tensor/array readers are
        # needed; do not install diffusers/transformers or request CUDA/token.
        try:
            probe = 'import torch, numpy'
            if logged([PYTHON, '-u', '-c', probe], 'CPU_DEPENDENCY_IMPORT'):
                if logged([PYTHON, '-m', 'pip', 'install', 'torch', 'numpy'], 'CPU_DEPENDENCY_INSTALL'):
                    raise RuntimeError('CPU reader dependency installation failed')
                if logged([PYTHON, '-u', '-c', probe], 'CPU_DEPENDENCY_RECHECK'):
                    raise RuntimeError('CPU reader import failed')
        except BaseException as exc:
            fail('CPU_DEPENDENCIES', exc)
            raise
        print('Recorded fixed-target CPU solve: optimal_inaccurate. No solver rerun or GPU work.')
    """)
    nb['cells'][5]=code("""
        from runtime.wan.local_joint_p48_secondary_v1 import finalize_interrupted
        rc = None
        try:
            rc = logged([PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_p48_secondary_v1_run',
                         '--config', str(OUTPUT / 'fixed_config.json'), '--output', str(RUN_OUTPUT),
                         '--decision', str(WORKSPACE / 'notebooks/local_joint_p48_secondary_v1_cpu_solution.json')],
                        'SAVED_SECONDARY_REPORT_ONLY', cwd=WORKSPACE)
        finally:
            result = finalize_interrupted(RUN_OUTPUT, 'report reader interrupted; return code ' + str(rc))
            write_json(ATTEMPT / 'execution_receipt.json', dict(returncode=rc,
                       result_status=None if result is None else result['status'], automatic_retry=False,
                       model_calls=0, solver_calls=0))
        if result is None: raise RuntimeError('No report; retain notebook attempt logs')
        fixed_slots.update(status=result['status'], endpoints=result['endpoints'], objectives=result['objectives'])
        write_json(ATTEMPT / 'fixed_slots.json', fixed_slots)
    """)
    nb['cells'][6]=code("""
        result = json.loads((RUN_OUTPUT / 'result.json').read_text())
        print('Fixed CPU outcome:', result['status'], '; scientific PASS: False')
        print('New physical calls:', result['actual_new_stage_calls'], '; no model or solver rerun')
        print('CPU validation used 0 primary solves and 1 fixed-target secondary solve; this notebook uses 0 solves.')
        print('Original task retained at:', result['source_run'])
        print('Original + new physical calls:', result.get('original_call_count'), '+ 0')
        print('CANDIDATE:', result['endpoints']['CANDIDATE'])
        print('Secondary receipt:', result.get('solution'))
        print('Original planned 7-call candidate stage was NOT ENTERED:', result['physical_budget_if_qualified_candidate'])
        print('Failures:', result['failures']); print('Resume:', result['resume_policy'])
        print('Return report directory:', OUTPUT)
    """)
    for i,cell in enumerate(nb['cells']):cell['id']='p48-secondary-'+str(i)
    nb['metadata']['colab']['name']=NAME+'_colab.ipynb'
    return nb


def main():
    (ROOT/('notebooks/'+NAME+'_colab.ipynb')).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+'\n')
    (ROOT/('notebooks/'+NAME+'_portable_source.zip')).write_bytes(portable_archive())


if __name__=='__main__':main()
