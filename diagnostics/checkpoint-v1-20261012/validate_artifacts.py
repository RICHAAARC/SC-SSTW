from pathlib import Path
import ast,json,os,subprocess,sys,tempfile,zipfile
root=Path('/home/richar/projects/Video-WM/worktrees/Paper-Results-V1')
sys.path.insert(0,str(root))
from experiments.paper_results_v1.build_confirmation_v1_colab import build_notebook,OUTPUT
from experiments.paper_results_v1.companion_source import companion_files
notebook=build_notebook()
OUTPUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
code=[c for c in notebook['cells'] if c['cell_type']=='code']
for c in code:
 ast.parse(''.join(c['source']))
 assert c['outputs']==[] and c['execution_count'] is None
assert ''.join(code[0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
zip_path=root/'notebooks/paper_results_v1_companion.zip'
expected=companion_files()
cli=[]
with zipfile.ZipFile(zip_path) as z:
 assert set(z.namelist())=={name for name,_ in expected}
 for name,path in expected:
  assert z.read(name)==path.read_bytes(),name
  if name.endswith('.py'):ast.parse(z.read(name).decode('utf-8'),filename=name)
 with tempfile.TemporaryDirectory(prefix='checkpoint-source-') as temp:
  source=Path(temp); z.extractall(source)
  env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
  for module,config,output,extra in [
   ('real_cli','real_eval.adopted.json','generation',[]),
   ('attack_cli','confirmation_v1.adopted.json','attack',[]),
   ('confirmation_v1_cli','confirmation_v1.adopted.json','confirmation',['--attack-output',str(source/'attack')])
  ]:
   command=[sys.executable,'-B','-m','experiments.paper_results_v1.'+module,'--config',str(source/'experiments/paper_results_v1'/config),'--output',str(source/output),'--phase','init',*extra]
   run=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True)
   assert run.returncode==0,(command,run.stderr)
   cli.append({'module':module,'phase':'init','no_git_source':True,'exit_code':run.returncode})
  state=json.loads((source/'confirmation/confirmation_state.json').read_text())
  assert len(state['rows'])==3600 and sum(r['status']=='PLANNED' for r in state['rows'])==2880
record={'status':'PASS_CPU_STATIC_ONLY','notebook':str(OUTPUT),'clean_code_cells':len(code),'companion':str(zip_path),'companion_files':len(expected),'companion_bytes':zip_path.stat().st_size,'notebook_builder_and_zip_synced':True,'no_git_cli_checks':cli,'model_gpu_codec_drive_execution':False}
out=root/'diagnostics/checkpoint-v1-20261012'
(out/'artifact_validation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
