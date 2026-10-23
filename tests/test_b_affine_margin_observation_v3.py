"""Model-free engineering tests; never invoke real construct, VAE, codec or FFT."""
from collections import Counter
from pathlib import Path
import ast,json,importlib.util,shutil,subprocess,sys
import numpy as np
import pytest
pytestmark=pytest.mark.unit
from main.tube_state import b_affine_margin_plan_v3 as p,b_affine_margin_v3 as c
from runtime.wan import b_affine_margin_observation_v3 as r

class Stub:
 def __init__(self,fail_construct=False,fail_raw=False,interrupt=False):self.calls=Counter();self.fail_construct=fail_construct;self.fail_raw=fail_raw;self.interrupt=interrupt;self.tagcount=0
 def op(self,name,value=None):self.calls[name]+=1;return value
 def initialize_PN(self):return self.op('PN',dict(status='STUB'))
 def load_native(self,path):return self.op('load',str(path))
 def construct_new(self,a,b,record):
  self.op('construct');record('native_receipt.json',dict(status='STUB_NO_REAL_CONSTRUCTION'))
  if self.interrupt:raise KeyboardInterrupt('stub interrupt consumes attempt')
  if self.fail_construct:raise ValueError('stub native unresolved')
  return 'NEW_STUB'
 def save_tensor(self,path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text('STUB')
 def load_tensor(self,path):
  if not Path(path).is_file():raise FileNotFoundError(path)
  return str(path)
 def load_vae(self):return self.op('VAE',dict(status='STUB'))
 def release_vae(self):self.op('release')
 def decode_PT(self,z):return self.op('D',dict(preclip='pixels',float_rgb='pixels'))
 def quantize(self,x):return 'rgb8'
 def save_mp4(self,pixels,path):self.op('encode');Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text('STUB');return dict(status='STUB')
 def probe(self,path):return self.op('probe',dict(status='STUB'))
 def read_media(self,path,probe,length):return self.op('decode',length)
 def prepare_full(self,pixels):return pixels
 def received104(self,pixels,attack):return 104
 def window45(self,*args):return 45
 def encode_full(self,x):return self.op('E181',dict(local=False))
 def encode_E45(self,x):return self.op('E45',dict(local=True))
 def tag_lll(self,z,local):return local
 def tag_scores(self,local,key,rows):
  self.tagcount+=1
  if self.fail_raw and self.tagcount==3:raise ValueError('one local raw failure')
  return dict(status='OBSERVED',rows=[dict(row=j,status='OBSERVED',pilot_scores=[0.]*16,data_scores=[0.]*32) for j in rows],truth_inputs=False,message_inputs=False)
 def payload_spectrum(self,z):return self.op('FFT',False)
 def local_payload_spectrum(self,z):return self.op('FFT',True)
 def votes(self,local):
  self.op('keyed');n=11 if local else 44;v=dict(status='STUB_PAYLOAD',signed_votes=np.ones((4,n,240),np.int8).tolist(),zero_coefficient_mask=np.zeros((4,n,240),np.int8).tolist(),local_cells=[dict(bits=[dict(decoded=1) for _ in range(8)]) for _ in range(n)],truth_inputs=False,message_inputs=False)
  if not local:v['decoded_bits']=[1]*32
  return v
 def payload_votes(self,s,key):return self.votes(False)
 def local_payload_votes(self,s,key):return self.votes(True)
 def infer(self,queries,save_arrays):
  self.op('blind');O,D,S=c.state_grid();out=[]
  for q in queries:
   save_arrays(q['id'],dict(O=O,d=D,s=S,D_score=np.zeros(len(O)),L_score=np.zeros(len(O)),R_score=np.ones(len(O)),gain_pilot=np.zeros(len(O)),gain_label=np.zeros(len(O)),all_candidate_c_pilot=np.zeros((len(O),16)),all_candidate_c_label=np.zeros((len(O),32)),candidate_c_pilot=np.empty((0,16)),candidate_c_label=np.empty((0,32)),corrected_pilot=np.empty((0,96,16)),corrected_label=np.empty((0,96,32)),candidate_pure_ids=np.empty((0,96),int),H=np.array([],int)))
   value=dict(decoded=[None]*32,selected_refs=[],read=0)
   out.append(dict(id=q['id'],status='UNRESOLVED_INPUT' if q['missing'] else 'STUB_REFERENCE',model_domain='MODEL_DOMAIN_UNVALIDATED',reference_set_paths=0,payload_E=value,payload_U=value,common_frames=[None]*104,common_g=[],corrected_row_evidence=dict(paths=0,rows=[]),scientific_PASS=None))
  return out
 def preclip(self,x):return self.op('clip',dict(status='STUB'))
 def paired_quality(self,*args):return self.op('quality',dict(status='STUB'))

@pytest.fixture
def source(tmp_path):
 path=tmp_path/'protocol.json';path.write_text(json.dumps(dict(config=dict(writer=dict(bits=[1]*32)))));return path

def execute(tmp_path,source,stub):return r.run('P0_PLACEHOLDER','OLD_PLACEHOLDER',tmp_path/'out',source,backend=stub)
def test_complete_budget_and_readonly(tmp_path,source):
 stub=Stub();result=execute(tmp_path,source,stub);assert result['status']=='COMPLETED_DESCRIPTION_ONLY';assert stub.calls['construct']==1
 assert {k:stub.calls[k] for k in ('VAE','D','E181','E45','FFT','keyed','encode','probe','decode','quality')}==dict(VAE=1,D=3,E181=3,E45=144,FFT=147,keyed=294,encode=9,probe=9,decode=9,quality=6)
 assert (tmp_path/'out/markers/blind_complete.json').is_file();assert (tmp_path/'out/post_truth_access.json').is_file()
 before=stub.calls.copy();second=execute(tmp_path,source,stub);assert second['status']=='READ_ONLY_EXISTING_ATTEMPT' and stub.calls==before
 assert r.read(tmp_path/'out/evaluation.json')['alignment']['E']['planned']==384

def test_new_unresolved_keeps_other_arms_and_denominator(tmp_path,source):
 stub=Stub(fail_construct=True);result=execute(tmp_path,source,stub);assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED';assert stub.calls['construct']==1 and stub.calls['D']==2 and stub.calls['FFT']==98 and stub.calls['keyed']==196
 assert (tmp_path/'out/evaluation.json').is_file();ledger=r.read(tmp_path/'out/ledger.json');assert ledger['NEW/full/D']['attempts']==0
 assert r.read(tmp_path/'out/raw/NEW/crop104/w0/p0/key00.json')['journal_status']=='UNAVAILABLE_DEPENDENCY'
 before=stub.calls.copy();execute(tmp_path,source,stub);assert stub.calls==before

def test_one_raw_failure_is_explicit_but_other_queries_post(tmp_path,source):
 stub=Stub(fail_raw=True);result=execute(tmp_path,source,stub);assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED';assert (tmp_path/'out/evaluation.json').is_file()
 value=r.read(tmp_path/'out/raw/P0/crop104/w0/p0/key00.json');assert value['status']=='TECHNICAL_UNRESOLVED' and value['journal_status']=='FAILED'
 assert r.read(tmp_path/'out/blind/P0/crop104/key00/decision.json')['status']=='UNRESOLVED_INPUT'
 assert r.read(tmp_path/'out/blind/NEW/delete13_s52/key00/decision.json')['status']=='STUB_REFERENCE'

def test_blind_persistence_failure_prevents_truth(tmp_path,source,monkeypatch):
 saved=r.save
 def fail(path,value):
  if str(path).endswith('/decision.json') and value.get('status')=='STUB_REFERENCE':raise OSError('blind storage failed')
  return saved(path,value)
 monkeypatch.setattr(r,'save',fail);execute(tmp_path,source,Stub());assert not (tmp_path/'out/post_truth_access.json').exists();assert not (tmp_path/'out/evaluation.json').exists()

def test_interruption_consumes_cpu_attempt(tmp_path,source):
 stub=Stub(interrupt=True);result=execute(tmp_path,source,stub);assert result['status']=='INTERRUPTED' and stub.calls['construct']==1 and stub.calls['VAE']==0
 before=stub.calls.copy();execute(tmp_path,source,stub);assert stub.calls==before

def test_zero_and_payload_order_no_truth_interface():
 records=c.payload_records([dict(signed_votes=np.ones((4,11,240),np.int8),zero_coefficient_mask=np.zeros((4,11,240),np.int8)) for _ in range(24)])
 out=c.recover(records,list(range(104)),set(),False);assert len(out['decoded'])==32 and out['truth_inputs'] is False
 assert list(__import__('inspect').signature(c.infer_many).parameters)==['queries','save_arrays']
 tree=ast.parse(__import__('inspect').getsource(c.infer_many));assert not any(isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value in ('truth','P0','writer_center','message') for x in ast.walk(tree))

def test_builder_notebook_and_nogit_cli(tmp_path):
 root=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location('v3builder',root/'scripts/build_b_affine_margin_observation_v3_notebook.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 files=b.source_files(root)
 for name in files:
  dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
 path=b.build(tmp_path);nb=json.loads(path.read_text());actual=json.loads((root/'notebooks/b_affine_margin_observation_v3.ipynb').read_text());assert nb==actual
 codes=[''.join(x['source']) for x in nb['cells'] if x['cell_type']=='code'];assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
 for text in codes:ast.parse(text)
 text='\n'.join(codes);assert 'b_affine_margin_observation_v3_run' in text and 'EXISTING_ARTIFACTS' in text and 'cvxpy,clarabel' in text
 assert 'b64decode' not in text and 'sha256' not in text
 completed=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_affine_margin_observation_v3_run','--help'],cwd=tmp_path,text=True,capture_output=True);assert completed.returncode==0,completed.stderr


def test_post_corrected_and_erasure_classes_are_descriptive():
 from runtime.wan.b_affine_margin_evaluation_v3 import corrected_description,classify_bit
 bk=c.book('watermark');g=3;cp=np.linspace(-.1,.1,16);cd=np.linspace(-.2,.2,32);yp=(2*(np.array(bk['pilot_bits'])^(g%2))-1)*.05+cp;yd=(2*(c.WORDS[g]^np.array(bk['mask']))-1)*.06+cd
 row=corrected_description(yp[None],yd[None],cp,cd,'watermark',[list(range(8*g-3,8*g+5))])[0]
 assert row['target_distance']==0 and row['decoded_valid_ID']==g and row['pilot_coherent_mean']>0 and row['receiver_output_changed'] is False
 value=classify_bit(dict(bit=31,read=False),'E',dict(reference_set_paths=0),{},list(range(104)))
 assert value['coverage_class']=='PHYSICALLY_ABSENT_GROUP'
 common,gg,audit,arrays=c.support(np.array([],int),None,None,None,None,None,None,None,None,None,None,None)
 assert arrays['corrected_label'].shape==(0,96,32) and audit['paths']==0 and len(common)==104


def test_per_query_summary_keeps_control_outputs_separate():
 from runtime.wan.b_affine_margin_evaluation_v3 import summarize_alignment
 align=[]
 for name in p.families():
  for variant in ('U','E','O_geom','O_admitted'):
   for bit in range(32):
    read=variant=='E' and ((name=='NEW/crop104/key00' and bit==0) or (name=='P0/crop104/key01' and bit==1))
    align.append(dict(id=name,variant=variant,bit=bit,decoded=1 if read else None,target=1,read=read,error=False if read else None,false_attribution=False,coverage_class='READ' if read else 'NO_COMPATIBLE_REFERENCE'))
 by,controls,pairs=summarize_alignment(align,[1]*32)
 assert by['NEW/crop104/key00']['variants']['E']['read']==1
 assert controls['queries']==8 and controls['reference_E_output_bits']==1 and controls['authentication_FPR'] is None
 assert len(pairs)==2*2*2*32 and all(x['bit'] in range(32) for x in pairs)
