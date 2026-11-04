"""Stdlib artificial scheduler checks; no model, codec, receiver or real data."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import contextlib
import importlib.util
import io
import json
import subprocess
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STEM='b_joint_time_packet_a_physical_v1'


def module(suffix):
    path=ROOT/'scripts'/(STEM+'_'+suffix+'.py')
    spec=importlib.util.spec_from_file_location('test_joint_'+suffix,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def stage(folder,terminal=True,errors=None,results=None):
    dump(folder/'attempt.json',{'consumed':True})
    if terminal:
        value=dict(status='FIXED_STAGE_TERMINATED',stage_terminal=True,errors=errors or {},results=results or {})
        dump(folder/'blind_terminal.json',value);dump(folder/'result.json',value)


class RunAllTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'output';self.entry=module('entry')
        self.packages=[]
        media=dict(layer_raw={'n01/OLD/layer/clip':'layer_raw/a.npz'},
                   time_raw={'n01/OLD/FULL':'time_raw/a.npz'},e02_old_media={'FULL':'e02_old_media/a.mp4'})
        post=dict(old_time={'n01/FULL':{'H':'old_time/H.json'}},writer_protocol={'n01':'/drive/original/protocol.json'})
        for i,files in enumerate(({'media_inputs.json':json.dumps(media),'layer_raw/a.npz':'artificial','e02_old_media/a.mp4':'artificial'},
                                  {'post_inputs.json':json.dumps(post),'time_raw/a.npz':'artificial','old_time/H.json':'NOT_FOR_BLIND'})):
            p=self.root/f'part{i}.zip'
            with zipfile.ZipFile(p,'w') as archive:
                for name,value in files.items():archive.writestr(name,value)
            self.packages.append(str(p))
        self.handoff=self.root/'cpu';dump(self.handoff/'cpu_handoff.json',{'stage_terminal':True})

    def launch(self,worker):
        original=self.entry.load_module
        def loader(path,name):
            if str(path).endswith('_setup.py'):return SimpleNamespace(setup_dependencies=lambda *a,**kw:None)
            return original(path,name)
        with patch.object(self.entry,'load_module',loader),patch.object(self.entry,'run_worker',worker):
            return self.entry.launch(self.handoff,self.packages,self.out,ROOT)

    def worker(self,calls,fail=None,partial=None):
        def run(command,root,log,receipt,folder):
            name=folder.name;calls.append(name)
            cfg=json.loads(Path(command[command.index('--config')+1]).read_text())
            if name=='cpu_media':
                self.assertTrue(self.entry.terminal(self.out/'gpu1'))
                self.assertNotIn('old_time',cfg);self.assertNotIn('writer_protocol',cfg)
                raw=Path(next(iter(cfg['time_raw'].values())))
                self.assertTrue(raw.is_file())
                self.assertFalse((raw.parent.parent/'post_inputs.json').exists())
                self.assertFalse((raw.parent.parent/'old_time').exists())
            if name=='post':
                self.assertTrue(all(self.entry.terminal(self.out/s) for s in ('cpu_media','gpu2')))
                self.assertEqual(Path(cfg['old_time']['n01/FULL']['H']).read_text(),'NOT_FOR_BLIND')
            stage(folder,terminal=name!=fail,errors={'artificial':'retained'} if name==partial else None)
            return 1 if name==fail else 0
        return run

    def test_current_gpu1_resume_and_consumed_reentry(self):
        stage(self.out/'gpu1');before=(self.out/'gpu1/attempt.json').read_bytes();calls=[]
        self.launch(self.worker(calls));self.assertEqual(calls,['cpu_media','gpu2','post'])
        self.assertEqual((self.out/'gpu1/attempt.json').read_bytes(),before)
        self.packages=['missing'];self.handoff=self.root/'missing'
        self.launch(lambda *a: self.fail('consumed stage executed'))
        self.assertTrue((self.out/'run_all_summary.md').is_file())

    def test_fresh_run_sequences_all_four(self):
        calls=[];self.launch(self.worker(calls));self.assertEqual(calls,list(self.entry.STAGES))

    def test_failed_consumed_stage_not_retried_and_no_false_terminal(self):
        stage(self.out/'gpu1');calls=[]
        first=self.launch(self.worker(calls,fail='cpu_media'))
        self.assertEqual(calls,['cpu_media']);self.assertEqual(first['stages']['post']['status'],'WAITING_INPUT')
        self.launch(lambda *a:self.fail('failed consumed stage retried'))
        self.assertFalse((self.out/'gpu2/attempt.json').exists());self.assertFalse(self.entry.terminal(self.out/'cpu_media'))

    def test_partial_terminal_continues_fixed_downstream(self):
        stage(self.out/'gpu1',errors={'n01':'artificial missing'});calls=[]
        self.launch(self.worker(calls,partial='cpu_media'));self.assertEqual(calls,['cpu_media','gpu2','post'])

    def test_both_blind_terminals_required_before_post_config(self):
        stage(self.out/'gpu1');stage(self.out/'cpu_media',terminal=False);stage(self.out/'gpu2')
        self.packages=['DO_NOT_OPEN'];calls=[]
        result=self.launch(self.worker(calls));self.assertEqual(calls,[])
        self.assertEqual(result['stages']['post']['dependencies'],['cpu_media'])

    def test_bootstrap_consumed_without_source_or_inputs(self):
        for name in self.entry.STAGES:stage(self.out/name)
        boot=module('bootstrap')
        with patch.object(boot.urllib.request,'urlretrieve',side_effect=AssertionError('network')):
            value=boot.launch('missing',['missing'],self.out,source_directory='missing')
        self.assertEqual(set(value['stages']),set(self.entry.STAGES))

    def test_zip_paths_and_selected_post_extraction(self):
        path=self.root/'unsafe.zip'
        with zipfile.ZipFile(path,'w') as archive:archive.writestr('../escape','bad')
        with self.assertRaises(ValueError):self.entry.extract(path,self.root/'unpacked')
        self.assertFalse((self.root/'escape').exists())

    def test_real_owned_subprocess_live_console_and_receipt(self):
        log=self.root/'console.log';receipt=self.root/'worker.json';stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code=self.entry.run_worker([sys.executable,'-u','-c',"print('artificial progress',flush=True)"],
                ROOT,log,receipt,self.out/'cpu_media')
        self.assertEqual(code,0);self.assertIn('artificial progress',stdout.getvalue())
        self.assertIn('artificial progress',log.read_text());value=json.loads(receipt.read_text())
        self.assertTrue(value['reaped']);self.assertTrue(value['console_drained']);self.assertFalse(value['worker_restarted'])

    def test_interrupt_term_kill_reap_without_retry(self):
        class Child:
            pid=123;stdout=None
            def __init__(self):self.waits=0;self.terminated=False;self.killed=False
            def wait(self,timeout=None):
                self.waits+=1
                if self.waits==1:raise KeyboardInterrupt()
                if timeout==30:raise subprocess.TimeoutExpired('fake',30)
                return -9
            def poll(self):return None
            def terminate(self):self.terminated=True
            def kill(self):self.killed=True
        child=Child();receipt=self.root/'worker.json'
        with self.assertRaises(KeyboardInterrupt):
            self.entry.run_worker([],ROOT,self.root/'console.log',receipt,self.out/'gpu2',process_factory=lambda *a,**k:child)
        value=json.loads(receipt.read_text());self.assertTrue(child.terminated and child.killed)
        self.assertTrue(value['reaped']);self.assertFalse(value['worker_restarted'])

    def test_report_includes_time_identity_errors_and_claim_ceiling(self):
        report=module('report');stages={}
        time=dict(units={'n01/NEW/FULL':{'counts':{'local/CORRECT':3,'local/WRONG':2,'local/UNKNOWN':176,'edge/UNKNOWN':180}}},
                  preservation={'n01':{'old_correct':5,'old_wrong':2,'baseline_unavailable':174,'corrected_old_wrong':1,'damaged_old_correct':2,'lost_old_correct':3,'new_local_unknown':176}})
        identity=dict(comparisons={'n01/FULL':dict(common_readable=20,common_comparable=10,OLD_common_errors=1,OLD_common_BER=.1,
            NEW_common_errors=2,NEW_common_BER=.2,added_readable=3,lost_readable=4,common_geometry_unresolved=10)})
        stage(self.out/'post',errors={'artificial':'retained'},results=dict(time=time,identity=identity))
        stages['post']=self.entry.saved(self.out/'post');report.build(self.out,stages)
        text=(self.out/'run_all_summary.md').read_text()
        for required in ('n01/NEW/FULL','n01/FULL','0.1','0.2','4932','38400','scientific_pass=null','retained'):
            self.assertIn(required,text)

    def test_delivered_notebook_matches_builder_and_calls_new_wiring(self):
        path=ROOT/'scripts'/('build_'+STEM+'_notebook.py')
        spec=importlib.util.spec_from_file_location('joint_builder',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        builder.build(self.root)
        self.assertEqual((ROOT/'notebooks'/(STEM+'.ipynb')).read_bytes(),(self.root/'notebooks'/(STEM+'.ipynb')).read_bytes())
        book=json.loads((self.root/'notebooks'/(STEM+'.ipynb')).read_text())
        self.assertEqual(len(book['cells']),4)
        self.assertEqual(''.join(book['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        run=''.join(book['cells'][3]['source']);ast.parse(run)
        self.assertIn('INPUT_PACKAGES',run);self.assertIn("'cpu_media', 'gpu2', 'post'",run)
        self.assertIn('display(Markdown',run);self.assertNotIn('/home/',builder.SETTINGS)

    def test_actual_notebook_bootstrap_entry_workers_and_metadata_only_repeat(self):
        source=self.root/'source';(source/'scripts').mkdir(parents=True)
        for suffix in ('bootstrap','entry','report'):
            shutil.copyfile(ROOT/'scripts'/(STEM+'_'+suffix+'.py'),source/'scripts'/(STEM+'_'+suffix+'.py'))
        (source/'scripts'/(STEM+'_setup.py')).write_text('def setup_dependencies(*args,**kwargs): pass\n')
        cli=source/'experiments/wan_state_clock'/(STEM+'_run.py');cli.parent.mkdir(parents=True)
        cli.write_text("""from pathlib import Path
import json,sys
stage=sys.argv[1];out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True)
cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
if stage=='post':
    assert all(json.loads(Path(cfg[k],'blind_terminal.json').read_text())['stage_terminal'] for k in ('cpu_media','gpu2'))
value=dict(status='ARTIFICIAL_TERMINAL',stage_terminal=True,errors={},results={})
for name in ('attempt','blind_terminal','result'):(out/(name+'.json')).write_text(json.dumps(value))
print('ARTIFICIAL_PROGRESS '+stage,flush=True)
""")
        book=json.loads((ROOT/'notebooks'/(STEM+'.ipynb')).read_text());run=''.join(book['cells'][3]['source'])
        rendered=[];stdout=io.StringIO()
        env=dict(CPU_HANDOFF=str(self.handoff),INPUT_PACKAGES=self.packages,OUTPUT=str(self.out),
                 SOURCE_DIRECTORY=str(source),SOURCE_REF='unused',MODEL_ID='artificial')
        ipython=SimpleNamespace(Markdown=lambda value:value,display=rendered.append)
        with patch.dict(sys.modules,{'IPython':SimpleNamespace(),'IPython.display':ipython}),contextlib.redirect_stdout(stdout):
            exec(compile(run,'delivered_notebook_cell','exec'),env)
            env['SOURCE_DIRECTORY']='missing';env['CPU_HANDOFF']='missing';env['INPUT_PACKAGES']=['missing']
            with patch('urllib.request.urlretrieve',side_effect=AssertionError('consumed notebook used network')):
                exec(compile(run,'delivered_notebook_cell','exec'),env)
        for name in self.entry.STAGES:self.assertTrue((self.out/name/'attempt.json').is_file())
        self.assertEqual(stdout.getvalue().count('ARTIFICIAL_PROGRESS'),4)
        self.assertEqual(len(rendered),2);self.assertIn('scientific_pass=null',rendered[-1])

    def test_stale_summary_after_consumed_post_interruption_reads_current_metadata_offline(self):
        for name in self.entry.STAGES:stage(self.out/name,terminal=name!='post')
        book=json.loads((ROOT/'notebooks'/(STEM+'.ipynb')).read_text());run=''.join(book['cells'][3]['source'])
        helper=self.out/'preparation'/(STEM+'_report.py');helper.parent.mkdir(parents=True)
        for helper_available in (False,True):
            with self.subTest(helper_available=helper_available):
                if helper_available:shutil.copyfile(ROOT/'scripts'/(STEM+'_report.py'),helper)
                (self.out/'run_all_summary.md').write_text('STALE_POST_WAITING_INPUT')
                rendered=[];stdout=io.StringIO()
                env=dict(CPU_HANDOFF='missing',INPUT_PACKAGES=['missing'],OUTPUT=str(self.out),
                         SOURCE_DIRECTORY=None,SOURCE_REF='unavailable',MODEL_ID='unused')
                ipython=SimpleNamespace(Markdown=lambda value:value,display=rendered.append)
                with patch.dict(sys.modules,{'IPython':SimpleNamespace(),'IPython.display':ipython}),\
                     patch('urllib.request.urlretrieve',side_effect=AssertionError('offline reentry network')),\
                     patch('subprocess.Popen',side_effect=AssertionError('consumed stage executed')),\
                     contextlib.redirect_stdout(stdout):
                    exec(compile(run,'delivered_notebook_cell','exec'),env)
                displayed=stdout.getvalue()+'\n'.join(rendered)
                self.assertNotIn('STALE_POST_WAITING_INPUT',displayed)
                self.assertIn('CONSUMED_WITHOUT_TERMINAL',displayed)
                self.assertIn('blind_terminal.json',displayed);self.assertIn('result.json',displayed)
                current=env['result']['stages']['post']
                self.assertIn('attempt',current['saved']);self.assertIn('result',current['read_errors'])
                self.assertFalse((self.out/'post/result.json').exists())
                if helper_available:
                    self.assertFalse(env['result']['summary']['stages']['post']['terminal'])
                    self.assertNotIn('STALE_POST_WAITING_INPUT',(self.out/'run_all_summary.md').read_text())


if __name__=='__main__':unittest.main()
