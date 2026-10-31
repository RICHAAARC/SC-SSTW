"""Offline artificial source/bootstrap/setup checks; no network, install, codec or model."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import shutil
import time
import urllib.request
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.quick
STEM = 'b_conditional_time_output_v1'


def module(relative):
    spec=importlib.util.spec_from_file_location('fixture_'+Path(relative).stem,ROOT/relative)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def scientific_source_files():
    # Actual runtime dependency closure for this CPU entry, copied as ordinary files.
    return [*('main/tube_state/'+s+'.py' for s in (STEM,'b_minimal_l_v1','b_soft_time_evidence_v1','b_framewise_qim_time_recovery_v1')),
            *('runtime/wan/'+s+'.py' for s in (STEM,'b_minimal_l_v1','b_framewise_qim_time_recovery_v1')),
            *('experiments/wan_state_clock/'+STEM+s+'.py' for s in ('_records','_experiment','_evaluation','_run')),
            'experiments/wan_state_clock/b_minimal_l_v1_records.py',
            *('scripts/'+s+'.py' for s in (STEM+'_entry',STEM+'_setup','b_framewise_qim_v1_entry','b_framewise_qim_v1_setup'))]


def ordinary_archive(path):
    files=scientific_source_files()+[p+'/__init__.py' for p in ('main','main/tube_state','runtime','runtime/wan','experiments','experiments/wan_state_clock')]
    with zipfile.ZipFile(path,'w') as archive:
        for relative in files: archive.write(ROOT/relative,'ordinary-source/'+relative)


def artificial_inputs(tmp_path):
    helpers=module('tests/test_b_conditional_time_output_v1.py')
    config=helpers.make_inputs(tmp_path,'semantic')
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    return helpers,config,config_path


def test_consumed_or_conflicting_output_before_config_source_setup(tmp_path,monkeypatch):
    entry=module('scripts/'+STEM+'_entry.py')
    monkeypatch.setattr(entry,'_common',lambda:pytest.fail('source/setup touched on recovery'))
    output=tmp_path/'used';output.mkdir();(output/'attempt.json').write_text('{}')
    result=entry.launch('/missing','/missing',output)
    assert result['status']=='READ_ONLY_EXISTING_ATTEMPT' and result['physical_calls_this_invocation']==0
    conflict=tmp_path/'conflict';conflict.mkdir();(conflict/'existing.txt').write_text('retain')
    assert entry.launch(None,None,conflict)['status']=='OUTPUT_CONFLICT'
    assert (conflict/'existing.txt').read_text()=='retain'


def test_unconfigured_does_not_download_or_reserve(tmp_path,monkeypatch):
    entry=module('scripts/'+STEM+'_entry.py');common=entry._common()
    monkeypatch.setattr(entry,'_common',lambda:common)
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('downloaded unconfigured input'))
    path=tmp_path/'config.json';path.write_text(json.dumps({'entry':None,'output':str(tmp_path/'out')}))
    assert entry.launch(None,path,scratch=tmp_path/'source')['status']=='UNCONFIGURED'
    assert not (tmp_path/'out').exists()


def test_remote_source_setup_and_actual_artificial_cli(tmp_path,monkeypatch):
    _,config,path=artificial_inputs(tmp_path)
    archive=tmp_path/'source.zip';ordinary_archive(archive)
    downloads=[]
    def retrieve(url,destination):
        downloads.append(url);shutil.copyfile(archive,destination);return str(destination),None
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    entry=module('scripts/'+STEM+'_entry.py');common=entry._common()
    monkeypatch.setattr(entry,'_common',lambda:common)
    native_sleep=time.sleep
    monkeypatch.setattr(common.time,'sleep',lambda seconds:native_sleep(.01))
    result=entry.launch(None,path,config['output'],scratch=tmp_path/'source-cache')
    assert result['evaluation_status']=='FINITE_ADDED_SUPPORT'  # Artificial JSON only.
    assert len(downloads)==1 and downloads[0].startswith('https://codeload.github.com/RICHAAARC/SC-SSTW/')
    output=Path(config['output']);counts=json.loads((output/'counts.json').read_text())
    assert counts['semantic_submit']['started']==5 and counts['codebook']['started']==0
    log=(output.with_name(output.name+'-logs')/'setup.log').read_text()
    assert 'import numpy,psutil' in log and 'WanPipeline' not in log and 'apt-get' not in log
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('recovery downloaded'))
    again=entry.launch('/missing','/missing',output)
    assert again['physical_calls_this_invocation']==0
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_unsafe_download_fails_before_attempt(tmp_path,monkeypatch):
    _,config,path=artificial_inputs(tmp_path)
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escaped.py','unsafe')
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda url,destination:shutil.copyfile(archive,destination))
    entry=module('scripts/'+STEM+'_entry.py')
    result=entry.launch(None,path,config['output'],scratch=tmp_path/'source-cache')
    assert result['status']=='PREPARATION_FAILED' and 'unsafe source archive path' in result['error']
    assert not (Path(config['output'])/'attempt.json').exists() and not (tmp_path/'escaped.py').exists()


def test_preparation_cache_cannot_overlap_inputs(tmp_path,monkeypatch):
    _,config,path=artificial_inputs(tmp_path)
    new=tmp_path/'out-entry-cache';(tmp_path/'inputs').rename(new)
    config['views']={view:str(new/Path(p).name) for view,p in config['views'].items()}
    config['truth']=str(new/'truth.json');path.write_text(json.dumps(config))
    entry=module('scripts/'+STEM+'_entry.py');common=entry._common()
    monkeypatch.setattr(entry,'_common',lambda:common)
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('source prepared despite overlap'))
    result=entry.launch(None,path,config['output'],scratch=tmp_path/'source-cache')
    assert result['status']=='PREPARATION_FAILED' and 'overlaps read-only input' in result['error']


def test_setup_failure_is_concrete_and_no_attempt(tmp_path,monkeypatch):
    _,config,path=artificial_inputs(tmp_path)
    entry=module('scripts/'+STEM+'_entry.py');common=entry._common()
    monkeypatch.setattr(entry,'_common',lambda:common)
    def fail(*args):raise ImportError('artificial actual import failure')
    monkeypatch.setattr(common,'load_file',lambda *args:SimpleNamespace(setup_dependencies=fail))
    result=entry.launch(ROOT,path,config['output'],scratch=tmp_path/'source-cache')
    assert result['status']=='PREPARATION_FAILED' and 'actual import failure' in result['error']
    assert not (Path(config['output'])/'attempt.json').exists()


def test_worker_error_after_reservation_is_not_zero_call_preparation_failure(tmp_path,monkeypatch):
    _,config,path=artificial_inputs(tmp_path)
    entry=module('scripts/'+STEM+'_entry.py');common=entry._common()
    monkeypatch.setattr(entry,'_common',lambda:common)
    monkeypatch.setattr(common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *args:None))
    def fail(session,root):
        session.output.mkdir();(session.output/'attempt.json').write_text('{}')
        raise OSError('artificial worker error after consumption')
    monkeypatch.setattr(common.Session,'_worker',fail)
    result=entry.launch(ROOT,path,config['output'],scratch=tmp_path/'source-cache')
    assert result['status']=='RUNNER_RESULT_UNRESOLVED' and result['attempt_consumed'] is True
    assert 'physical_calls_this_invocation' not in result
    assert entry.launch(None,None,config['output'])['physical_calls_this_invocation']==0


@pytest.mark.parametrize('kind',['semantic','raw','media'])
def test_setup_only_selected_dependencies_no_gpu_or_version_gate(tmp_path,monkeypatch,kind):
    setup=module('scripts/'+STEM+'_setup.py');calls=[]
    fake=SimpleNamespace(ensure_imports=lambda *a:calls.append(('imports',a)),command=lambda *a:calls.append(('command',a)))
    monkeypatch.setattr(setup.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)))
    monkeypatch.setattr(setup.importlib.util,'module_from_spec',lambda spec:fake)
    binaries=iter([None,'/usr/bin/ffmpeg','/usr/bin/ffprobe'])
    monkeypatch.setattr(setup.shutil,'which',lambda name:next(binaries) if kind=='media' else pytest.fail('nonmedia needs codec'))
    setup.setup_dependencies(ROOT,tmp_path,kind)
    assert calls[0][1][0]=='import numpy,psutil' and calls[0][1][1]==['numpy','psutil']
    commands=[row[1][0] for row in calls if row[0]=='command']
    assert sum(c[0]=='apt-get' for c in commands)==(2 if kind=='media' else 0)
    assert all('WanPipeline' not in str(c) and '==' not in str(c) for c in commands)


def test_notebook_remote_bootstrap_and_saved_branch_without_colab(tmp_path,monkeypatch):
    import nbformat
    builder=module('scripts/build_'+STEM+'_notebook.py')
    built=builder.build(tmp_path);nb=nbformat.read(built['notebook'],as_version=4);nbformat.validate(nb)
    assert nb==nbformat.read(ROOT/'notebooks'/(STEM+'.ipynb'),as_version=4)
    assert nb.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    calls=[]
    def retrieve(url,path):
        calls.append(url)
        if str(path).endswith(STEM+'_entry.py'):
            Path(path).write_text("def launch(source,config,output,ref):\n return {'status':'ARTIFICIAL_LAUNCH','source':source,'config':config,'ref':ref}\n")
        else:
            relative=url.split('/test-ref/')[1];shutil.copyfile(ROOT/relative,path)
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    namespace=dict(SOURCE_DIRECTORY=None,SOURCE_REF='test-ref',CONFIG_PATH=str(tmp_path/'config.json'),OUTPUT=str(tmp_path/'out'))
    exec(builder.RUN,namespace)
    assert namespace['result']['status']=='ARTIFICIAL_LAUNCH' and namespace['result']['source'] is None
    assert namespace['result']['ref']=='test-ref' and len(calls)==4
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('saved/unconfigured branch downloaded'))
    output=tmp_path/'used';output.mkdir();(output/'attempt.json').write_text('{}')
    namespace.update(OUTPUT=str(output),CONFIG_PATH=None,SOURCE_DIRECTORY='/missing')
    exec(builder.RUN,namespace)
    assert namespace['result']['status']=='READ_ONLY_EXISTING_ATTEMPT'
    namespace.update(OUTPUT=str(tmp_path/'empty'),CONFIG_PATH=None,SOURCE_DIRECTORY=None)
    exec(builder.RUN,namespace)
    assert namespace['result']['status']=='UNCONFIGURED' and not (tmp_path/'empty').exists()
