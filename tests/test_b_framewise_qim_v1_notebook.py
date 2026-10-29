"""Affected layering/entry checks only: JSON, fake workers and inert CPU imports."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile
import nbformat
import pytest
from scripts import build_b_framewise_qim_v1_notebook as builder
from scripts import b_framewise_qim_v1_entry as entry
from scripts import b_framewise_qim_v1_setup as setup
from experiments.wan_state_clock import b_framewise_qim_v1_experiment as runner
from experiments.wan_state_clock import b_framewise_qim_v1_records as records
from experiments.wan_state_clock import b_framewise_qim_v1_report as report

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def notebook_codes():
    nb = nbformat.read(ROOT/'notebooks/b_framewise_qim_v1.ipynb', as_version=4)
    return [cell.source for cell in nb.cells if cell.cell_type == 'code']


def source_files():
    """Static test/publication inventory, never a runtime admission check."""
    pending = ['experiments/wan_state_clock/b_framewise_qim_v1_run.py', entry.CONFIG,
               'scripts/build_b_framewise_qim_v1_notebook.py', 'scripts/b_framewise_qim_v1_setup.py',
               'docs/b_framewise_qim_v1.md', *entry.CACHE_FILES]
    found = set()
    while pending:
        rel = pending.pop()
        if rel in found: continue
        path = ROOT/rel
        assert path.is_file(), path
        found.add(rel)
        for parent in path.relative_to(ROOT).parents:
            init = (parent/'__init__.py').as_posix()
            if (ROOT/init).is_file() and init not in found: pending.append(init)
        if path.suffix != '.py': continue
        package = list(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import): names = [n.name.split('.') for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = (package[:len(package)-node.level+1] if node.level else [])+(node.module.split('.') if node.module else [])
                names = [prefix]+[prefix+n.name.split('.') for n in node.names if n.name != '*']
            for parts in names:
                if not parts or parts[0] not in ('main','runtime','experiments','scripts'): continue
                for candidate in (Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py'):
                    if (ROOT/candidate).is_file() and candidate.as_posix() not in found: pending.append(candidate.as_posix())
    return sorted(found)


def copy_source(destination):
    for rel in source_files():
        path = destination/rel; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, path)
    return destination


def settings(tmp_path, inputs=True, source_directory=ROOT):
    source, output = tmp_path/'input', tmp_path/'output'
    if inputs:
        terminal = source/'private/b2-m3-e01/P0_terminal.pt'
        terminal.parent.mkdir(parents=True, exist_ok=True); terminal.write_text('fake, never deserialized')
        records.dump(source/'protocol.json', dict(config=dict(writer=dict(bits=[0]*32))))
    env = {}; exec(notebook_codes()[1], env)
    env.update(SOURCE=source, OUTPUT=output, SOURCE_DIRECTORY=source_directory)
    return env


def forbidden(*args, **kwargs): pytest.fail('forbidden network/setup/worker call')


def fake_setup(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda name:'/fake/'+name)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw:types.SimpleNamespace(returncode=0))


class FakeBackend:
    def __init__(self, cfg, output, event): self.event = event
    def capture(self, path):
        self.event('D181', False)
        raise RuntimeError('artificial failure; no latent/model read')
    def close(self): pass


def fake_worker(monkeypatch, env, launches):
    class Worker:
        def __init__(self, args, **kwargs):
            assert args[args.index('-m')+1] == 'experiments.wan_state_clock.b_framewise_qim_v1_run'
            assert Path(args[args.index('--source')+1]) == env['SOURCE']
            assert not records.has_existing_output(env['OUTPUT'])
            launches.append(args)
            result = runner.run(records.load(args[args.index('--config')+1]), env['SOURCE'], env['OUTPUT'], FakeBackend)
            assert result['status'] == 'STOPPED_FIXED_ATTEMPT'
            self.returncode = 0
        def poll(self): return self.returncode
    monkeypatch.setattr(subprocess, 'Popen', Worker)


def test_thin_structure_builder_sync_and_dependency_direction(tmp_path):
    notebook = nbformat.read(ROOT/'notebooks/b_framewise_qim_v1.ipynb', as_version=4)
    nbformat.validate(notebook)
    codes = notebook_codes()
    assert len(codes) == 5
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None and cell.outputs == []
            assert not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.walk(ast.parse(cell.source)))
    assert not hasattr(builder, 'recovery_source') and not hasattr(builder, 'setup_source')
    assert not (ROOT/'runtime/wan/b_framewise_qim_v1.py').exists()
    for folder in ('main/tube_state','runtime/wan'):
        for path in (ROOT/folder).glob('b_framewise_qim_v1*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                assert not any(name.startswith(('experiments','scripts','notebooks')) for name in names)
    for name in ('experiment','records','run'):
        text = (ROOT/f'experiments/wan_state_clock/b_framewise_qim_v1_{name}.py').read_text()
        assert 'b_framewise_qim_v1_report' not in text and 'b_framewise_qim_v1_entry' not in text
    result = builder.build(tmp_path)
    assert Path(result['notebook']).read_bytes() == (ROOT/'notebooks/b_framewise_qim_v1.ipynb').read_bytes()


@pytest.mark.parametrize('download', [False, True])
def test_first_launch_then_cached_reopen(tmp_path, monkeypatch, download):
    env = settings(tmp_path, source_directory=None if download else ROOT)
    downloads = []
    archive = tmp_path/'source.zip'
    if download:
        with zipfile.ZipFile(archive, 'w') as bundle:
            for rel in source_files(): bundle.write(ROOT/rel, 'source/'+rel)
    def retrieve(url, destination):
        downloads.append(url)
        shutil.copyfile(archive if 'codeload' in url else ROOT/entry.ENTRY, destination)
    monkeypatch.setattr(entry.urllib.request, 'urlretrieve', retrieve if download else forbidden)
    # Keep the helper's scratch outside actual /content in this local fixture.
    original_prepare = entry.prepare_source
    fake_setup(monkeypatch)
    launches = []; fake_worker(monkeypatch, env, launches)
    exec(notebook_codes()[2], env)
    env['entry'].prepare_source = lambda directory, ref, scratch:original_prepare(directory, ref, tmp_path/'scratch')
    exec(notebook_codes()[3], env); exec(notebook_codes()[4], env)
    assert len(launches) == 1 and len(downloads) == (2 if download else 0)
    session = env['session']
    assert (session.logs/'setup.log').exists() and (session.logs/'runner.log').exists()
    assert not (session.output/'setup.log').exists() and not (session.output/'runner.log').exists()
    assert all((session.cache/rel).is_file() for rel in entry.CACHE_FILES)
    monkeypatch.setattr(subprocess, 'run', forbidden); monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(entry.urllib.request, 'urlretrieve', forbidden)
    env = settings(tmp_path, source_directory=None)
    for code in notebook_codes()[2:]: exec(code, env)
    summary = env['session'].display()
    assert summary['physical_calls_this_invocation'] == 0
    assert summary['calls']['D181']['started'] == 1 and summary['calls']['D181']['returned'] == 0


@pytest.mark.parametrize('must_kill', [False, True])
def test_interrupt_and_reopen(tmp_path, monkeypatch, must_kill):
    env = settings(tmp_path); fake_setup(monkeypatch)
    monkeypatch.setattr(entry.urllib.request, 'urlretrieve', forbidden)
    events = []
    class Worker:
        returncode = None
        def __init__(self, args, **kwargs):
            assert not records.has_existing_output(env['OUTPUT'])
            records.dump(env['OUTPUT']/'attempt.json', dict(status='STARTED'))
            journal = runner.Journal(env['OUTPUT']); journal.event('D181', False)
            journal.rows['BASE/preclip']['status'] = 'STARTED'; records.dump(env['OUTPUT']/'state.json', journal.rows)
        def poll(self): return None
        def terminate(self): events.append('terminate')
        def wait(self, timeout=None):
            if must_kill and timeout is not None: raise subprocess.TimeoutExpired('fake', timeout)
            events.append('wait')
        def kill(self): events.append('kill')
    monkeypatch.setattr(subprocess, 'Popen', Worker)
    monkeypatch.setattr(entry.time, 'sleep', lambda n:(_ for _ in ()).throw(KeyboardInterrupt()))
    exec(notebook_codes()[2], env)
    with pytest.raises(KeyboardInterrupt): exec(notebook_codes()[3], env)
    assert events == (['terminate','kill','wait'] if must_kill else ['terminate','wait'])
    monkeypatch.setattr(subprocess, 'run', forbidden); monkeypatch.setattr(subprocess, 'Popen', forbidden)
    env = settings(tmp_path)
    for code in notebook_codes()[2:]: exec(code, env)
    assert env['session'].display()['state_counts']['UNRESOLVED_INTERRUPTED'] == 1


def test_missing_input_no_download_setup_or_attempt(tmp_path, monkeypatch, capsys):
    env = settings(tmp_path, inputs=False, source_directory=None)
    monkeypatch.setattr(entry.urllib.request, 'urlretrieve', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden); monkeypatch.setattr(subprocess, 'Popen', forbidden)
    for code in notebook_codes()[2:]: exec(code, env)
    # Direct helper callers have the same missing-input behavior.
    session = entry.Session(env['SOURCE'], env['OUTPUT'], 'unused')
    session.execute()
    assert not env['OUTPUT'].exists() and not session.cache.exists() and not session.logs.exists()
    output = capsys.readouterr().out
    assert 'INPUT_MISSING' in output and 'P0_terminal.pt' in output and 'protocol.json' in output


@pytest.mark.parametrize('cache_mode', ['complete','missing','report_missing'])
def test_existing_output_under_scientific_import_and_network_block(tmp_path, cache_mode):
    output = tmp_path/'out'; records.dump(output/'attempt.json', dict(status='STARTED'))
    records.dump(output/'state.json', {'BASE/preclip':dict(status='STARTED')})
    cache = output.with_name(output.name+'-entry-cache')
    if cache_mode != 'missing':
        entry.cache_entry(ROOT, cache)
        if cache_mode == 'report_missing': (cache/'experiments/wan_state_clock/b_framewise_qim_v1_report.py').unlink()
    before = {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    probe = r'''
import ast,builtins,json,pathlib,sys,urllib.request,socket,subprocess
class BlockScience:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','numpy','diffusers','transformers'):
            raise AssertionError('scientific import forbidden: '+fullname)
sys.meta_path.insert(0,BlockScience())
def forbidden(*args,**kwargs): raise AssertionError('network/process forbidden')
urllib.request.urlretrieve=forbidden;urllib.request.urlopen=forbidden
socket.socket=forbidden;subprocess.run=forbidden;subprocess.Popen=forbidden
notebook=json.loads(pathlib.Path(sys.argv[1]).read_text())
codes=[c['source'] for c in notebook['cells'] if c['cell_type']=='code']
codes=[''.join(c) if isinstance(c,list) else c for c in codes]
env=dict(Path=pathlib.Path,SOURCE=pathlib.Path('/missing-input'),OUTPUT=pathlib.Path(sys.argv[2]),SOURCE_REF='unused',SOURCE_DIRECTORY=None)
for code in codes[2:]: exec(compile(code,'thin-cell','exec'),env)
assert not any(name in sys.modules for name in ('torch','numpy','diffusers'))
print('READ_ONLY_WITHOUT_SCIENCE_OR_NETWORK_OK')
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(ROOT/'notebooks/b_framewise_qim_v1.ipynb'), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'READ_ONLY_WITHOUT_SCIENCE_OR_NETWORK_OK' in result.stdout
    assert ('详细报告代码不可用' in result.stdout) == (cache_mode != 'complete')
    assert before == {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['missing','syntax'])
def test_optional_report_does_not_block_new_worker(tmp_path, monkeypatch, damage):
    source_root = copy_source(tmp_path/'editable')
    path = source_root/'experiments/wan_state_clock/b_framewise_qim_v1_report.py'
    if damage == 'missing': path.unlink()
    else: path.write_text('invalid python !')
    env = settings(tmp_path, source_directory=source_root)
    fake_setup(monkeypatch); launches = []; fake_worker(monkeypatch, env, launches)
    session = entry.Session(env['SOURCE'], env['OUTPUT'], 'unused', source_root)
    session.execute(); session.display()
    assert len(launches) == 1 and session.report is None
    assert (env['OUTPUT']/'result.json').is_file()


def test_report_loader_uses_requested_files_not_prior_imports(tmp_path):
    a = tmp_path/'a'; b = tmp_path/'b'
    entry.cache_entry(ROOT, a); entry.cache_entry(ROOT, b)
    path = b/'experiments/wan_state_clock/b_framewise_qim_v1_records.py'
    path.write_text(path.read_text().replace('sources=1, arms=2', 'sources=7, arms=2'))
    first, second = entry.load_report(a), entry.load_report(b)
    assert first.denominators()['sources'] == 1 and second.denominators()['sources'] == 7
    assert first.denominators()['sources'] == 1


def test_no_git_cli_and_core_work_without_outer_report_entry(tmp_path):
    copy_source(tmp_path)
    for relative in (*entry.CACHE_FILES[0:1], 'scripts/b_framewise_qim_v1_setup.py',
                     'scripts/build_b_framewise_qim_v1_notebook.py',
                     'experiments/wan_state_clock/b_framewise_qim_v1_report.py'):
        (tmp_path/relative).unlink()
    probe = "import sys;sys.path.insert(0,sys.argv[1]);from main.tube_state import b_framewise_qim_v1,b_framewise_qim_v1_solver;from runtime.wan import b_framewise_qim_v1_operator;from runtime.wan.b_framewise_qim_v1_backend import Backend;from experiments.wan_state_clock.b_framewise_qim_v1_records import load;from experiments.wan_state_clock.b_framewise_qim_v1_experiment import run;b=Backend(load(sys.argv[1]+'/experiments/wan_state_clock/configs/b_framewise_qim_v1.json'),sys.argv[1],lambda *a:None);assert b.loaded is False and b.base.vae is None;assert not any('b_framewise_qim_v1_report' in n or 'b_framewise_qim_v1_entry' in n for n in sys.modules);print('CORE_RUNTIME_CLI_WITHOUT_OUTER_LAYER_OK')"
    result = subprocess.run([sys.executable, '-I', '-B', '-c', probe, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'CORE_RUNTIME_CLI_WITHOUT_OUTER_LAYER_OK' in result.stdout
    result = subprocess.run([sys.executable, '-B', '-m', 'experiments.wan_state_clock.b_framewise_qim_v1_run', '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and '--source' in result.stdout and '--output' in result.stdout
    records.dump(tmp_path/'saved/attempt.json', dict(status='STARTED'))
    result = subprocess.run([sys.executable, '-B', '-m', 'experiments.wan_state_clock.b_framewise_qim_v1_run', '--source', str(tmp_path/'missing'), '--output', str(tmp_path/'saved')], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['physical_calls_this_invocation'] == 0


@pytest.mark.parametrize('failure,expected', [
    ("ModuleNotFoundError: No module named 'numpy'", ['install','numpy']),
    ("cannot import name 'FqnToConfig' from torchao.quantization", ['uninstall','-y','torchao'])])
def test_moved_setup_actual_failure_repairs(tmp_path, monkeypatch, failure, expected):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            kwargs['stdout'].write(failure+'\n')
            return types.SimpleNamespace(returncode=1)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    setup.ensure_imports('import numpy', ['numpy'], 'fake probe', ROOT, tmp_path/'setup.log')
    assert calls[1][3:] == expected and len(calls) == 3


def test_corrupt_records_keep_unknown_and_fixed(tmp_path):
    records.dump(tmp_path/'state.json', {'BASE/preclip':dict(status='RETURNED',slots=0,kind='fake')})
    records.dump(tmp_path/'result.json', [])
    snapshot = records.recover(tmp_path); summary = report.recovery_summary(snapshot)
    assert snapshot['rows']['BASE/preclip']['slots'] == 114030
    assert snapshot['rows']['BASE/preclip']['kind'] == 'time_read'
    assert summary['status'] == 'UNRESOLVED_RECORDS' and 'result' in snapshot['read_errors']
    assert all(row['started'] is None for row in summary['calls'].values())
    assert all(row['bit_errors'] is None for row in summary['time_layers'])
    assert all(row['observed_bits'] is None for row in summary['identity'])

def completed_saved_snapshot():
    states=records.plan()
    for row in states.values():row['status']='RETURNED'
    time_rows=[dict(key=f'{arm}/{layer}',status='RETURNED',fixed_slots=114030,active_slots=112770,
        observed_active_slots=112770,bit_errors=0,erased_active_slots=0,wrong_origin_candidate_count=0)
        for arm in records.ARMS for layer in records.LAYERS]
    identity=[]
    for arm in records.ARMS:
        for start in (0,40):
            for phase in range(4):
                for region in ('R0','R1'):
                    for local_row in range(1,11):
                        for bit in range(8):
                            identity.append(dict(arm=arm,start=start,phase=phase,region=region,local_row=local_row,bit=bit,
                                status='AVAILABLE',value=0,common_complete_cell_eligible=phase==0,
                                group=0 if phase==0 else None,correct=True if phase==0 else None))
    return dict(rows=states,saved=dict(result=dict(status='COMPLETED_FIXED_RECORDS',evaluation_status='POST_RECORDED'),
        counts={k:dict(started=n,returned=n) for k,n in records.BUDGETS.items()},
        evaluation=dict(time=time_rows,identity=identity,identity_comparison=dict(fixed_paired_slots=1280,
            common_complete_observed_slots=320,base_common_errors=0,qim_common_errors=0,
            base_common_ber=0.,qim_common_ber=0.,added_support=0,lost_support=0))))


@pytest.mark.parametrize('damage',['identity_missing_fields','identity_duplicate','negative_time_errors','malformed_time_status','comparison_missing'])
def test_nested_corrupt_post_records_do_not_fabricate_completion_or_zero_errors(damage):
    snapshot=completed_saved_snapshot()
    assert report.recovery_summary(snapshot)['status']=='COMPLETED_FIXED_RECORDS'
    evaluation=snapshot['saved']['evaluation']
    if damage=='identity_missing_fields':evaluation['identity']=[dict(arm=a) for a in records.ARMS for _ in range(1280)]
    elif damage=='identity_duplicate':evaluation['identity'][1]=dict(evaluation['identity'][0])
    elif damage=='negative_time_errors':evaluation['time'][0]['bit_errors']=-7
    elif damage=='malformed_time_status':evaluation['time'][0]['status']=[]
    else:evaluation['identity_comparison']={}
    summary=report.recovery_summary(snapshot)
    assert summary==report.recovery_summary(snapshot)
    assert summary['status']=='UNRESOLVED_POST_RECORD'
    if damage.startswith('identity'):
        assert summary['identity'][0]['full_cell_errors'] is None
        assert summary['identity'][0]['observed_bits'] is None
    elif damage.startswith('comparison'):assert summary['identity_comparison'] is None
    else:assert summary['time_layers'][0]['bit_errors'] is None
    assert summary['denominators']['identity_local_bit_slots']==2560
