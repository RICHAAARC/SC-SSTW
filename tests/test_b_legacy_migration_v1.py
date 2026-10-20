"""Card77 fixed synthetic/dispatch/package CPU evidence, no models or media."""
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast,inspect,json,subprocess,sys,zipfile
import pytest
import torch
import nbformat
from main.tube_state import b_legacy_migration_v1 as method
from main.tube_state.legacy_reference_v1.codebook import CodebookConfig,build_tubelet_codebook
from main.tube_state.legacy_reference_v1.tubelet_partition import build_tubelet_partition_config,build_tubelet_descriptors,dot_tubelet_direction
from main.tube_state.legacy_reference_v1.tensor_artifact import FloatTensorArtifact,write_float_tensor_npy
from main.tube_state.legacy_reference_v1.embedding import projection_margin
from main.tube_state.legacy_reference_v1.reference import extract_public_reference,PublicReferenceExtractor
from main.tube_state.legacy_reference_v1.synchronization import build_offset_scale_search_result
from main.tube_state.legacy_reference_v1.fusion import sync_rescue_fusion
from runtime.wan import b_legacy_migration_v1 as runtime

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def old_config():
    return dict(tubelet_length=4,tubelet_partition=dict(spatial_patch_size=[4,4]),
        embedding_margin=1.,score_calibration=dict(embedding_projection_support_weight=.09),
        sync_search=dict(offset_search_min=-8,offset_search_max=8,search_score_rule='penalized_prior',
            min_sync_alignment_coverage_ratio=.125,min_sync_alignment_matched_count=64,
            min_payload_rescue_gain=.01,min_aligned_payload_score=.095),lambda_sync=.01)


def test_archived_margin_position_signature_and_offset_port(tmp_path):
    shape=(32,4,16,16);partition=build_tubelet_partition_config(tubelet_length=4,spatial_patch_size=(4,4))
    original=FloatTensorArtifact(shape=shape,values=array('f',[0.])*(32*4*16*16))
    marked,receipt=projection_margin(original,'pre_registered_fixture',partition,embedding_margin=1.,enable_sync=False)
    descriptors=build_tubelet_descriptors(shape,partition)
    book=build_tubelet_codebook('pre_registered_fixture',descriptors,len(descriptors[0].flat_indices),CodebookConfig(),False)
    other=build_tubelet_codebook('different_registered_fixture',descriptors,len(descriptors[0].flat_indices),CodebookConfig(),False)
    assert book.directions!=other.directions
    wrong=build_tubelet_codebook('pre_registered_fixture',descriptors,len(descriptors[0].flat_indices),replace(CodebookConfig(),direction_key='fixed_wrong_key'),False)
    assert book.directions!=wrong.directions
    assert book.directions[0]!=book.directions[16]
    assert all(book.payload_codes[d.tubelet_index]*dot_tubelet_direction(marked,d,book.directions[d.tubelet_index])>=1-1e-6 for d in descriptors)
    assert receipt['current_B_net32_writer_called'] is False
    path=tmp_path/'fixed_cropped.npy'
    write_float_tensor_npy(path,(28,4,16,16),marked.values[4*4*16*16:])
    value=extract_public_reference(path,'pre_registered_fixture',shape,old_config())
    assert value['trace']['sync_estimated_offset']==-4
    assert value['trace']['sync_estimated_scale']==1
    assert value['trace']['sync_alignment_error'] is None
    assert value['writer_support_enabled'] is False and value['attack_metadata_used'] is False
    assert abs(value['trace']['S_payload_aligned']-value['trace']['sync_candidate_score_penalized'])<=1e-6
    assert 'mean_projection_before' not in value['trace'] and 'mean_projection_after' not in value['trace']
    assert list(inspect.signature(extract_public_reference).parameters)==['received_latent_path','reference_id','reference_shape','public_method_config','codebook_config']


def test_archived_truth_is_post_only_and_public_scale_grid():
    scores={(-4,1.):.8,(4,1.):.1,(-4,.5):.2}
    a=build_offset_scale_search_result(scores,ground_truth_offset=-4,ground_truth_scale=1.)
    b=build_offset_scale_search_result(scores,ground_truth_offset=4,ground_truth_scale=.5)
    assert a['sync_estimated_offset']==b['sync_estimated_offset']==-4
    assert a['sync_estimated_scale']==b['sync_estimated_scale']==1.
    assert a['sync_alignment_error']!=b['sync_alignment_error']
    extractor=PublicReferenceExtractor('tubelet_sync',old_config(),{'tubelet':True,'sync':True},'sync_rescue_fusion')
    extractor._method_config['sync_search'].update(enable_scale_search=True,scale_candidates=[.5,1.,2.])
    sample=SimpleNamespace(applied_attack_params={'speed_ratio':2.},latent_shape=(32,4,16,16),mechanism_trace={})
    assert extractor._resolve_scale_candidates(sample)==[.5,1.,2.]
    assert extractor._resolve_sync_search_score_rule(sample)=='penalized_prior'
    assert sync_rescue_fusion({'S_tubelet':.1},.2,.01,True)==.3
    assert sync_rescue_fusion({'S_tubelet':.1},.2,.01,False)==.1


@pytest.fixture(scope='module')
def current_books():
    return {key:method.identity.codebook(secret) for key,secret in zip(method.KEY_IDS,method.identity.KEYS)}


def install_fixture(monkeypatch,tmp_path,books,failure=None):
    source=tmp_path/'source';output=tmp_path/'output';history=tmp_path/'history.json'
    history.write_text('{"records":21000}')
    local_z=torch.zeros(()).expand(1,16,3,40,64)
    for arm_index,arm in enumerate(method.ARMS):
        for target,index in enumerate([arm_index*6,arm_index*6+3]):
            for key in method.KEY_IDS:
                p=source/'blind/local'/f'{index:02d}'/(key+'.json');p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(method.identity.observe(local_z,books[key],local=True)))
    (source/'result.json').write_text('{"model":{"dtype":"float32","synthetic":true}}')
    for pair in ['P0_to_S_base','P0_to_S_ID','S_base_to_S_ID']:
        for layer in method.LAYERS:
            p=source/'quality'/pair/(layer+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"status":"FIXTURE_REFERENCE_ONLY"}')
    pixels=torch.zeros((),dtype=torch.uint8).expand(method.identity.VIDEO_SHAPE)
    full_z=torch.zeros(()).expand(method.identity.LATENT_SHAPE)
    events=[]
    def enter():
        assert (output/'attempt.json').exists() and not (output/'numeric_completion.json').exists()
        ledger=json.loads((output/'ledger.json').read_text());active=[k for k,v in ledger.items() if v['status']=='STARTED']
        assert len(active)==1;events.append(active[0])
        if active[0]==failure:raise RuntimeError('fixed CPU dispatch fault')
        if failure=='interrupt' and active[0]=='full_E/P0/rgb8':raise KeyboardInterrupt('fixed CPU interruption')
    class Backend:
        device='cpu'
        blocks={'encoder':0,'decoder':0}
        def load_pixels(self,path):enter();assert path.name in ['rgb8.pt','readback.pt'];return pixels
        def load(self):enter();return {'synthetic':True,'dtype':'float32'}
        def encode_full(self,value):enter();self.blocks['encoder']+=46;return full_z
        def encode_local(self,value):enter();assert value.shape==(9,320,512,3);self.blocks['encoder']+=3;return local_z
    original=runtime.evaluate
    def post(values):
        assert (output/'numeric_completion.json').exists()
        assert len(list((output/'blind').glob('row*.json')))==48
        return original(values)
    monkeypatch.setattr(runtime,'evaluate',post)
    return source,output,history,Backend(),events


@pytest.mark.parametrize('failure',[None,'full_E/S_base/mp4','model','interrupt'])
def test_fixed12E_reuse_failure_independence_and_no_reentry(monkeypatch,tmp_path,current_books,failure):
    source,output,history,backend,events=install_fixture(monkeypatch,tmp_path,current_books,failure)
    result=runtime.run(source,output,history,backend=backend)
    assert result['status']==('EXECUTION_COMPLETE' if failure is None else 'INCOMPLETE')
    evaluated=json.loads((output/'evaluation.json').read_text())
    assert len(evaluated['rows'])==2208 and sum(r['geometric_role']=='POSITIVE' for r in evaluated['rows'])==8
    assert len(list((output/'reused').glob('*.json')))==12
    assert all(not r['phase_certified'] for r in evaluated['rows'])
    assert evaluated['payload_BER']=='UNMEASURED_FFT0' and evaluated['scientific_pass'] is None
    for role in method.plan():
        if role['layer']=='mp4' and role['history']=='local':
            reuse=json.loads((output/'reused'/(role['row']+'.json')).read_text())
            assert not reuse['new_physical_observation']
            assert json.loads((output/'blind'/(role['row']+'.json')).read_text())==reuse['original']['vectors'][0]
    if failure is None:
        assert result['counts']['posterior_encode']['returned']==12
        assert backend.blocks['encoder']==294 and backend.blocks['decoder']==0
        assert len(result['calls'])==72 and all(v['status']=='RETURNED' for v in result['calls'].values())
    if failure=='full_E/S_base/mp4':
        failed=[r for r in evaluated['vectors'] if r['arm']=='S_base' and r['layer']=='mp4' and r['history']=='full']
        assert all(r['status']=='UPSTREAM_FAILED' for r in failed)
        assert result['calls']['full_E/S_ID/mp4']['status']=='RETURNED'
        assert result['counts']['posterior_encode']['returned']==11
    prior=list(events);files={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    repeat=runtime.run(source,output,history,backend=backend)
    assert repeat['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED' and events==prior
    assert files=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_portable_notebook_text_and_all_cli_entrypoints(tmp_path):
    from scripts.build_b_legacy_migration_v1_notebook import build,SOURCE_FILES,ROOT
    notebook,archive=build()
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'SOURCE_TEXT=' in codes[1]
    source_tree=ast.parse(codes[1])
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('urlretrieve','upload') for n in ast.walk(source_tree))
    assert 'b_legacy_migration_v1_run' in codes[2]
    assert 'ensure_media_tools' not in codes[2] and 'apt-get' not in codes[2]
    for code in codes:ast.parse(code)
    class LocalPaths(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('OUTPUT','SOURCE_ROOT'):
                name=node.targets[0].id
                target=tmp_path/('inline_output' if name=='OUTPUT' else 'inline_source')
                node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(target))],keywords=[])
            return self.generic_visit(node)
    inline=ast.fix_missing_locations(LocalPaths().visit(ast.parse(codes[1])))
    env={};exec(compile(inline,'inline-source-only','exec'),env)
    for name in SOURCE_FILES:assert (env['SOURCE_ROOT']/name).read_bytes()==(ROOT/name).read_bytes()
    (env['OUTPUT']/'attempt.json').write_text('{}')
    sentinel=env['SOURCE_ROOT']/SOURCE_FILES[0];sentinel.write_text('local editable sentinel')
    reopened={};exec(compile(inline,'inline-source-only','exec'),reopened)
    assert 'SOURCE_ROOT' not in reopened and sentinel.read_text()=='local editable sentinel'
    extracted=tmp_path/'editable';extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for name in SOURCE_FILES:assert bundle.read(name)==(ROOT/name).read_bytes()
        bundle.extractall(extracted)
    for name in ['b_legacy_migration_v1_run','legacy_saved_reproduction_v1_run','legacy_reference_v1_run']:
        result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.'+name,'--help'],cwd=extracted,capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_saved_pixel_format_and_full_target_indices(tmp_path,current_books):
    pixel_path=tmp_path/'rgb8.pt'
    pixels=torch.zeros((),dtype=torch.uint8).expand(method.identity.VIDEO_SHAPE)
    torch.save(pixels,pixel_path)
    backend=object.__new__(runtime.WanBackend)
    loaded=backend.load_pixels(pixel_path)
    assert loaded.dtype==torch.uint8 and tuple(loaded.shape)==method.identity.VIDEO_SHAPE
    torch.save(torch.zeros(()).expand(method.identity.VIDEO_SHAPE),pixel_path)
    with pytest.raises(ValueError,match='uint8'):backend.load_pixels(pixel_path)
    latent=torch.zeros(method.identity.LATENT_SHAPE)
    book=current_books['key00']
    for index,time in enumerate([16,28]):
        for roi,(y0,y1,x0,x1) in enumerate(method.identity.ROIS):
            latent[0,4:16,time,y0:y1,x0:x1]=(book['directions'][roi,0]*(.25+.5*index)).float()
    values=method.observe_full(latent,book)
    assert [v['latent_time'] for v in values]==[16,28]
    assert values[0]['y'][0]==.25 and values[1]['y'][0]==.75
    assert all(v['phase_status']=='NOT_CERTIFIED' for v in values)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf')])
def test_legacy_nonfinite_file_rejected_before_extraction(tmp_path,bad):
    from main.tube_state.legacy_reference_v1.tensor_artifact import read_float_tensor_npy
    path=tmp_path/'bad.npy';values=array('f',[0.])*256;values[19]=bad
    write_float_tensor_npy(path,(4,4,4,4),values)
    with pytest.raises(ValueError,match='nonfinite latent value at flat index 19'):
        read_float_tensor_npy(path)
    with pytest.raises(ValueError,match='nonfinite latent value at flat index 19'):
        extract_public_reference(path,'registered_fixture',(4,4,4,4),old_config())


def test_comparison_cross_run_and_reused_side_metadata():
    observations={r['row']:method.unavailable('NOT_ENTERED',2 if r['history']=='local' else method.TARGETS[r['target']][1]) for r in method.plan()}
    result=runtime.evaluate(observations)
    comparisons=result['comparisons']
    assert comparisons['mp4_history']['cross_run'] is True
    assert comparisons['mp4_history']['reused_side']=='before'
    assert comparisons['local_codec']['cross_run'] is True
    assert comparisons['local_codec']['reused_side']=='after'
    for name in ['rgb8_history','full_codec']:
        assert comparisons[name]['cross_run'] is False and comparisons[name]['reused_side'] is None
    assert result['cross_run_MP4_history_contrast'] is True
