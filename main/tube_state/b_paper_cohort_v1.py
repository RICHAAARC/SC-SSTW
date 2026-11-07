"""Frozen twelve-content panel; pure protocol and received-only plan adapters."""
from copy import deepcopy

EXPERIMENT = 'paper-cohort-v1'
MODEL = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
OUTPUT_ROOT = '/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1'
VIEWS = ('FULL','CROP','DELETE1','REPEAT1','DELETE3')
LENGTHS = (181,101,180,182,178)
ARMS = ('OFF','P0','NEW')
KEYS = ('key00','key01')
CLAIMS = {'q0':'A6D39C5E','q1':'26D39C5E'}
STAGES = ('prepare','solve','replay','receive','identity')
GENERATION = dict(height=320,width=512,frames=181,fps=8,steps=50,guidance_scale=5.,max_sequence_length=512)
INNER = dict(transformer_conditional=100,transformer_unconditional=100,native_step=100,
             payload_gradient=25,D181=2,baseline_head=46,head_replay=46,K=4201,KT=4200,writer_BCH_encode=1790)
LIMITS = dict(receive_seconds=9450,identity_seconds=18900,rss_bytes=8*1024**3,
              codec_seconds=300,hard_seconds=600,relative_seconds=600,bch_calls=36990)
DENOMINATORS = dict(sources=1,method_operations=1114,time_operations=113,identity_operations=926,
    claims=60,quality_pairs=15,media=15,nodes=2466,edges=2451,NEW_nodes=822,NEW_edges=817,
    banks=180,bank_keys=360,raw_slots=57600,local_slots=3840,final_slots=1280,
    main_final_slots=960,auxiliary_final_slots=320,B_slots=160,U_slots=160,max_comparisons=1920)


def query_bits(name):
    return [int(b) for b in f'{int(CLAIMS[name],16):032b}']


def source_config(source_id):
    source = next(s for s in ROSTER if s['id']==source_id)
    return dict(experiment=EXPERIMENT,source=deepcopy(source),model_id=MODEL,generation=dict(GENERATION),
                writer=dict(key='watermark',gamma=1,bits=query_bits('q0')),arms=list(ARMS),views=list(VIEWS))


def validate(config):
    if config != source_config(config['source']['id']):
        raise ValueError('adopted source, writer, geometry and method semantics required')
    return config


def batch_sources(batch):
    if batch not in ('TEST-A','TEST-B'): raise ValueError('fixed TEST-A or TEST-B required')
    return [deepcopy(s) for s in ROSTER if s['batch']==batch]


def inputs():
    return [dict(id=f'input{a*5+i}',arm=arm,view=view,length=n)
            for a,arm in enumerate(ARMS) for i,(view,n) in enumerate(zip(VIEWS,LENGTHS))]


def strategies(item):
    return (f"B_{item['arm']}",f"K_{item['arm']}") + (
        ('U_NEW',) if item['arm']=='NEW' else ('P_PAIRED',) if item['arm']=='P0' else ())


def bank(item):
    n=item['length']
    return [dict(view_id=f"{item['id']}/E41/a{w+p}",start=w+p,query=q,query_start=w,phase=p,
                 encoder_input_support=list(range(w+p,w+p+41)))
            for q,w in enumerate((0,(n-45)//2,n-45)) for p in range(4)]


def key_for(strategy): return 'key01' if strategy.startswith('K_') else 'key00'


def operation_plan():
    rows={}
    def add(address,kind,stage):
        rows[address]=dict(kind=kind,stage=stage,status='NOT_STARTED',started=0,returned=0)
    for stage,names in dict(prepare=('generation_load','generate_OFF','generate_P0','capture_vae_load','decode_OFF','capture','head_prepare'),
        solve=('q0','target','solver','prediction','handoff'),replay=('replay_vae_load','native_replay')).items():
        for name in names: add(name,name,stage)
    for arm in ARMS:
        for op in ('encode','probe','decode_RGB'): add(f'{arm}/first/{op}',op,'receive')
    for item in inputs():
        stem=item['id']
        for op in ('encode','probe','decode_RGB','decode_Y','hard_receive','relative_compare'):
            add(f'time/{stem}/{op}',op,'receive')
        for op in ('plan','probe','decode_RGB'): add(f'identity/{stem}/{op}',op,'identity')
        for b in bank(item):
            for op in ('E41','project','key00','key01'):
                add(b['view_id']+'/'+op,'bits' if op.startswith('key') else op,'identity')
        for strategy in strategies(item):
            prefix=f'{stem}/{strategy}'
            for q in range(3): add(f'{prefix}/q{q}/local','local','identity')
            add(prefix+'/aggregate','aggregate','identity')
        for key in KEYS:
            for claim in CLAIMS: add(f'{stem}/{key}/{claim}','claim','identity')
    add('identity/model_load','load','identity')
    for pair in ('NEW_P0','P0_OFF','NEW_OFF'):
        for view in VIEWS: add(f'quality/{pair}/{view}','quality','identity')
    return rows


def layout():
    rows=[]
    for item in inputs():
        for strategy in strategies(item):
            stem=f"{item['id']}/{strategy}"
            base=dict(input_id=item['id'],strategy=strategy,key_id=key_for(strategy))
            for q in range(3):
                rows.append(dict(base,path=f'local/{stem}/q{q}.json',operation=f'{stem}/q{q}/local',kind='local'))
            rows.append(dict(base,path=f'final/{stem}.json',operation=stem+'/aggregate',kind='final'))
    return rows


def freeze(item, relative, new_relative=None):
    from . import b_relative_time_payload_v1 as prior
    time=prior.saved_relative(relative,item['length'])
    result=dict(time=time,bank=bank(item),strategies={})
    for strategy in strategies(item):
        selected=time; borrowed=None
        if strategy=='U_NEW': selected=dict(status='READY',mapping=list(range(item['length'])),source='PUBLIC_RECEIVED_CLOCK')
        if strategy=='P_PAIRED':
            selected=prior.saved_relative(new_relative,item['length'])
            borrowed='input'+str(10+VIEWS.index(item['view']))
        original='P_BASE' if strategy=='P_PAIRED' else 'U_NEW' if strategy=='U_NEW' else 'K_NEW' if strategy.startswith('K_') else 'B_NEW'
        plans=prior.plans(item['id'],item['length'],selected,original,borrowed_from=borrowed)
        for plan in plans:
            plan['arm']=strategy
            for group in plan['groups']:
                for ref in group['references']: ref['candidate_id']=strategy
        result['strategies'][strategy]=plans
    return result

ROSTER = [{'id': 'paper-joint-t01',
  'batch': 'TEST-A',
  'seed': 2026110701,
  'prompt': 'A locked-off wide shot of a blue bicycle parked in a leafy courtyard, with small leaves moving '
            'in a light breeze. Natural continuous motion, documentary style, with the entire scene visible '
            'throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t02',
  'batch': 'TEST-A',
  'seed': 2026110702,
  'prompt': 'A locked-off wide shot of a red toy train moving along a straight track on a wooden tabletop. '
            'Natural continuous motion, documentary style, with the entire scene visible throughout the '
            'shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t03',
  'batch': 'TEST-A',
  'seed': 2026110703,
  'prompt': 'A locked-off wide shot of pedestrians and a few cars moving through a city street intersection '
            'in daylight. Natural continuous motion, documentary style, with the entire scene visible '
            'throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t04',
  'batch': 'TEST-A',
  'seed': 2026110704,
  'prompt': 'A locked-off wide shot of clear water pouring into a glass bowl on a dark kitchen counter, with '
            'visible ripples and droplets. Natural continuous motion, documentary style, with the entire '
            'scene visible throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t05',
  'batch': 'TEST-B',
  'seed': 2026110705,
  'prompt': 'A locked-off wide shot of a field of golden wheat swaying in the wind beneath a cloudy sky. '
            'Natural continuous motion, documentary style, with the entire scene visible throughout the '
            'shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t06',
  'batch': 'TEST-B',
  'seed': 2026110706,
  'prompt': 'A locked-off wide shot of small boats tied to a harbor dock, with reflections moving on the '
            'water. Natural continuous motion, documentary style, with the entire scene visible throughout '
            'the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t07',
  'batch': 'TEST-B',
  'seed': 2026110707,
  'prompt': 'A locked-off wide shot of a stone fountain in a plaza, with several water jets and falling '
            'droplets. Natural continuous motion, documentary style, with the entire scene visible '
            'throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t08',
  'batch': 'TEST-B',
  'seed': 2026110708,
  'prompt': 'A locked-off wide shot of colorful fish swimming among plants inside a rectangular aquarium. '
            'Natural continuous motion, documentary style, with the entire scene visible throughout the '
            'shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t09',
  'batch': 'TEST-B',
  'seed': 2026110709,
  'prompt': 'A locked-off wide shot of a ceiling fan rotating slowly in a plain sunlit room. Natural '
            'continuous motion, documentary style, with the entire scene visible throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t10',
  'batch': 'TEST-B',
  'seed': 2026110710,
  'prompt': 'A locked-off wide shot of white curtains moving beside an open window and casting changing '
            'shadows on a wall. Natural continuous motion, documentary style, with the entire scene visible '
            'throughout the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t11',
  'batch': 'TEST-B',
  'seed': 2026110711,
  'prompt': 'A locked-off wide shot of a small flock of birds flying across an open sky above a distant tree '
            'line. Natural continuous motion, documentary style, with the entire scene visible throughout '
            'the shot.',
  'negative_prompt': ''},
 {'id': 'paper-joint-t12',
  'batch': 'TEST-B',
  'seed': 2026110712,
  'prompt': 'A locked-off wide shot of smoke rising from a brick chimney above a quiet roof. Natural '
            'continuous motion, documentary style, with the entire scene visible throughout the shot.',
  'negative_prompt': ''}]
