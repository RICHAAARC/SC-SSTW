"""Standard-library fixed record format and read-only basic recovery."""
from pathlib import Path
import json

ARMS = ('BASE','QIM')
LAYERS = ('preclip','clip','rgb8','h264')
BUDGETS = dict(D181=1, baseline_head=46, head_replay=46, K=4201, KT=4201,
               E41=16, codec_encode=2, codec_probe=2, codec_decode=2, DiT=0, training=0)


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def plan():
    rows = {f'{a}/{l}':dict(kind='time_read', slots=181*5*2*63, status='NOT_STARTED') for a in ARMS for l in LAYERS}
    rows.update({f'{a}/identity/{s}/{phase}':dict(kind='identity_read', slots=160, status='NOT_STARTED')
                 for a in ARMS for s in (0,40) for phase in range(4)})
    return rows


def denominators():
    return dict(sources=1, arms=2, frames=181, time_layers=4, time_read_slots=912240,
                identity_views=16, identity_local_bit_slots=2560, paired_identity_slots=1280)


def has_existing_output(output):
    output = Path(output)
    return output.exists() and any(output.iterdir())


def recover(output):
    output = Path(output); errors = {}; saved = {}
    for name in ('attempt','state','counts','solver','solver_progress','native','evaluation','result'):
        try:
            value = load(output/(name+'.json'))
            if not isinstance(value,dict): raise ValueError('saved record must be an object')
            saved[name] = value
        except (OSError, ValueError) as e: errors[name] = f'{type(e).__name__}: {e}'
    rows = plan()
    state = saved.get('state')
    legal = {'NOT_STARTED','STARTED','RETURNED','FAILED','INTERRUPTED'}
    for key, row in rows.items():
        value = state.get(key) if isinstance(state,dict) else None
        if not isinstance(value,dict) or not isinstance(value.get('status'),str) or value['status'] not in legal:
            row.update(status='UNRESOLVED_RECEIPT',cause='missing or invalid saved row')
        else:
            row.update(value)
            row.update(kind=plan()[key]['kind'],slots=plan()[key]['slots'])
    for row in rows.values():
        if row['status'] == 'STARTED': row['status'] = 'UNRESOLVED_INTERRUPTED'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_calls_this_invocation=0,
                denominators=denominators(), rows=rows, saved=saved, read_errors=errors)
