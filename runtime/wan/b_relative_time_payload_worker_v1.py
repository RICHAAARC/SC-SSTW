"""One scientific worker; all geometry is saved before model or identity bits."""
from pathlib import Path
import time
from . import b_relative_time_payload_records_v1 as rec


def quality_pair(new, base):
    import numpy as np
    if new.shape != base.shape or new.dtype != np.uint8 or base.dtype != np.uint8 or new.ndim != 4:
        raise ValueError('paired actual uint8 frames of equal shape required')
    total, maximum = 0., 0.
    for a, b in zip(new, base):
        difference = (a.astype(np.float64)-b.astype(np.float64))/255.
        total += float(np.square(difference).sum())
        maximum = max(maximum, float(np.abs(difference).max()))
    mse = total/new.size
    return dict(status='RETURNED', frames=len(new), scalar_samples=int(new.size), data_range=1,
                mse=mse, psnr_db=None if mse == 0 else float(-10*np.log10(mse)),
                psnr_is_positive_infinity=mse == 0, zero_difference=mse == 0,
                maxabs=maximum, quality_pass=None, scope='TIME_PACKET_INCREMENT_NEW_VS_BASE')


def worker(output, backend_factory=None, evaluator=None, *, source_id='joint-relative-r01', defer_post=False):
    output = Path(output)
    if not (output/'attempt.json').is_file(): raise ValueError('reserved attempt required')
    try:
        with (output/'worker_started.json').open('x') as stream: stream.write('{"consumed": true}')
    except FileExistsError:
        return rec.recover(output)
    config = rec.read(output/'protocol.json'); source = Path(config['source'])
    journal = rec.Journal(output); start = time.monotonic()
    backend, failure, evaluation = None, None, None
    extras = dict(blind_terminal_saves=0, post=dict(started=0, returned=0), quality={})
    plans, raw = {}, {}
    for item in rec.inputs():
        for unit in rec.bank(item):
            raw[unit['view_id']] = dict(unit, input_id=item['id'], status='NOT_ENTERED', keys={})
    rec.write(output/'raw_bank.json', raw)
    try:
        from main.tube_state import b_relative_time_payload_v1 as method
        from main.tube_state import b_time_payload_cdc32_v1 as previous
        from main.tube_state import b_cdc32_aggregation_v1 as cdc
        for item in rec.inputs():
            def freeze(item=item):
                if item['arm'] == 'NEW':
                    path = source/'blind'/source_id/'NEW'/item['view']/'relative.json'
                    try: relative = rec.read(path)
                    except (OSError, ValueError) as error: relative = dict(input_error=str(error))
                    value = method.new_input(item['id'], item['length'], relative)
                else:
                    partner = 'input'+str(rec.VIEWS.index(item['view']))
                    value = method.paired_input(item['id'], item['length'], plans[partner], partner)
                rec.write(output/'plans'/f"{item['id']}.json", value)
                return rec.read(output/'plans'/f"{item['id']}.json")
            plans[item['id']] = journal.call(item['id']+'/plan', freeze)
        rec.write(output/'plans.json', plans); plans = rec.read(output/'plans.json')
        rec.write(output/'plans_frozen.json', dict(status='ALL_PLANS_SAVED', saved_at=time.time(),
                                                  physical_bank=raw, message_visible=False))
        if backend_factory is None:
            from .b_relative_time_payload_backend_v1 import Backend
            backend_factory = Backend
        backend = backend_factory(config['model_id'])
        journal.call('model_load', lambda: save_return(output/'model.json', backend.load()))
        for view in rec.VIEWS:
            paired_pixels = {}
            for item in [i for i in rec.inputs() if i['view'] == view]:
                stem, length = item['id'], item['length']
                path = source/'media'/source_id/item['arm']/view/'received.mp4'
                journal.call(stem+'/probe', lambda: save_return(output/'media'/stem/'probe.json', backend.probe(path, length)))
                pixels = journal.call(stem+'/decode_RGB', lambda: backend.decode(path, length))
                paired_pixels[item['arm']] = pixels
                for unit in rec.bank(item):
                    uid = unit['view_id']; folder = output/'bank'/uid
                    latent = journal.call(uid+'/E41', lambda: backend.encode(pixels, unit['start'], length))
                    raw[uid].update(status='ENCODED', E41='RETURNED'); rec.write(output/'raw_bank.json', raw)
                    projection = journal.call(uid+'/project', lambda: save_return(folder/'projection.json', backend.project(latent)))
                    latent = None
                    raw[uid]['project'] = 'RETURNED'
                    for key in ('key00', 'key01') if item['arm'] == 'NEW' else ('key00',):
                        def decode(key=key):
                            decoded = backend.bits(projection, uid, key)
                            rec.write(folder/f'{key}.json', decoded)
                            flat = {row['evidence_id']: {key: {str(bit['local_bit']): bit for bit in row['bits']}}
                                    for region in ('R0', 'R1') for row in decoded['regions'][region]['local_cells']}
                            rec.write(folder/f'{key}_cache.json', flat)
                            return decoded['status']
                        journal.call(uid+'/'+key, decode)
                        raw[uid]['keys'][key] = 'RETURNED'; raw[uid]['status'] = 'RETURNED'
                        rec.write(output/'raw_bank.json', raw)
                    projection = None
                loaded, memo = {}, {}
                def cache(eid, key, bit):
                    uid = eid.rsplit('/row', 1)[0]; address = (uid, key)
                    if address not in loaded: loaded[address] = rec.read(output/'bank'/uid/f'{key}_cache.json')
                    return loaded[address][eid][key][str(bit)]
                for strategy, group in plans[stem]['strategies'].items():
                    local = []
                    receipts = {uid: dict(row, status='RETURNED' if row['keys'].get(group[0]['key_id']) == 'RETURNED'
                                         else 'TECHNICAL_UNREAD') for uid, row in raw.items()}
                    for q, plan in enumerate(group):
                        attached = previous.attach_receipts(plan, receipts)
                        value = journal.call(f'{stem}/{strategy}/q{q}/local', lambda: save_return(
                            output/'local'/stem/strategy/f'q{q}.json', cdc.read_time_local(attached, cache, memo=memo)))
                        local.append(value)
                    journal.call(f'{stem}/{strategy}/aggregate', lambda: save_return(
                        output/'final'/stem/f'{strategy}.json', cdc.aggregate_time(local)))
                loaded.clear(); pixels = None
            quality_start = time.monotonic()
            extras['quality'][view] = dict(started=1, returned=0)
            rec.write(output/'extra_counts.json', extras)
            quality = quality_pair(paired_pixels['NEW'], paired_pixels['BASE'])
            quality['seconds'] = time.monotonic()-quality_start
            rec.write(output/'quality'/f'{view}.json', quality)
            extras['quality'][view].update(returned=1, seconds=quality['seconds'])
            paired_pixels.clear()
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    try:
        rec.finalize_blind(output, failure); extras['blind_terminal_saves'] = 1
        rec.write(output/'extra_counts.json', extras)
        if failure is None and not defer_post:
            if evaluator is None:
                from .b_relative_time_payload_evaluation_v1 import evaluate
                evaluator = evaluate
            extras['post']['started'] = 1; rec.write(output/'extra_counts.json', extras)
            post_start = time.monotonic()
            evaluation = evaluator(output, source)
            extras['post'].update(returned=1, seconds=time.monotonic()-post_start)
    except BaseException as error:
        failure = failure or f'{type(error).__name__}: {error}'
    resources = {}
    if backend is not None:
        try: resources = backend.close()
        except BaseException as error: failure = failure or f'close: {error}'
    rec.write(output/'backend_resources.json', resources)
    rec.write(output/'extra_counts.json', extras)
    quality = {view: rec.read(output/'quality'/f'{view}.json') if (output/'quality'/f'{view}.json').is_file()
               else dict(status='NOT_ENTERED') for view in rec.VIEWS}
    rec.write(output/'quality.json', quality)
    result = dict(status='STOPPED_FIXED_ATTEMPT' if failure else 'COMPLETED', failure=failure,
                  denominators=rec.DENOMINATORS, evaluation_status=evaluation.get('status') if evaluation else 'NOT_EVALUATED',
                  worker_seconds=time.monotonic()-start, scientific_pass=None, consumed=True, post_deferred=defer_post)
    rec.write(output/'result.json', result)
    return result


def save_return(path, value):
    rec.write(path, value)
    return value
