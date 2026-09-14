"""Explicit bounded recomputation; no saved-activation transfer to host."""


class CheckpointLedger:
    """Additional block/chunk replays have their own budget, never top-level calls."""
    def __init__(self, limits, *, resource_check=None):
        self.limits = dict(limits)
        for name in ('transformer_block', 'vae_chunk'):
            if type(self.limits.get(name)) is not int or self.limits[name] < 0:
                raise ValueError('explicit nonnegative recomputation limit: ' + name)
        self.counts = {name: {phase: dict(attempted=0, completed=0) for phase in ('forward','recompute')} for name in self.limits}
        self.resource_check = resource_check
        self.boundaries = {}
        self._seen_boundary_storage = {}
    def new_decode(self):
        index = len(self.boundaries)
        self.boundaries[index] = {phase: dict(chunks=[], unique_storage_bytes=0) for phase in ('forward','recompute')}
        return index
    def record_boundary(self, decode_id, phase, chunk_index, cache):
        # Only numbers/identities are retained, never tensors or storage objects.
        unique = {}
        logical = 0
        for value in cache:
            if not hasattr(value, 'untyped_storage'):continue
            storage=value.untyped_storage()
            key=(value.device.type,value.device.index,storage.data_ptr(),storage.nbytes())
            unique[key]=storage.nbytes();logical+=value.numel()*value.element_size()
        record=self.boundaries[decode_id][phase]
        seen=self._seen_boundary_storage.setdefault((decode_id,phase),set())
        record['unique_storage_bytes']+=sum(size for key,size in unique.items() if key not in seen)
        seen.update(unique)
        record['chunks'].append(dict(index=chunk_index, tensor_slots=sum(hasattr(x,'untyped_storage') for x in cache), logical_bytes=logical, unique_storage_bytes=sum(unique.values())))
    def summary(self):return dict(limits=self.limits,counts=self.counts,cache_boundaries=self.boundaries)
    def start(self, name, phase):
        if self.resource_check:self.resource_check()
        counter = self.counts[name][phase]
        if phase == 'recompute' and counter['attempted'] >= self.limits[name]:
            raise RuntimeError('RECOMPUTE_BUDGET_' + name)
        counter['attempted'] += 1
    def finish(self, name, phase):
        self.counts[name][phase]['completed'] += 1


def checkpoint_call(ledger, name, function, *args, boundary=None):
    """A fresh wrapper per segment: replay always gets its own functional inputs."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    calls = 0
    def measured(*inputs):
        nonlocal calls
        phase = 'forward' if calls == 0 else 'recompute'
        calls += 1
        ledger.start(name, phase)
        result = function(*inputs)
        if boundary is not None:boundary(phase,result)
        ledger.finish(name, phase)
        return result
    # Complete each replay, so the extra-work ledger counts whole block/chunk calls.
    with set_checkpoint_early_stop(False):
        return checkpoint(measured, *args, use_reentrant=False, preserve_rng_state=True)


def enable_transformer_checkpointing(transformer, ledger):
    def counted(module, *args):
        return checkpoint_call(ledger, 'transformer_block', module, *args)
    transformer.enable_gradient_checkpointing(gradient_checkpointing_func=counted)


def checkpoint_decode(vae, z, ledger):
    """Use native decode with the previously exercised functional frame wrapper.

    Engineering source: archived SC-SSTW-Feasibility flow_guidance_embedder.py
    _checkpoint_wan_decoder_frames. No archive import or scientific claim.
    Native decode owns post_quant_conv, chunk scheduling, patch/clamp and cache
    clearing. Replay uses only explicit cache tensors and a local cursor.
    """
    import torch
    if vae.use_tiling:raise ValueError('frame checkpoint budget excludes spatial tiling')
    if z.shape[0] != 1:raise ValueError('single-video checkpoint decoder')
    decoder = vae.decoder
    original_forward = decoder.forward
    decode_id = ledger.new_decode()
    call_index = 0

    def checkpointed_forward(x, feat_cache=None, feat_idx=None, first_chunk=False):
        nonlocal call_index
        if feat_cache is None or feat_idx is None or feat_idx[0] != 0:
            raise ValueError('native decoder causal cache/cursor unavailable')
        layout = []
        cache_tensors = []
        for item in feat_cache:
            if torch.is_tensor(item):
                layout.append(('tensor',len(cache_tensors)));cache_tensors.append(item)
            elif item is None:layout.append(('none',None))
            elif item == 'Rep':layout.append(('rep',None))
            else:raise ValueError('unexpected decoder input cache value')
        layout = tuple(layout)
        expected_output_layout = None
        expected_consumed = None
        chunk_index = call_index

        def functional_frame(x_value,*cache_values):
            nonlocal expected_output_layout,expected_consumed
            local = [cache_values[index] if kind=='tensor' else ('Rep' if kind=='rep' else None) for kind,index in layout]
            cursor = [0]
            output = original_forward(x_value,feat_cache=local,feat_idx=cursor,first_chunk=first_chunk)
            consumed = int(cursor[0])
            if consumed <= 0 or consumed > len(local):raise ValueError('decoder cache consumption invalid')
            if expected_consumed is None:expected_consumed=consumed
            elif consumed != expected_consumed:raise ValueError('replay changed decoder cache consumption')
            kinds=[];tensors=[]
            for item in local:
                if torch.is_tensor(item):kinds.append('tensor');tensors.append(item)
                elif item is None:kinds.append('none')
                elif item == 'Rep':kinds.append('rep')
                else:raise ValueError('unexpected decoder output cache value')
            kinds=tuple(kinds)
            if expected_output_layout is None:expected_output_layout=kinds
            elif kinds != expected_output_layout:raise ValueError('replay changed decoder cache layout')
            return (output,*tensors)

        outputs=checkpoint_call(ledger,'vae_chunk',functional_frame,x,*cache_tensors,
            boundary=lambda phase,result:ledger.record_boundary(decode_id,phase,chunk_index,result[1:]))
        if expected_output_layout is None or len(outputs)!=1+expected_output_layout.count('tensor'):
            raise ValueError('decoder checkpoint output malformed')
        tensors=iter(outputs[1:])
        feat_cache[:]=[next(tensors) if kind=='tensor' else ('Rep' if kind=='rep' else None) for kind in expected_output_layout]
        feat_idx[0]=expected_consumed
        call_index+=1
        return outputs[0]

    decoder.forward=checkpointed_forward
    try:
        decoded=vae.decode(z,return_dict=False)[0]
        if call_index != z.shape[2]:raise ValueError('native decoder chunk count differs from latent frames')
        return decoded
    finally:
        decoder.forward=original_forward
