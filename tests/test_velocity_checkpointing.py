"""CPU architecture checks; no pretrained weights or GPU feasibility claims."""
import gc
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
from main.tube_state import state_clock, projection_margin as carrier, velocity_coefficients as method
from runtime.wan import velocity_direction as runtime

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def tiny_wan(dtype=torch.float32, *, patch=(1,2,2), channels=2, layers=3):
    torch.manual_seed(918)
    model=WanTransformer3DModel(patch_size=patch,num_attention_heads=2,attention_head_dim=8,
        in_channels=channels,out_channels=channels,text_dim=8,freq_dim=8,ffn_dim=32,num_layers=layers)
    model.eval().requires_grad_(False).to(dtype=dtype)
    if dtype==torch.bfloat16:
        for name,parameter in model.named_parameters():
            if any(part in model._keep_in_fp32_modules for part in name.split('.')):
                parameter.data=parameter.data.float()
    model.disable_gradient_checkpointing()
    return model


def counter():
    counts=Counter()
    def count(kind,done):counts[kind+('_completed' if done else '_attempted')]+=1
    return counts,count


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_real_wan_full_tail_matches_outer_and_native(dtype,monkeypatch):
    # Full production coefficient shape, scatter, loss and six-step UniPC path.
    # Only the randomly initialized architecture is reduced (larger patches).
    model=tiny_wan(dtype,patch=(2,8,8),channels=16,layers=2)
    prompt=torch.randn(1,3,8).to(dtype);negative=-prompt
    book=state_clock.codebook(b'WanProjection-first-validation-key-v1')
    directions=torch.tensor(book['directions']);codes=torch.tensor(book['codes'])
    s=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3.)
    s.set_timesteps(50);s.set_begin_index(0)
    z=torch.randn(carrier.SHAPE)*.1
    for i in range(44):z=s.step(torch.sin(z*.1),s.timesteps[i],z,return_dict=False)[0]
    results=[]
    live_history=[]
    native_step=UniPCMultistepScheduler.step
    def observed_step(scheduler,*args,**kwargs):
        value=native_step(scheduler,*args,**kwargs)
        if torch.is_grad_enabled() and any(t is not None and t.grad_fn is not None for t in scheduler.model_outputs):
            live_history.append((scheduler.step_index,scheduler.last_sample.grad_fn is not None))
        return value
    monkeypatch.setattr(UniPCMultistepScheduler,'step',observed_step)
    for outer,inner in ((False,False),(True,False),(True,True)):
        a=torch.zeros(method.COEFFICIENT_SHAPE,requires_grad=True)
        counts,count=counter()
        output=runtime.tail(SimpleNamespace(transformer=model),s,z,prompt,negative,dtype,5.,a,directions,.02,
            count,lambda *args:None,use_checkpoint=outer,use_block_checkpoint=inner)
        forward_counts=counts.copy()
        forward_history=list(live_history)
        gradient=torch.autograd.grad(method.terminal_loss(output,directions,codes[0]),a)[0]
        assert live_history==forward_history
        assert [index for index,linked in live_history[-6:]]==list(range(45,51))
        assert all(linked for index,linked in live_history[-5:])
        assert counts['scheduler_step_completed']==forward_counts['scheduler_step_completed']==6
        assert counts['shadow_step_completed']==forward_counts['shadow_step_completed']==6
        assert counts['transformer_completed']==12
        assert s.step_index==44 and model.gradient_checkpointing is False
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert torch.isfinite(gradient).all() and (gradient.abs().sum(dim=1)>0).all()
        if outer:assert counts['transformer_replay_completed']==10
        if inner:
            assert counts['transformer_block_forward_completed']==20
            assert counts['transformer_block_outer_replay_completed']==20
            assert counts['transformer_block_replay_completed']==20
        results.append((output.detach(),gradient.detach()))
    for output,gradient in results[1:]:
        torch.testing.assert_close(output,results[0][0],rtol=0,atol=0)
        torch.testing.assert_close(gradient,results[0][1],rtol=0,atol=0)


class WeakStorageInventory:
    """C++ weak storage handles also see aliases with different Tensor wrappers."""
    def __init__(self):self.handles={}
    def add(self,tensor):
        storage=tensor.untyped_storage()
        handle=storage._weak_ref()
        if handle in self.handles:
            torch.UntypedStorage._free_weak_ref(handle)
        else:self.handles[handle]=storage.nbytes()
    def live_bytes(self):
        return sum(size for handle,size in self.handles.items() if not torch.UntypedStorage._expired(handle))
    def close(self):
        for handle in self.handles:torch.UntypedStorage._free_weak_ref(handle)
        self.handles.clear()
    def __del__(self):self.close()


def test_nested_boundary_lifetime_and_exception_restoration():
    model=tiny_wan()
    prompt=torch.randn(1,3,8);timestep=torch.tensor([2.])
    previous_function=model._gradient_checkpointing_func
    original_refs=[];replay_refs=[];phase=['forward']
    boundary_peak={'forward':0,'replay':0}
    boundary_stores={phase:WeakStorageInventory() for phase in boundary_peak}
    def observe(module,args):
        refs=original_refs if phase[0]=='forward' else replay_refs
        refs.append(weakref.ref(args[0]))
        store=boundary_stores[phase[0]]
        store.add(args[0])
        boundary_peak[phase[0]]=max(boundary_peak[phase[0]],store.live_bytes())
    hooks=[block.register_forward_pre_hook(observe) for block in model.blocks]
    counts,count0=counter()
    def count(kind,done):
        count0(kind,done)
        if kind=='transformer_replay' and not done:phase[0]='replay'
    x=torch.randn(1,2,4,4,4,requires_grad=True)
    y=x
    for _ in range(10):
        y=runtime.transformer_output(model,y,timestep,prompt,count,True)
    gc.collect()
    assert len(original_refs)==30 and all(ref() is None for ref in original_refs)
    assert boundary_stores['forward'].live_bytes()==0
    torch.autograd.grad(y.square().mean(),x)
    gc.collect()
    assert counts['transformer_replay_completed']==10
    assert counts['transformer_block_replay_completed']==30
    assert all(ref() is None for ref in replay_refs)
    assert boundary_stores['replay'].live_bytes()==0
    # At most one call's three hidden boundaries (1*16 tokens*16 width*4 bytes).
    assert boundary_peak['replay']<=3*16*16*4
    print('CPU observed live block-input storage peak, not allocator peak:',boundary_peak)
    assert model.gradient_checkpointing is False and model._gradient_checkpointing_func is previous_function
    # Fail inside a real block replay; leave no global hooks, input captures, or flags.
    phase[0]='forward';original_refs.clear();replay_refs.clear()
    def fail(kind,done):
        count(kind,done)
        if kind=='transformer_block_replay' and not done:raise RuntimeError('injected block replay failure')
    def failed_graph():
        value=torch.randn(1,2,4,4,4,requires_grad=True)
        out=runtime.transformer_output(model,value,timestep,prompt,fail,True)
        torch.autograd.grad(out.square().mean(),value)
    with pytest.raises(RuntimeError,match='injected block replay failure'):failed_graph()
    gc.collect()
    assert model.gradient_checkpointing is False and model._gradient_checkpointing_func is previous_function
    assert all(ref() is None for ref in original_refs+replay_refs)
    assert all(store.live_bytes()==0 for store in boundary_stores.values())
    for store in boundary_stores.values():store.close()
    for hook in hooks:hook.remove()
    # A subsequent independent call remains usable after the failed graph dies.
    value=torch.randn(1,2,4,4,4,requires_grad=True)
    out=runtime.transformer_output(model,value,timestep,prompt,count0,True)
    assert torch.isfinite(torch.autograd.grad(out.sum(),value)[0]).all()


class SavedStorage:
    """Weak storage inventory, excluding model weights; not process/CUDA memory."""
    def __init__(self,model):
        self.excluded={p.untyped_storage().data_ptr() for p in model.parameters()}
        self.refs=[]
    def pack(self,tensor):
        self.refs.append(weakref.ref(tensor))
        return tensor
    def unpack(self,tensor):return tensor
    def report(self):
        tensors=[r() for r in self.refs if r() is not None]
        tensors=[t for t in tensors if t.numel() and t.untyped_storage().data_ptr() not in self.excluded]
        unique={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        return {'logical_bytes':sum(t.numel()*t.element_size() for t in tensors),
                'unique_storage_bytes':sum(unique.values()),'tensor_slots':len(tensors)}


def test_nested_saved_storage_does_not_keep_ten_calls_block_boundaries():
    model=tiny_wan();prompt=torch.randn(1,3,8);time=torch.tensor([2.])
    reports={}
    for mode in ('outer','block_only','nested'):
        count0,count=counter();storage=SavedStorage(model)
        x=torch.randn(1,2,4,4,4,requires_grad=True);y=x
        with torch.autograd.graph.saved_tensors_hooks(storage.pack,storage.unpack):
            for _ in range(10):
                if mode=='block_only':
                    with runtime.block_checkpointing(model,count,'transformer',True):
                        y=model(hidden_states=y,timestep=time,encoder_hidden_states=prompt,return_dict=False)[0]
                else:
                    y=runtime.transformer_output(model,y,time,prompt,count,True,use_block_checkpoint=mode=='nested')
        reports[mode]=storage.report()
        torch.autograd.grad(y.square().mean(),x)
        del x,y
    assert reports['nested']==reports['outer']
    assert reports['nested']['unique_storage_bytes']<reports['block_only']['unique_storage_bytes']
    print('CPU saved storage after ten forwards (not peak allocator bytes):',reports)
