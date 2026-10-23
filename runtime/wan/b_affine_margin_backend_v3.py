"""Physical adapter; imported only after durable attempt. No work on import."""
from pathlib import Path
from runtime.wan.b_lowrate_local_time_label_backend_v1 import Backend as Previous
from runtime.wan import b_time_label_tensor_v1 as tensor
from main.tube_state import b_affine_margin_v3 as core
class Backend(Previous):
 def construct_new(self,p0,old,record):
  import torch
  p0=tensor.validate(p0,46);old=tensor.validate(old,46)
  z,r=core.construct(p0[0].numpy(),old[0].numpy(),record)
  record('native_receipt.json',r)
  if z is None:raise ValueError('V3 native construct unresolved: '+r['status'])
  return tensor.validate(torch.from_numpy(z[None].copy()),46)
 def received104(self,pixels,attack):
  import torch
  self.validate_pixels(pixels,181)
  if attack=='crop104':v=pixels[33:137]
  elif attack=='delete13_s52':v=torch.cat([pixels[33:85],pixels[98:150]])
  else:raise ValueError('fixed attack only')
  return self.validate_pixels(v.clone(),104)
 def window45(self,received,wi,phase):
  self.validate_pixels(received,104)
  if type(wi) is not int or wi not in range(3) or type(phase) is not int or phase not in range(8):raise ValueError('fixed window/phase')
  start=(0,26,52)[wi]+phase
  return self.validate_pixels(received[start:start+45].clone(),45)
 def infer(self,queries,save_arrays):return core.infer_many(queries,save_arrays)
 def paired_quality(self,left,right,meaning):
  import torch,math
  from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame,_stats,_combine
  if tuple(left.shape)!=tuple(right.shape) or tuple(left.shape[1:])!=(320,512,3) or len(left) not in (104,181):raise ValueError('paired complete pixel arrays')
  frames=[];adj=[];previous=None
  for t in range(len(left)):
   d=_frame(right[t])-_frame(left[t])
   if not bool(torch.isfinite(d).all()):raise ValueError('nonfinite quality')
   frames.append(dict(frame=t,whole=_stats(d)))
   if previous is not None:adj.append(dict(frame_before=t-1,frame_after=t,whole=_stats(d-previous)))
   previous=d
  whole=_combine([x['whole'] for x in frames])
  return dict(status='SCORED',meaning=meaning,frames=frames,adjacent=adj,frame_denominator=len(frames),adjacent_denominator=len(adj),whole=whole,adjacent_residual=_combine([x['whole'] for x in adj]),psnr_db_range1=-20*math.log10(whole['rms']) if whole['rms'] else None,quality_pass=None)
