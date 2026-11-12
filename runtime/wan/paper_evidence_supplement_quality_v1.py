"""User-run MUSIQ-SPAQ: fixed existing media and frame indices only."""
from pathlib import Path
from collections import Counter
import math
import time
from main.tube_state import paper_evidence_supplement_v1 as p
from . import paper_evidence_supplement_records_v1 as rec

WEIGHTS='https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth'
# Integer zero is ignored by Decord. This positive fraction sets the recovery
# threshold below one, making its first cached-frame replacement raise.
DECODE_FAULT_FRACTION=1e-12


def preprocess(frame):
    import torch
    import torch.nn.functional as F
    if not isinstance(frame,torch.Tensor) or frame.ndim!=3 or frame.shape[0]!=3 or frame.dtype!=torch.uint8:
        raise ValueError('RGB uint8 CHW frame required')
    image=frame.float().unsqueeze(0);height,width=frame.shape[-2:]
    if min(height,width)<1: raise ValueError('empty frame')
    if max(height,width)>512:
        scale=512/max(height,width);shape=(int(height*scale),int(width*scale))
        if min(shape)<1: raise ValueError('invalid resized dimensions')
        image=F.interpolate(image,size=shape,mode='bilinear',align_corners=False,antialias=False)
    return image.div(255.)


class Backend:
    def __init__(self,cache,device='cuda'):
        self.cache=Path(cache);self.device=device;self.model=None
        self.calls=dict(video_readers=0,decode_batches=0,forward_calls=0)

    def load(self):
        import urllib.request
        from pyiqa.archs.musiq_arch import MUSIQ
        self.cache.mkdir(parents=True,exist_ok=True);path=self.cache/'musiq_spaq_ckpt-358bb6af.pth'
        if not path.is_file():
            temporary=path.with_suffix('.partial');urllib.request.urlretrieve(WEIGHTS,temporary);temporary.replace(path)
        self.model=MUSIQ(pretrained_model_path=str(path)).float().eval().to(self.device)
        return dict(model='MUSIQ-SPAQ',dtype='float32',device=self.device,weights_source=WEIGHTS)

    def decode(self,path):
        import decord
        import torch
        self.calls['video_readers']+=1
        video=decord.VideoReader(str(path),num_threads=1,fault_tol=DECODE_FAULT_FRACTION);length=len(video)
        if not 0<DECODE_FAULT_FRACTION*length<1:
            raise ValueError('decoder recovery threshold must reject the first substituted frame')
        indices=[index for index in p.INDICES if index<length]
        if indices: self.calls['decode_batches']+=1
        frames=video.get_batch(indices).asnumpy() if indices else []
        result={index:torch.from_numpy(frame.copy()).permute(2,0,1) for index,frame in zip(indices,frames)}
        return dict(actual_frames=length,frames=result)

    def score(self,frame):
        import torch
        with torch.inference_mode():
            image=preprocess(frame).to(device=self.device,dtype=torch.float32)
            self.calls['forward_calls']+=1
            value=self.model(image)
        if value.numel()!=1: raise ValueError('one scalar MUSIQ score required')
        score=float(value.detach().cpu().item())
        if not math.isfinite(score): raise ValueError('nonfinite MUSIQ score')
        return score


def readonly(output):
    """A consumed package is never resumed, including after an abrupt process kill."""
    output=Path(output);attempt=rec.optional(output/'attempt.json') or {}
    operations=rec.optional(output/'operations.json') or {}
    for stream in attempt.get('streams',[]):
        for suffix in ('decode',*(f'frame{i}' for i in p.INDICES)):
            key=stream['id']+'/'+suffix
            update=rec.optional(output/'operation_updates'/(key+'.json'))
            if isinstance(update,dict): operations[key]=update
    update=rec.optional(output/'operation_updates/model_load.json')
    if isinstance(update,dict): operations['model_load']=update
    for row in operations.values():
        if row.get('status')=='STARTED': row['status']='INTERRUPTED_CONSUMED'
    return dict(status='READ_ONLY_CONSUMED',saved=rec.optional(output/'quality_summary.json'),
        fixed_logical_streams=84,fixed_frame_slots=1344,
        operation_statuses=dict(Counter(r.get('status','UNKNOWN') for r in operations.values())),
        recovered_frame_scores=sum(r.get('kind')=='frame_score' and r.get('returned')==1 for r in operations.values()),
        note='No model/decode/score retry; absent summary or slots remain unknown. Small operation updates retain scores.')


def run(result_root=p.ROOT,output=p.ROOT+'/paper_evidence_supplement_v1/attempt01/quality',cache='/content/paper-evidence-supplement-cache',*,backend_factory=Backend,baseline_references=None):
    root=Path(result_root).resolve();cache=Path(cache).resolve()
    output=rec.separate_output(output,[root/'main_method',root/'baselines'])
    if 'Video-WM_Result' in cache.parts or cache==root or cache.is_relative_to(root) or root.is_relative_to(cache) or cache==output or cache.is_relative_to(output) or output.is_relative_to(cache): raise ValueError('model cache must be separate from retained results')
    streams=p.streams(str(root));book=rec.QualityBook(output,streams,dict(result_root=str(root),cache=str(cache),budget=p.QUALITY_BUDGET))
    if book.reused: return readonly(output)
    if baseline_references is None: baseline_references=rec.baseline_references(root)
    rec.write(output/'baseline_references.json',baseline_references)
    frames={s['id']:[dict(source=s['source'],stream=s['stream'],frame_index=i,status='NOT_STARTED',score=None) for i in p.INDICES] for s in streams}
    clips=[];physical={};interrupted=False;failure=None;began=time.monotonic();backend=None
    def score_frame(frame):
        value=backend.score(frame)
        if type(value) not in (int,float) or not math.isfinite(value): raise ValueError('finite scalar MUSIQ score required')
        return float(value)
    try:
        backend=backend_factory(cache)
        loaded=book.call('model_load',backend.load)
        if loaded is not None:
            rec.write(output/'model.json',loaded)
            for stream in streams:
                sid=stream['id'];pointer=str(Path(stream['media']).resolve())
                if pointer in physical:
                    earlier=physical[pointer];book.rows[sid+'/decode'].update(status='REUSED_POINTER',pointer_source=earlier)
                    frames[sid]=[dict(row,source=stream['source'],stream=stream['stream'],reused_from=earlier) for row in frames[earlier]]
                    for index in p.INDICES: book.rows[sid+f'/frame{index}'].update(status='REUSED_POINTER',pointer_source=earlier)
                    book.save();continue
                physical[pointer]=sid;decoded=book.call(sid+'/decode',lambda:backend.decode(pointer))
                if decoded is None:
                    for row in frames[sid]: row.update(status='DECODE_UNAVAILABLE',error=book.rows[sid+'/decode'].get('error'))
                else:
                    rec.write(output/'media'/stream['source']/(stream['stream']+'.json'),dict(media=pointer,actual_frames=decoded['actual_frames'],fixed_indices=list(p.INDICES)))
                    for row in frames[sid]:
                        index=row['frame_index'];key=sid+f'/frame{index}'
                        if index not in decoded['frames']:
                            row.update(status='MISSING_FIXED_FRAME',actual_frames=decoded['actual_frames']);continue
                        score=book.call(key,lambda:score_frame(decoded['frames'][index]))
                        row.update(status='RETURNED' if score is not None else book.rows[key]['status'],score=score,error=book.rows[key].get('error'))
                        rec.write(output/'frames'/stream['source']/(stream['stream']+'.json'),frames[sid])
                rec.write(output/'frames'/stream['source']/(stream['stream']+'.json'),frames[sid])
                del decoded
        else:
            for rows in frames.values():
                for row in rows: row.update(status='MODEL_UNAVAILABLE',error=book.rows['model_load'].get('error'))
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}'
        if isinstance(error,(KeyboardInterrupt,SystemExit)): interrupted=True;raise
    finally:
        for key,row in book.rows.items():
            if row['status']=='STARTED': row['status']='INTERRUPTED_CONSUMED'
            elif row['status']=='NOT_STARTED': row['status']='NOT_EXECUTED_RETAINED'
        book.save()
        for stream in streams:
            rows=frames[stream['id']]
            for row in rows:
                if row['status']=='NOT_STARTED':
                    operation=book.rows[stream['id']+f"/frame{row['frame_index']}"]
                    row.update(status=operation['status'],error=operation.get('error'))
            clips.append(p.clip_summary(stream,rows))
        flat=[row for stream in streams for row in frames[stream['id']]]
        pairs=p.paired_scores(clips,baseline_references)
        rec.table(output,'quality_frames',flat);rec.table(output,'quality_clips',clips);rec.table(output,'quality_pairs',pairs)
        summary=dict(title=p.TITLE,budget=p.QUALITY_BUDGET,logical_streams=len(clips),planned_frame_slots=len(flat),
            frame_statuses=dict(Counter(row['status'] for row in flat)),complete_clips=sum(row['status']=='COMPLETE' for row in clips),
            operation_counts={kind:dict(started=sum(row['started'] for row in book.rows.values() if row['kind']==kind),returned=sum(row['returned'] for row in book.rows.values() if row['kind']==kind)) for kind in ('model_load','media_decode','frame_score')},
            distributions={name:p.distribution([c['complete_mean'] for c in clips if c['stream']==name]) for name in p.STREAMS},
            paired_distributions={name:p.distribution([r['difference'] for r in pairs if r['left']+'-'+r['right']==name]) for name in sorted({r['left']+'-'+r['right'] for r in pairs})},
            actual_backend_calls=getattr(backend,'calls',{}),
            elapsed_seconds=time.monotonic()-began,interrupted=interrupted,failure=failure,status='FIXED_SLOTS_ACCOUNTED',scientific_pass=None,
            scope='sampled-frame imaging proxy only; no flicker, transient-artifact or temporal-consistency certification')
        rec.write(output/'quality_summary.json',summary)
        report=['# '+p.TITLE,'','仅为固定帧成像代理；非VBench官方全帧/总分，不评闪烁、短暂伪影、时间连贯、审美或提示词。',
            f"完整流 {summary['complete_clips']}/84；固定帧槽1344。科学PASS未定义。",'',
            '| 流 | 完整源数/12 | 完整源平均原始分 |','|---|---:|---:|']
        for name,dist in summary['distributions'].items():
            score='未知' if dist['mean'] is None else f"{dist['mean']:.6f}"
            report.append(f"| {name} | {dist['valid_count']} | {score} |")
        report += ['', '只使用16帧全齐的完整均值；部分均值保留在quality_clips表，不混入完整汇总。',
            '配对差值要求同源固定索引全齐；baseline还需明确保存的OFF路径匹配。没有路径依据时差值未知，绝对WM分仍保留。',
            '实际调用数及失败见quality_summary.json与operations.json。再次Run all只读，不补帧、不重载模型、不重跑失败包。','']
        (output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    return summary
