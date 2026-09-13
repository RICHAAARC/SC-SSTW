"""CPU external review only. Never called by generation/public acquisition."""
import argparse,json
from pathlib import Path
from main.sc_sstw.local_temporal_warp import reference_rows

def evaluate(stream,annotations,output,off=None):
    import numpy as np
    from PIL import Image,ImageDraw
    stream=Path(stream);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((stream/'decode_metadata.json').read_text());h,w=meta['height'],meta['width']
    data=np.frombuffer((stream/'decoded_gray.raw').read_bytes(),dtype=np.uint8).reshape(-1,h,w)
    q=json.loads((stream/'observations.json').read_text());a=json.loads(Path(annotations).read_text())
    if len(data)!=31 or len(q)!=31 or len(a)!=31:raise ValueError('31 full-axis rows required')
    rows=reference_rows(q,a,w,h)
    # A priori outer border region, well outside fixed mask; diagnostic only.
    region=np.ones((h,w),bool);region[40:320,32:448]=False
    im=Image.fromarray(data[0]).convert('RGB');ImageDraw.Draw(im).rectangle((32,40,447,319),outline='red',width=2);im.save(output/'fixed_background_region.png')
    ref=None
    if off is not None:
        ref=np.frombuffer((Path(off)/'decoded_gray.raw').read_bytes(),dtype=np.uint8).reshape(data.shape)
    for i,row in enumerate(rows):
        r=a[i];outline=r.get('outline_bounds_pixel');row['outline_bounds_pixel']=outline;row['width_height_pixel']=None
        if outline is not None:
            if len(outline)!=4 or not all(isinstance(x,(float,int)) and np.isfinite(x) for x in outline) or not(0<=outline[0]<outline[1]<w and 0<=outline[2]<outline[3]<h):raise ValueError('outline bounds')
            if r.get('status')=='REVIEWED' and r.get('alignment_confirmed') is True:row['width_height_pixel']=[outline[1]-outline[0],outline[3]-outline[2]]
        row['shape_review']=r.get('shape_review','UNDETERMINED');row['background_review']=r.get('background_review','UNDETERMINED');row['quality_review']=r.get('quality_review','UNDETERMINED')
        row['width_height_change_pixel']=[row['width_height_pixel'][j]-rows[i-1]['width_height_pixel'][j] for j in range(2)] if i and row['width_height_pixel'] is not None and rows[i-1].get('width_height_pixel') is not None else None
        if i:
            d=np.abs(data[i].astype(float)-data[i-1]);row['background_pair_absolute_mean']=float(d[region].mean());row['background_pair_absolute_max']=float(d[region].max())
        if ref is not None:row['background_vs_off_absolute_mean']=float(np.abs(data[i].astype(float)-ref[i])[region].mean())
        if not row['eligible']:row['evaluation_status']='UNDETERMINED_REFERENCE_OR_Q'
        else:row['evaluation_status']='DESCRIPTIVE_ONLY'
    (output/'all_rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    summary=dict(total_rows=31,pairs=30,eligible=sum(r['eligible'] for r in rows),q_valid=sum(r['valid'] for r in q),reference_missing=sum(r.get('p_pixel') is None for r in a),review_pending=sum(r.get('status')!='REVIEWED' for r in a),background_pixel_difference_not_displacement=True,subjective_radius_not_calibrated=True,status='REQUIRES_HUMAN_MULTI_LAYER_INTERPRETATION',science_denominator=0)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stream',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--off-stream');a=p.parse_args();print(evaluate(a.stream,a.annotations,a.output,a.off_stream))
