"""Local follow-up input locations. Generating these does not start any stage."""
from pathlib import Path
import argparse
import json

VIEWS=('FULL','CROP','DELETE1','REPEAT1','DELETE3')


def build(project_root,delivery):
    root,delivery=Path(project_root).resolve(),Path(delivery).resolve();diag=root/'diagnostics'
    old_n01=diag/'n01-h-l-attribution-cpu-20261103/inputs/receiver/received'
    old_e02=diag/'e02-y-regression-c02-cache-implementation-20261102/run/fixed-e02-y-regression-v1/units'
    layer_raw={};time_raw={};old_time={}
    for source in ('n01','e02'):
        for arm in ('OLD','BASE'):
            for layer,original in zip(('preclip','clip','rgb8','firstRGB'),('preclip','clip','rgb8','h264')):
                if source=='n01':p=diag/'n01-training-cause-method-review-20261103/asset_meta'/f"{'B_QIM' if arm=='OLD' else 'A_BASE'}_{original}.npz"
                elif arm=='BASE' and layer=='preclip':p=delivery/'inputs/q0/e02_BASE_preclip.npz'
                else:p=delivery/'inputs/old/e02'/arm/(original+'.npz')
                layer_raw[f'{source}/{arm}/layer/{layer}']=str(p)
            for i,view in enumerate(VIEWS):
                if source=='e02':p=old_e02/f"v{i+(5 if arm=='BASE' else 0):02}"
                elif arm=='OLD':p=old_n01/view
                else:p=delivery/'inputs/old/n01/BASE'/view
                time_raw[f'{source}/{arm}/{view}']=str(p/'readout.npz')
                if arm=='OLD':old_time[source+'/'+view]={label:str(p/name) for label,name in [('H','Y_H.json'),('L','candidates.json'),('final','submission.json')]}
    media=dict(gpu1=str(delivery/'inputs/gpu1-run01'),layer_raw=layer_raw,time_raw=time_raw,
        e02_old_media={v:str(diag/'minimal-l-e02-local-confirmation-20261031/fixed-e02-ten-views-codecfix-v1/received'/f'v{i:02}'/'received.mp4') for i,v in enumerate(VIEWS)})
    post=dict(cpu_media=str(delivery/'cpu-media/run01'),gpu2=str(delivery/'inputs/gpu2-run01'),old_time=old_time,
              writer_protocol={s:str(delivery/'inputs/post'/(s+'_source_protocol.json')) for s in ('n01','e02')})
    delivery.mkdir(parents=True,exist_ok=True)
    for name,value in [('CPU_MEDIA_INPUT_CONFIG.json',media),('POST_INPUT_CONFIG.json',post)]:
        (delivery/name).write_text(json.dumps(value,indent=2)+'\n')
    return dict(status='INPUT_LOCATIONS_PREPARED',stages_executed=0,missing_is_preserved_not_refilled=True,
                cpu_media_output=str(delivery/'cpu-media/run01'),post_output=str(delivery/'post/run01'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--delivery',required=True)
    args=parser.parse_args();print(json.dumps(build(args.project_root,args.delivery),indent=2))
