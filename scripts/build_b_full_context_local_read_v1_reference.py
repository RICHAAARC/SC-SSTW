"""Copy audited saved fields into ordinary gzip JSON; no scientific recomputation."""
from pathlib import Path
import argparse,gzip,json
ROOT=Path(__file__).resolve().parents[1]
NAME='b_full_context_local_read_v1_reference.json.gz'
RUN=ROOT.parents[1]/'diagnostics/b-line-time-label-transmission-attribution-diagnostic-20261023/run'
OLD=ROOT.parents[1]/'diagnostics/b-line-time-label-local-channel-result-20261023/saved'
def select(row,fields):return {k:row[k] for k in fields if k in row}
def build(run=RUN,old=OLD,output=None):
    run,old=Path(run),Path(old)
    read=lambda p:json.loads(p.read_text())
    native=[select(x,('host','row','saved_unit','input_issues')) for x in read(run/'blind_native_packets.json')]
    refs=[]
    for x in read(run/'native_reference_cache.json'):
        refs.append(dict(**select(x,('host','cell','chip','q_N','native_saved_projection','projection_reference_status','epsilon_N','pilot_soft')),
            reference_record=select(x['reference_record'],('chip','ROI','channel','bit','k','target_projection','q_after_FP32','projection_FP32_error'))))
    local=[select(x,('host','view','phase','row','saved_unit','input_issues')) for x in read(run/'blind_local_packets.json') if x['view'].startswith('clean')]
    joint=[select(x,('host','role','view','phase','row','source_cell','source_frames','stored_true_distance','stored_nearest_distance','saved_final_state')) for x in read(run/'pilot_RM_joint_rows.json') if x['view'].startswith('clean')]
    chips=[select(x,('host','view','phase','row','chip','bit_error','D','epsilon_L','w','signed_same_parity_margin','nearest_half_integer_boundary_distance','pilot_soft')) for x in read(run/'chip_diagnostics.json') if x['view'].startswith('clean')]
    null=[]
    for x in read(run/'null_saved_states.json'):
        null.append(dict(coordinate=x['coordinate'],status=x['status'],saved_state=select(x.get('saved_state') or {},
            ('role','status','pilot_P','pilot_H','pilot_detected','code_status','nearest_distance','code_id_candidate','tag_label','technical'))))
    value=dict(format='saved-field-copy-v1',native_packets=native,native_reference=refs,local_packets=local,local_rows=joint,local_chips=chips,
        null_saved_states=null,thresholds=read(old/'thresholds.json'),protocol=read(old/'protocol.json'),
        meaning='Copied audited saved fields only; no cos/distance/projection/decoder execution. Post-only reference; do not parse before new full blind save.')
    output=Path(output or ROOT/'data'/NAME);output.parent.mkdir(exist_ok=True,parents=True)
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    with gzip.open(output,'wb') as f:f.write(raw)
    return dict(path=str(output),JSON_bytes=len(raw),gzip_bytes=output.stat().st_size,
        native_rows=len(native),reference_chips=len(refs),local_rows=len(local),local_chip_records=len(chips),null_rows=len(null),
        scientific_recomputations=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior-run',default=str(RUN));p.add_argument('--prior-saved',default=str(OLD));p.add_argument('--output')
    a=p.parse_args();print(json.dumps(build(a.prior_run,a.prior_saved,a.output),indent=2))
