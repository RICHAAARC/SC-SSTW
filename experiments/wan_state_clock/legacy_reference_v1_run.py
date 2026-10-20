"""CPU old-core public-reference signature control; no model or writer trace."""
import argparse,json
from pathlib import Path
from main.tube_state.legacy_reference_v1.reference import extract_public_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-latent',type=Path,required=True)
    parser.add_argument('--reference-id',required=True)
    parser.add_argument('--reference-shape',required=True,help='predeclared framewise T,C,H,W')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    shape=tuple(int(x) for x in args.reference_shape.split(','))
    config=dict(tubelet_length=4,tubelet_partition=dict(spatial_patch_size=[4,4]),
        embedding_margin=1.,score_calibration=dict(embedding_projection_support_weight=0.),
        sync_search=dict(offset_search_min=-8,offset_search_max=8,search_score_rule='penalized_prior',
            min_sync_alignment_coverage_ratio=.125,min_sync_alignment_matched_count=64,
            min_payload_rescue_gain=.01,min_aligned_payload_score=.095),lambda_sync=.01)
    value=extract_public_reference(args.input_latent,args.reference_id,shape,config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value['scores']))
    return 0
if __name__=='__main__':raise SystemExit(main())
