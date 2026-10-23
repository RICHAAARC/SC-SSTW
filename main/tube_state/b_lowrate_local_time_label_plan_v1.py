"""Fixed public low-rate local-tag protocol. Private preparation origins live in runtime."""
HOSTS=('b101-s01','b101-s03','b101-s02','b101-s04')
ARMS=('OFF','P0','TAG')
NEW_ARMS=('P0','TAG')
KEYS={'key00':'watermark','key01':'watermark-wrong'}
SOURCE='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Lowrate-Local-Time-Label-V1/fixed-viewed-development-v1'
MODEL='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
CRF=23

def tag_tables():
    return [('native',h,a,-1,k) for h in HOSTS for a in ARMS for k in KEYS]+[('mp4',h,a,-1,k) for h in HOSTS for a in NEW_ARMS for k in KEYS]+[('local',h,a,p,k) for h in HOSTS for a in ARMS for p in range(8) for k in KEYS]
def row_indices(view):return list(range(1,5)) if view=='local' else list(range(1,22))
def families():return [(h,a,k) for h in HOSTS for a in ARMS for k in KEYS]
def payload_tables():return [('mp4',h,a,-1,k) for h in HOSTS for a in NEW_ARMS for k in KEYS]+[('local',h,a,p,'key00') for h in HOSTS for a in NEW_ARMS for p in range(8)]
def fixed():return dict(native_rows=504,mp4_rows=336,local_rows=768,raw_rows=1608,correlations=77184,families=24,final_local_slots=96,local_nominal_FULL_positive=16,local_MIXED_positive_material=112,local_null_rows=640,payload_tables=80,full_payload_group_pairs=128,full_payload_cell_pairs=1408,local_payload_pairs=2816,local_payload_FULL_pairs=704,local_payload_MIXED_pairs=2112,quality_pairs=16,quality_frames=2896,quality_adjacent=2880,clipping=8)
