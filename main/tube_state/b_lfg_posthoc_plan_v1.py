"""Fixed adopted stage1 public plan; stdlib only, no fixture truth."""
HOSTS=('b101-s01','b101-s03','b101-s02','b101-s04')
ARMS=('P0','MSG')
VIEWS=('native','rgb8_clean','mp4')
MEDIA_VIEWS=('rgb8_clean','mp4')
KEYS={'key00':'watermark','key01':'watermark-wrong'}
MODEL='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
SOURCE='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-LFG-Posthoc-Reference-V1/fixed-viewed-stage1-v1'
CRF=23
ALPHA=.15
N=512
M=235520
SHAPE=(1,16,46,40,64)
LLL_SHAPE=(16,23,20,32)

def tables():return [(v,h,a,k) for v in VIEWS for h in HOSTS for a in ARMS for k in KEYS]
def fixed():return dict(message_tables=48,raw_coded_bits=24576,information_bits=20736,positive_messages=12,positive_coded_bits=6144,positive_information_bits=5184,baseline_wrong_key_tables=36,payload_tables=32,payload_group_pairs=256,payload_cell_pairs=2816,quality_pairs=16,quality_frames=2896,quality_adjacent=2880,clipping=8)
