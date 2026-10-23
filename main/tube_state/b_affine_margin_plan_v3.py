"""Fixed s01 observation protocol, no truth consulted by inference."""
HOST='b101-s01'
ARMS=('P0','OLD','NEW')
KEYS={'key00':'watermark','key01':'watermark-wrong'}
ATTACKS=('crop104','delete13_s52')
WINDOWS=(0,26,52)
MODEL='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
P0='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1/hosts/b101-s01/private/P0_terminal.pt'
OLD='/content/drive/MyDrive/Video-WM/B-Lowrate-Local-Time-Label-V1/fixed-viewed-development-v1/native/b101-s01/TAG.pt'
TRUTH='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1/protocol.json'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Affine-Margin-Observation-V3/fixed-s01-once-v1'
def views():
 return [a+'/full' for a in ARMS]+[f'{a}/{attack}/w{w}/p{p}' for a in ARMS for attack in ATTACKS for w in range(3) for p in range(8)]
def families():return [f'{a}/{attack}/{k}' for a in ARMS for attack in ATTACKS for k in KEYS]
def fixed():return dict(hosts=1,arms=3,native_construct=1,VAE_load=1,D181=3,E181=3,E45=144,E_total=147,FFT=147,keyed_reads=294,codec_encode=9,codec_probe=9,codec_decode=9,sourceprep=0,DiT=0,source_gradient=0,raw_tag_rows=1278,tag_correlations=61344,R104_queries=6,R104_query_keys=12,logical_bits_each_variant=384,variants=4,logical_bits_total=1536,quality_records=6,quality_frames=778,quality_adjacent=772)
