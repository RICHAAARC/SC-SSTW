"""Package actual new-route source; no model, media, old carrier or remote dependency."""
from pathlib import Path
import zipfile,argparse,ast,json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(a.output)
 files=list((root/'main/sc_sstw').glob('*.py'))+[root/x for x in ('runtime/public_statistic/phase1.py','experiments/public_statistic/run_phase1.py','experiments/public_statistic/build_source_package.py','configs/public_luma_phase1.json','tests/test_public_statistic.py','docs/public_statistic/README.md','notebooks/public_statistic_phase1_cpu.ipynb')]
 for f in files:
  if f.suffix=='.py':ast.parse(f.read_text())
 nb=json.loads((root/'notebooks/public_statistic_phase1_cpu.ipynb').read_text())
 for c in nb['cells']:
  if c['cell_type']=='code':ast.parse(''.join(c['source']))
 assert ''.join(nb['cells'][0]['source']).splitlines()==['from google.colab import drive',"drive.mount('/content/drive')"]
 out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(f,f.relative_to(root))
 with zipfile.ZipFile(out) as z:
  assert all(z.read(str(f.relative_to(root)))==f.read_bytes() for f in files)
 print(json.dumps(dict(files=len(files),zip=str(out),ast='passed',notebook_media_execution='NOT_RUN')))
if __name__=='__main__':main()
