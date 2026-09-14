"""Create an optional backup source ZIP; Colab uses GitHub by default."""
import ast
import argparse
import json
import zipfile
from pathlib import Path


def build(output):
    root = Path(__file__).resolve().parents[2]
    names = [str(p.relative_to(root)) for p in (root/'main/sc_sstw').glob('*.py')]
    names += ['runtime/public_statistic/phase2.py','runtime/public_statistic/phase2_execution.py','experiments/public_statistic/run_phase2.py','experiments/public_statistic/build_phase2_package.py','configs/public_luma_phase2_gpu.json','notebooks/public_statistic_phase2_gpu.ipynb','docs/public_statistic/phase2_executable.md']
    nb = json.loads((root/'notebooks/public_statistic_phase2_gpu.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source']).splitlines() == ['from google.colab import drive',"drive.mount('/content/drive')"]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
    for name in names:
        if name.endswith('.py'): ast.parse((root/name).read_text())
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(root/name, name)
    return dict(zip=str(out),files=len(names),notebook_syntax='PASS',media_execution='NOT_RUN')
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.output)))
