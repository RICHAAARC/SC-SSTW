"""Build the fixed user-run Uniform-Tanh-44/46 Colab notebook."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCATOR_SOURCE = """SOURCE_RUN = 'flow_tube_response_selection_20260921T013844126172Z'
# Drive parent chain and both cases were verified before notebook publication.
INPUT = DRIVE_ROOT / 'Video-WM' / 'FlowTubeResponseSelection' / SOURCE_RUN
print('Fixed input:', INPUT, flush=True)
required_names = ('generation.json', 'config.json', 'codebook.npz',
                  'OFF_nodes.pt', 'OFF_snapshots.pt', 'prompt.pt', 'negative.pt')
missing_inputs = [str(INPUT / case / name)
                  for case in ('dev_p0_s0', 'dev_p1_s0') for name in required_names
                  if not (INPUT / case / name).is_file()]
if missing_inputs:
    print('Missing fixed inputs (runner will record failures):', missing_inputs, flush=True)
else:
    print('Fixed inputs present: 14/14; runner will verify hashes.', flush=True)
"""


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch("[0-9a-f]{40}", source_commit):
            raise ValueError("full source SHA required")
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", source_commit + ":experiments/wan_state_clock/flow_tube_uniform_tanh_run.py"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", source_commit + ":runtime/wan/tube_uniform_tanh.py"], check=True)
    cells = []

    def cell(kind, name, text):
        row = {"cell_type": kind, "id": name, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            row.update(execution_count=None, outputs=[])
        cells.append(row)

    cell("code", "drive-mount", "from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell("markdown", "scope", """# Uniform-Tanh-44/46 固定开发集机制比较（用户 Run all）

固定两个原开发内容/seed，每例 `OFF / SINGLE46_A/B / MULTI44_46_A/B`，共10个逻辑视频。复用成功 response-selection run 中逐文件 hash 验证的 saved44 `z/v`、完整 UniPC history、prompt/negative 和 codebook；原模型 revision 未记录，因此历史权重身份未知。本轮固定加载已成功配置 revision `0fad780a534b6463e45facd96134c9f345acfa5b`，同批重建 OFF44–49。

唯一目标为温度1 tanh，原 carrier/key/A/B 与原 blind receiver 不变。固定 `R*=0.042943312697648145`，其来源是四个完整历史 dev LOCAL native-response 预算的最小值；这是历史 future-LAST oracle provenance，但新运行不读取任何未来 LAST 预算。SINGLE46 在46使用 `R*`；MULTI 在44和46各使用 `R*/2`。预算是每步 same-history controlled-minus-zero native response RMS 的累计和；它不表示等能量或等终态位移。

MULTI 的46控制在44控制和自由45之后的真实 latent 与完整 scheduler history 上，重新计算 CPU detached clean46 tanh gradient、same-history zero shadow 与 unit response。47–49自由采样，无末步补写、无 LAST、无校准名单、无 partial receiver。全轮固定84 Transformer forwards、52 formal native steps、12 unit probes、4 second-control zero shadows、12 CPU clean-leaf backwards、10 decode/save、120 receiver encodes。所有失败保留10视频、30媒体层、120相位槽分母。

每个控制步保存控制前后、unit response、realized native D、累计 RMS/平方和；保存终态相对同批 OFF。每个终态以及 floatRGB、RGB8、真实 MP4 三层保留 correct/wrong absolute score 与 gap。这里只支持固定开发集机制比较，不支持等能量、存在阈值、FPR、感知质量或推广结论。实现仅做过 CPU/fake 验证；本 notebook 的 GPU/真实模型运行由用户发起。

源码SHA：""" + str(source_commit) + "。\n")
    cell("code", "source-and-software", f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source commit: rebuild with the full source SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_uniform_tanh_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None
if version('torch') != '2.11.0+cu128':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers==0.40.0', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
subprocess.run([sys.executable, '-c', "import torch,diffusers; assert str(torch.__version__) == '2.11.0+cu128', torch.__version__; assert diffusers.__version__ == '0.40.0', diffusers.__version__"], check=True)
""")
    cell("code", "fixed-experiment", """DRIVE_ROOT = Path('/content/drive/MyDrive')
""" + LOCATOR_SOURCE + """OUTPUT = DRIVE_ROOT / 'Video-WM' / 'FlowTubeUniformTanh' / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
print('Output:', OUTPUT, 'Launcher log:', LOG, flush=True)
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_uniform_tanh_run', '--source', str(INPUT), '--output', str(OUTPUT)]
with LOG.open('w') as log:
    process = subprocess.Popen(command, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        except ProcessLookupError:
            pass
        raise
print('Input:', INPUT, 'Output:', OUTPUT, 'Launcher log:', LOG, 'Source archive:', ARCHIVE)
if (OUTPUT / 'result.json').exists():
    result = json.loads((OUTPUT / 'result.json').read_text())
    print(json.dumps({key: result.get(key) for key in ('status', 'video_denominator', 'media_layer_denominator', 'receiver_encode_denominator', 'fixed_calls', 'actual_calls_observed', 'budget', 'paired_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / "flow_tube_uniform_tanh_colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    print(build(parser.parse_args().source_commit))
