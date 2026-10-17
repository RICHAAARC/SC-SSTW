"""Build the fixed single-read Colab and its small ordinary source companion."""
import ast
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'paper_results_v1_single_read_v1'


def companion_files():
    names = [
        'main/__init__.py', 'main/tube_state/__init__.py',
        'main/tube_state/grow_video_reference.py',
        'main/tube_state/video_trajectory_temporal_edit_receiver_v1.py',
        'main/tube_state/segmented_payload_v1.py',
        'runtime/__init__.py', 'runtime/wan/__init__.py', 'runtime/wan/conditional_joint/__init__.py',
        'runtime/wan/conditional_joint/generation.py', 'runtime/wan/conditional_joint/vae.py',
        'experiments/__init__.py', 'experiments/paper_results_v1/__init__.py',
        'experiments/paper_results_v1/single_read_v1.py',
        'experiments/paper_results_v1/single_read_v1_cli.py',
        'experiments/paper_results_v1/single_read_v1_evaluation.py',
        'experiments/paper_results_v1/single_read_v1.plan.json',
    ]
    return [ROOT / name for name in names]


def cell(kind, source, identifier):
    value = dict(cell_type=kind, id=identifier, metadata={}, source=source.splitlines(True))
    if kind == 'code': value.update(execution_count=None, outputs=[])
    return value


def build():
    output = ROOT / 'notebooks'; output.mkdir(exist_ok=True)
    archive_path = output / (NAME + '_companion.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in companion_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    # Reuse successful P2 import repairs without its task phases or model downloads.
    original = json.loads((output / 'paper_results_v1_distinct_confirmation_p2_colab.ipynb').read_text())
    setup = ''.join(next(c for c in original['cells'] if c.get('id') == 'source-and-fixed-plan')['source'])
    node = next(n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_imports')
    repairs = ast.get_source_segment(setup, node)
    bootstrap = textwrap.dedent('''\
        import json, os, signal, subprocess, sys, urllib.request, uuid, zipfile
        from pathlib import Path
        OUTPUT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Single-Read-20261018/fixed-single-read')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        saved_names=('attempt.json','state.json','blind_plan.json','normalized.npy','votes.npz','read_result.json','row_result.json','summary.json')
        SAVED=any((OUTPUT/n).exists() or (OUTPUT/(n+'.tmp')).exists() for n in saved_names)
        if SAVED:
            print('已有任务记录：本次Run all仅展示已保存结果；不继续、不重试任何调用。')
        else:
            URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_single_read_v1_companion.zip'
            ARCHIVE=OUTPUT/'companion_source.zip'
            CODE=Path('/content')/('single-read-v1-'+uuid.uuid4().hex[:10]);CODE.mkdir()
            def readable_archive(path):
                with zipfile.ZipFile(path) as archive:
                    archive.read('experiments/paper_results_v1/single_read_v1_cli.py')
                    for member in archive.infolist():
                        target=(CODE/member.filename).resolve()
                        if target!=CODE.resolve() and CODE.resolve() not in target.parents:
                            raise ValueError('unsafe source archive path')
            try:
                pending=ARCHIVE.with_suffix('.download.zip')
                with urllib.request.urlopen(URL,timeout=60) as response:pending.write_bytes(response.read())
                readable_archive(pending);pending.replace(ARCHIVE)
            except Exception as exc:
                print('Source download unavailable:',type(exc).__name__,str(exc))
                readable_archive(ARCHIVE)
                print('Using available readable source already in this output directory.')
            with zipfile.ZipFile(ARCHIVE) as archive:archive.extractall(CODE)
            ENV={**os.environ,'PYTHONPATH':str(CODE),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            LOG=OUTPUT/'execution.log'
            CLI=[sys.executable,'-u','-m','experiments.paper_results_v1.single_read_v1_cli']
            def command(args,label,check=True):
                print(label,flush=True)
                with LOG.open('a') as stream:
                    stream.write('COMMAND '+repr(args)+'\\n');stream.flush()
                    child=subprocess.Popen(args,cwd=CODE,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=child.wait()
                    except BaseException:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise
                if code and check:raise RuntimeError(label+' failed; see '+str(LOG))
                return code
    ''')
    environment = repairs + '\n\n' + textwrap.dedent('''\
        if not SAVED:
            ensure_imports('import numpy',['numpy'],'array dependency')
            ensure_imports('import torch',['torch'],'tensor dependency')
            ensure_imports('import safetensors,huggingface_hub; from diffusers import AutoencoderKLWan',
                           ['diffusers','accelerate','safetensors','huggingface-hub'],'Wan VAE import only')
            print('仅使用同Drive已有Wan VAE；没有checkpoint下载。')
    ''')
    run = textwrap.dedent('''\
        if not SAVED:
            command([*CLI,'--output',str(OUTPUT)],'一次固定单行补读',check=False)
        # No model/array imports, truth joins or re-evaluation on a later Run all.
        for name in ('summary.json','state.json','read_result.json','attempt.json'):
            path=OUTPUT/name
            if path.exists():
                try:print(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2))
                except (OSError,ValueError) as exc:print('保存记录不可读:',path,type(exc).__name__,str(exc))
                break
        else:print('任务尚未开始；请查看前面具体的环境或源码错误。')
        print('完整回交目录:',OUTPUT)
        print('STARTED而没有COMPLETE表示运行中或中断/结果未完成；再次Run all不会继续。')
    ''')
    intro = '''# A 单行补读：只补后验 RAW 候选的16未知位

在 GPU Colab 中直接 **Run all**。固定源 `distinct_confirm_20261017_01` / `SHORT17_HOLD_MEDIA` / `DISTINCT_RAW_U`；复用已保存盲计划，35帧原收到RGB映射为17帧。固定4段32位=16待补存在+16结构缺失，原路径误差1保留。

最多1次已有FP32 Wan VAE加载、1次17帧编码、1次原分段读取/FFT/新票归约。只选择同次received.npy或POST原始RGB一个容器；选定打开无效就保留失败，不换容器重试。旧025完整映射不同，不能借用；本包不搜索或拼接其它latent。已有模型缓存可采用可用子目录，不要求固定版本或目录名。

调用前持久记录尝试。一旦任务开始，失败、中断或保存不完整都消费该次机会；以后Run all只读取保存记录，不重新加载、编码、读取，也不自动续做尚未开始的后半链。不要删除输出目录或尝试记录来重跑。同一Drive目录保存全部结果。

新读数先保存，再在独立评价阶段接入消息；不重新读取旧票、评分、选择路径、codec、生成或梯度，不下载checkpoint，不改变默认CENTERED。仅补后验候选16位证据缺口，不构成独立确认或预设0错。已完成CPU/stub/static工程检查；真实GPU、模型、Drive与Colab执行仍由用户完成。源码ZIP直接下载可用dev源码，无手动上传、摘要、版本或GPU型号门禁。
'''
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        cells=[cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
               cell('markdown',intro,'scope'),cell('code',bootstrap,'source-or-saved'),
               cell('markdown','## 准备本次所需依赖\n','dependencies-title'),cell('code',environment,'dependencies'),
               cell('markdown','## 执行一次并展示保存结果\n','run-title'),cell('code',run,'run-and-display')])
    path = output / (NAME + '_colab.ipynb')
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    return dict(notebook=str(path), companion=str(archive_path), files=len(companion_files()))


if __name__ == '__main__': print(json.dumps(build(), indent=2))
