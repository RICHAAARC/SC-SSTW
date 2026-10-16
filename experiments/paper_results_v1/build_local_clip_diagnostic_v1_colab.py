"""Build the user-run local-clip notebook and ordinary public companion ZIP."""
import json
from pathlib import Path
import textwrap
import zipfile
from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK=ROOT/'notebooks/paper_results_v1_local_clip_diagnostic_v1_colab.ipynb'
ZIP=ROOT/'notebooks/paper_results_v1_local_clip_diagnostic_v1_companion.zip'


def companion_files():
    paths=[ROOT/f'{name}/__init__.py' for name in ('main','runtime','experiments')]
    for folder in ('main/tube_state','runtime/wan','experiments/paper_results_v1'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.append(ROOT/'experiments/paper_results_v1/local_clip_diagnostic_v1.adopted.json')
    return sorted(set(paths))


def build():
    ZIP.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in companion_files():
            info=zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            archive.writestr(info,path.read_bytes())
    setup=textwrap.dedent('''\
        import json, os, signal, subprocess, sys, urllib.request, uuid, zipfile
        from pathlib import Path
        SOURCE_RUN=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1')
        OUTPUT_ROOT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Local-Clip-Diagnostic-V1/resumable-local-clip-diagnostic-v1')
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        ARCHIVE=OUTPUT_ROOT/'companion_source.zip'
        SOURCE_REF='dev/paper-results-v1'
        SOURCE_URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/notebooks/paper_results_v1_local_clip_diagnostic_v1_companion.zip'
        print('Load available local-clip source:',SOURCE_URL,flush=True)
        try:
            with urllib.request.urlopen(SOURCE_URL,timeout=60) as response:
                package_bytes=response.read()
            ARCHIVE.write_bytes(package_bytes)
        except Exception as exc:
            raise RuntimeError('local-clip source download/save failed: '+SOURCE_URL+'; '+repr(exc)) from exc
        PORTABLE_ROOT=Path('/content')/('local-clip-source-'+uuid.uuid4().hex[:10])
        PORTABLE_ROOT.mkdir(parents=True,exist_ok=False)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.getinfo('experiments/paper_results_v1/local_clip_diagnostic_v1_cli.py')
            for member in archive.infolist():
                target=(PORTABLE_ROOT/member.filename).resolve()
                if target != PORTABLE_ROOT.resolve() and PORTABLE_ROOT.resolve() not in target.parents:
                    raise ValueError('unsafe source archive path')
            archive.extractall(PORTABLE_ROOT)
        ENV={**os.environ,'PYTHONPATH':str(PORTABLE_ROOT)}
        CLI=[sys.executable,'-u','-m','experiments.paper_results_v1.local_clip_diagnostic_v1_cli']
        LOG=OUTPUT_ROOT/'execution.log'
        def command(args,label,check=True):
            print(label,flush=True)
            with LOG.open('a') as stream:
                stream.write('COMMAND '+repr(args)+'\\n');stream.flush()
                child=subprocess.Popen(args,cwd=PORTABLE_ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=child.wait()
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if code and check:raise RuntimeError(label+' failed; see '+str(LOG))
            return code
        CACHE=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers')
        candidates=[CACHE/'available']+([p for p in CACHE.iterdir() if p.is_dir()] if CACHE.is_dir() else [])
        MODEL=next((p for p in candidates if (p/'vae/config.json').is_file() and list((p/'vae').glob('*.safetensors'))),CACHE/'available')
        command([*CLI,'--output',str(OUTPUT_ROOT),'--source-run',str(SOURCE_RUN),'--model-path',str(MODEL),'--phase','init'],'Initialize fixed 24 rows')
        summary=json.loads((OUTPUT_ROOT/'summary.json').read_text())
    ''')
    prepare=textwrap.dedent('''\
        def ensure_imports(probe,packages,label):
            if command([sys.executable,'-c',probe],'Probe '+label,check=False):
                command([sys.executable,'-m','pip','install',*packages],'Prepare '+label)
                command([sys.executable,'-c',probe],'Recheck '+label)
        ensure_imports('import numpy',['numpy'],'saved numeric records')
        if summary['pending_technical_rows']:
            ensure_imports('import torch',['torch'],'RAW input tensors')
        command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','prepare'],'Select fixed local observations with RAW_U and CENTERED_U')
        summary=json.loads((OUTPUT_ROOT/'summary.json').read_text())
        print('Fixed 24 rows / 768 bits; pending technical rows:',summary['pending_technical_rows'], 'pending unique Wan inputs:',summary['pending_unique_wan_inputs'])
    ''')
    environment=textwrap.dedent('''\
        if summary['needs_wan_model']:
            ensure_imports('import safetensors,huggingface_hub; from diffusers import AutoencoderKLWan',
                           ['diffusers','accelerate','safetensors','huggingface-hub'],'Wan VAE only')
            if not ((MODEL/'vae/config.json').is_file() and list((MODEL/'vae').glob('*.safetensors'))):
                script='from huggingface_hub import snapshot_download; snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",local_dir='+repr(str(MODEL))+',allow_patterns=["model_index.json","vae/*"])'
                command([sys.executable,'-c',script],'Prepare readable Wan VAE files')
            import shutil
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                command(['apt-get','-qq','update'],'Prepare saved-media decoder dependency')
                command(['apt-get','-qq','install','-y','ffmpeg'],'Install FFmpeg')
        else:
            print('No pending clip input needs a model; completed records or this task own latents are reused.')
    ''')
    execute=textwrap.dedent('''\
        try:
            command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','all'],'Read fixed 24 local-clip logical rows')
        finally:
            command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','report'],'Retain complete/interrupted report',check=False)
        summary=json.loads((OUTPUT_ROOT/'summary.json').read_text())
        print(json.dumps(summary,indent=2))
        print('Return the complete task directory:',OUTPUT_ROOT)
    ''')
    intro='# 当前残差 A：中间片段、删除与重复的固定诊断\n\n在 GPU Colab 中直接 **Run all**。第一格挂载既有 Drive，随后从当前 dev/paper-results-v1 下载普通源码 ZIP，无需手动上传。同一 Drive 任务目录内续跑；已完成结果、本任务自身 latent 和精确同输入共享均持久保留。\n\n四个既有新源全部保留。每源固定取原源帧 **37–125**（零起点 `[37:126]`，89 帧，非四帧块边界起点），建立三种相关视图：原裁剪 89 帧；删除局部 `[0,11,23,35,47,59,71,88]` 后 81 帧；在同样位置的原帧后各追加一份后 97 帧。RAW_U 与 CENTERED_U 各 12 行/384 位，合计 **24 行/768 位**；独立源只有 4 个。\n\n复用四份 clean 时钟原始分数，先按视图取行，再对收到的局部矩阵重新计算列中位数和现有 U 路径。已有 score 不可读时，仅可由同次保存的 framewise latent 做 CPU 分数重算；两者都缺则保留 MISSING，不新增 framewise 模型。保存分数路径分析不冒充裁剪后实际 Wan 编码。\n\n沿用当前 writer、时钟、强度、key、固定181源时钟、visible-span 对齐和原32位reader。只对 supported 操作新增真实 Wan 输入读取；精确相同输入可共享物理编码，完整操作相同才能共享读取。无中断上限 **1 Wan VAE load、24 encode、24 read**。优先复用源 received RGB，缺失时每源最多解码一个既有 candidate MP4（共最多4次）；0 新视频编码、生成、writer、framewise编码或原 RAW8 重跑。实际数量由保存的盲操作和调用记录逐项报告，不以 R 相同代替输入相同。\n\n先保存固定24行的盲选择，再分别报告时间路径/区间、支持误差、不确定/不支持、实际载荷错误和缺失。front boundary copies、nearest填补、synthetic/dropped/repeated indices 和完整映射均保留；填补不代表找回丢失画面，重复帧/视图不增加独立证据。不预设 BER 收益，也不把一个起点/长度推广为任意片段鲁棒。\n\n每个完成单元即时保存，并保留周期 checkpoint；中断后挂载同一 Drive 再 Run all，只继续技术未完成单元，历史失败和额外调用累计保留。源码/摘要/版本/Git状态不设门禁。实际 GPU 结果尚待用户运行。\n'
    intro += '\n完成固定 prepare 遍历并保存选择或具体失败后才评价路径；固定 Wan 遍历结束且所有行均为终态后才评价消息。普通方法字段记录用于防止已知不同 crop/edit/key 等方法混用；旧可选记录缺失、不可读或损坏时记录原因并允许续跑，可选记录写入失败也仅保留具体警告。必要执行数据保存失败仍报告，中断仍传递。来源 schema 标签仅记录，实际解析检查所需字段、固定 key 和唯一 clean 行。\n'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
        cell('markdown',intro,'scope'),cell('code',setup,'source-and-fixed-task'),
        cell('markdown','## 复用观察并保存固定局部选择\n','prepare-title'),cell('code',prepare,'prepare'),
        cell('markdown','## 准备尚需的 Wan VAE\n','environment-title'),cell('code',environment,'environment'),
        cell('markdown','## 真实局部载荷读取与分层报告\n','run-title'),cell('code',execute,'run-and-report')])
    NOTEBOOK.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(NOTEBOOK),companion=str(ZIP),files=len(companion_files()))


if __name__=='__main__':print(json.dumps(build(),indent=2))
