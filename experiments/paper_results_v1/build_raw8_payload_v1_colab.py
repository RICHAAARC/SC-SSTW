"""Build the user-run RAW8 notebook and ordinary public companion ZIP."""
import json
from pathlib import Path
import textwrap
import zipfile
from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK=ROOT/'notebooks/paper_results_v1_raw8_payload_v1_colab.ipynb'
ZIP=ROOT/'notebooks/paper_results_v1_raw8_payload_v1_companion.zip'


def companion_files():
    paths=[ROOT/f'{name}/__init__.py' for name in ('main','runtime','experiments')]
    for folder in ('main/tube_state','runtime/wan','experiments/paper_results_v1'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.append(ROOT/'experiments/paper_results_v1/raw8_saved_operations.json')
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
        OUTPUT_ROOT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-RAW8-Payload-V1/resumable-raw8-payload-v1')
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        ARCHIVE=OUTPUT_ROOT/'companion_source.zip'
        SOURCE_REF='dev/paper-results-v1'
        SOURCE_URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/notebooks/paper_results_v1_raw8_payload_v1_companion.zip'
        print('Load available RAW8 source:',SOURCE_URL,flush=True)
        try:
            with urllib.request.urlopen(SOURCE_URL,timeout=60) as response:
                package_bytes=response.read()
            ARCHIVE.write_bytes(package_bytes)
        except Exception as exc:
            raise RuntimeError('RAW8 source download/save failed: '+SOURCE_URL+'; '+repr(exc)) from exc
        PORTABLE_ROOT=Path('/content')/('raw8-source-'+uuid.uuid4().hex[:10])
        PORTABLE_ROOT.mkdir(parents=True,exist_ok=False)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.getinfo('experiments/paper_results_v1/raw8_payload_v1_cli.py')
            for member in archive.infolist():
                target=(PORTABLE_ROOT/member.filename).resolve()
                if target != PORTABLE_ROOT.resolve() and PORTABLE_ROOT.resolve() not in target.parents:
                    raise ValueError('unsafe source archive path')
            archive.extractall(PORTABLE_ROOT)
        ENV={**os.environ,'PYTHONPATH':str(PORTABLE_ROOT)}
        CLI=[sys.executable,'-u','-m','experiments.paper_results_v1.raw8_payload_v1_cli']
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
        command([*CLI,'--output',str(OUTPUT_ROOT),'--source-run',str(SOURCE_RUN),'--model-path',str(MODEL),'--phase','init'],'Initialize fixed eight rows')
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
        command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','prepare'],'Consume saved RAW operations and same-run media locators')
        summary=json.loads((OUTPUT_ROOT/'summary.json').read_text())
        print('Fixed 8 rows / 256 bits. Pending technical rows:',summary['pending_technical_rows'])
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
            print('No unresolved RAW encode needs a model; saved RAW latents or completed rows are reused.')
    ''')
    execute=textwrap.dedent('''\
        try:
            command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','all'],'Read fixed eight RAW_U inputs')
        finally:
            command([*CLI,'--output',str(OUTPUT_ROOT),'--phase','report'],'Retain complete/interrupted report',check=False)
        summary=json.loads((OUTPUT_ROOT/'summary.json').read_text())
        print(json.dumps(summary,indent=2))
        print('Return the complete task directory:',OUTPUT_ROOT)
    ''')
    intro='''# 新四源 RAW_U 实际载荷对照：固定八行

GPU Colab 中直接 **Run all**。挂载原有 Drive 后，自动从现有 dev/paper-results-v1 分支下载普通 companion ZIP，无需手动上传。每次加载可用源码，同一 Drive 目录内已经完成的 RAW 单元继续复用。

消费同次已保存候选视频与八个已保存 RAW_U 操作，不重新选择路径。固定四源 × CLEAN_MP4/DELETE_G0，8 行/256 位；七行 181 帧/R44，delete_g0/04 为 149 帧/R37。无中断预算：1 Wan VAE load、8 encode、8 read；缺 RGB 缓存时最多解码四个既有 MP4。同次解码复用于既定删帧。0 生成/写入/新 codec/framewise 模型/新 selector；不借用 CENTERED_U latent 或 votes。

原消息只在固定 RAW 遍历结束后用于配对报告。RAW 错而 centered 消除才支持本批实际载荷收益；两者都零则只支持路径改善和载荷保持。缺失、失败、不支持、歧义保持八行分母，不按结果换源/路径。

每个完整单元即时保存，60 秒 checkpoint 保存最后已提交状态。中断后同一 Drive Run all：复用本任务自己的 RAW latent/read，完成结果及科学终态不重跑；技术故障修复后对未完成单元追加一次尝试，旧失败与累计实际调用保留。源码、环境版本和摘要不作门禁。实际模型执行仍由用户进行。
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
        cell('markdown',intro,'scope'),cell('code',setup,'source-and-fixed-task'),
        cell('markdown','## 读取固定输入与已保存状态\n','prepare-title'),cell('code',prepare,'prepare'),
        cell('markdown','## 准备尚需的 Wan VAE\n','environment-title'),cell('code',environment,'environment'),
        cell('markdown','## 执行、保存与配对报告\n','run-title'),cell('code',execute,'run-and-report')])
    NOTEBOOK.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(NOTEBOOK),companion=str(ZIP),files=len(companion_files()))


if __name__=='__main__':print(json.dumps(build(),indent=2))
