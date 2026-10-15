"""Build the adopted fixed four-source Colab handoff and ordinary source ZIP."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import zipfile

from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/paper_results_v1_independent_four_v1_colab.ipynb'
ZIP = ROOT / 'notebooks/paper_results_v1_independent_four_v1_companion.zip'


def companion_files():
    paths = [ROOT / f'{name}/__init__.py' for name in ('main', 'runtime', 'experiments')]
    for folder in ('main/tube_state', 'runtime/wan', 'experiments/paper_results_v1'):
        paths.extend((ROOT / folder).rglob('*.py'))
    paths.append(ROOT / 'experiments/paper_results_v1/independent_four_v1.adopted.json')
    return sorted(set(paths))


def build():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in companion_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    setup = textwrap.dedent('''\
        import json, os, signal, subprocess, sys, time, urllib.request, uuid, zipfile
        from pathlib import Path

        OUTPUT_ROOT = Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1')
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        ARCHIVE = OUTPUT_ROOT / 'companion_source.zip'
        ZIP_URL = 'https://github.com/RICHAAARC/SC-SSTW/raw/refs/heads/dev/paper-results-v1/notebooks/paper_results_v1_independent_four_v1_companion.zip'
        # An uploaded readable source ZIP can be used before/without Git publication.
        download = ARCHIVE.with_suffix('.download')
        try:
            urllib.request.urlretrieve(ZIP_URL, download)
            with zipfile.ZipFile(download) as probe:
                probe.getinfo('experiments/paper_results_v1/independent_four_v1_cli.py')
            os.replace(download, ARCHIVE)
        except Exception as exc:
            if not ARCHIVE.is_file():
                raise FileNotFoundError('Place the delivered companion ZIP at ' + str(ARCHIVE) + '; source download failed: ' + str(exc))
            print('Using readable saved companion ZIP; download unavailable:', type(exc).__name__, str(exc))

        PORTABLE_ROOT = Path('/content') / ('independent-four-source-' + uuid.uuid4().hex[:10])
        PORTABLE_ROOT.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(ARCHIVE) as archive:
            for member in archive.infolist():
                target = (PORTABLE_ROOT / member.filename).resolve()
                if target != PORTABLE_ROOT.resolve() and PORTABLE_ROOT.resolve() not in target.parents:
                    raise ValueError('unsafe source archive path')
            archive.extractall(PORTABLE_ROOT)
        ENV = {**os.environ, 'PYTHONPATH': str(PORTABLE_ROOT)}
        CLI = [sys.executable, '-u', '-m', 'experiments.paper_results_v1.independent_four_v1_cli']
        LOG = OUTPUT_ROOT / 'notebook_setup.log'
        def command(args, label, check=True):
            print(label, flush=True)
            with LOG.open('a') as stream:
                stream.write('COMMAND ' + repr(args) + '\\n'); stream.flush()
                child = subprocess.Popen(args, cwd=PORTABLE_ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait()
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise
            if code and check: raise RuntimeError(label + ' failed; see ' + str(LOG))
            return code
        command([*CLI, '--help'], 'Check readable CLI')
        config = json.loads((PORTABLE_ROOT / 'experiments/paper_results_v1/independent_four_v1.adopted.json').read_text())
        CACHE = Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models')
        for name, folder in [('wan', 'Wan2.1-T2V-1.3B-Diffusers'), ('framewise', 'sd-vae-ft-mse')]:
            base = CACHE / folder
            candidates = [base / str(config['models'][name].get('revision', 'available')), base / 'available']
            if base.is_dir(): candidates += [p for p in base.iterdir() if p.is_dir()]
            if name == 'wan':
                available = next((p for p in candidates if (p / 'model_index.json').is_file() or (p / 'vae/config.json').is_file()), None)
            else:
                available = next((p for p in candidates if (p / 'config.json').is_file()), None)
            available = available or base / 'available'
            config['models'][name]['local_snapshot_path'] = str(available)
        # The runner reads prior effective method fields before replacing them.
        CONFIG = OUTPUT_ROOT / 'notebook_requested_config.json'
        CONFIG.write_text(json.dumps(config, indent=2) + '\\n')
        command([*CLI, '--output', str(OUTPUT_ROOT), '--config', str(CONFIG), '--phase', 'init'], 'Initialize fixed rows')
        summary = json.loads((OUTPUT_ROOT / 'summary.json').read_text())
        PENDING_STAGES = summary['pending_stages']
        print('Four fresh sources; 24 receiver rows / 768 bits; 24 quality pairs. Pending stages:', len(PENDING_STAGES))
    ''')
    environment = textwrap.dedent('''\
        if PENDING_STAGES:
            requirements = summary['setup_requirements']
            def ensure_imports(probe, packages, label):
                if command([sys.executable, '-c', probe], 'Probe ' + label, check=False):
                    command([sys.executable, '-m', 'pip', 'install', '--upgrade', *packages], 'Repair ' + label)
                    command([sys.executable, '-c', probe], 'Recheck ' + label)
            base = [(name, package) for name, package, flag in
                    [('numpy', 'numpy', 'numpy'), ('PIL', 'pillow', 'pillow'), ('torch', 'torch', 'torch')]
                    if requirements[flag]]
            if base:
                ensure_imports('import ' + ','.join(name for name, _ in base),
                               [package for _, package in base], 'remaining CPU/tensor dependencies')
            # Recheck readable cached latents after their CPU libraries are available.
            # This may remove a model requirement; it never calls a model.
            command([*CLI, '--output', str(OUTPUT_ROOT), '--config', str(CONFIG), '--phase', 'report'], 'Refresh remaining dependencies')
            requirements = json.loads((OUTPUT_ROOT / 'summary.json').read_text())['setup_requirements']
            classes = []
            if requirements['wan_full']: classes.append('WanPipeline')
            if requirements['wan_vae']: classes.append('AutoencoderKLWan')
            if requirements['framewise']: classes.append('AutoencoderKL')
            if classes:
                probe = 'import safetensors,huggingface_hub; from diffusers import ' + ','.join(classes)
                packages = ['diffusers', 'accelerate', 'safetensors', 'huggingface-hub']
                if requirements['wan_full']:
                    probe += '; import transformers'
                    packages += ['transformers', 'sentencepiece', 'ftfy']
                ensure_imports(probe, packages, 'remaining model imports')
            import shutil
            if requirements['ffmpeg'] and (not shutil.which('ffmpeg') or not shutil.which('ffprobe')):
                command(['apt-get', '-qq', 'update'], 'Prepare media dependency')
                command(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], 'Install FFmpeg')
            targets = []
            if requirements['wan_full'] or requirements['wan_vae']: targets.append('wan')
            if requirements['framewise']: targets.append('framewise')
            for name in targets:
                model = config['models'][name]
                local = Path(model['local_snapshot_path'])
                if name == 'wan' and requirements['wan_full']:
                    present = (local / 'model_index.json').is_file() and all((local / sub).is_dir() for sub in ('transformer', 'text_encoder', 'tokenizer', 'scheduler', 'vae'))
                    present = present and all(list((local / sub).glob('*.safetensors')) for sub in ('transformer', 'text_encoder', 'vae'))
                    patterns = None
                elif name == 'wan':
                    present = (local / 'vae/config.json').is_file() and bool(list((local / 'vae').glob('*.safetensors')))
                    patterns = ['model_index.json', 'vae/*']
                else:
                    present = (local / 'config.json').is_file() and (local / 'diffusion_pytorch_model.safetensors').is_file()
                    patterns = ['config.json', '*.safetensors']
                if not present:
                    script = 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=' + repr(model['upstream_id']) + ', local_dir=' + repr(str(local)) + ', allow_patterns=' + repr(patterns) + ')'
                    command([sys.executable, '-c', script], 'Prepare ' + name + ' model files')
        else:
            print('All fixed units completed or scientific terminal; no model setup required.')
    ''')
    execute = textwrap.dedent('''\
        try:
            command([*CLI, '--output', str(OUTPUT_ROOT), '--config', str(CONFIG), '--phase', 'all'], 'Run fixed independent four-source task')
        finally:
            command([*CLI, '--output', str(OUTPUT_ROOT), '--config', str(CONFIG), '--phase', 'report'], 'Save final or interrupted report', check=False)
        summary = json.loads((OUTPUT_ROOT / 'summary.json').read_text())
        print(json.dumps(summary, indent=2))
        print('Return the full task directory:', OUTPUT_ROOT)
    ''')
    intro = '''# 独立四新源：固定生成、12次媒体往返与联合接收

在 GPU Colab 中 **Run all**。四条已采用 prompt/seed 全部新生成，OFF/PAYLOAD 每源共用初始噪声；本批 K0 八行先新算，再汇总到 24 条接收记录。输出使用同一个 Drive 任务目录，续跑跳过完成单元及已观测的不支持/未解析结果。具体导入、下载或执行故障修复后，再次 Run all 会对未完成技术单元追加一次尝试；同一次 Run all 不循环重试。旧失败、中断和追加调用均保留，名义预算不替代累计实测。

只编码 OFF_NATIVE、PAYLOAD_NATIVE 参考、残差候选各四次，共12次 encode/probe/readback。RECON/M05仅保留PRE，固定delete_g0不二次编码。名义预算：800 DiT前向、100局部FFT求导（0 DiT/VAE VJP）、16模型加载、接收16framewise clips/352batch8、24selectors、最多24Wan encode/read。逐次开始/返回/失败与累计/本次账本独立保留，未知内部chunk/内存/时长不当实测。

固定24 receiver/768 bits和24 quality pairs，所有失败/缺失/不支持保留，不筛选或换源。盲选择只用received RGB、key与公开协议；truth只在之后评价。NATIVE_PRE/POST是PAYLOAD_NATIVE参考，质量是残差时钟增量，不是相对OFF的总水印画质。无质量/presence阈值，无FPR/泛化/科学PASS结论。

源码包是普通ZIP，不使用B64、摘要、manifest或版本相等门禁。发布前可将随本notebook交付的ZIP复制为固定输出目录下的 `companion_source.zip`；发布后可直接下载。60秒checkpoint保存最后已提交状态，完整单元即时落盘；尚未声称真实跨Colab恢复已测。

依赖准备按真实未完成单元决定：仅质量/残差时准备CPU依赖，仅媒体往返时准备FFmpeg；生成需要完整Wan，仅VAE encode/decode则只准备Wan VAE，framewise模型也仅在实际需要时准备。空的阶段wrapper不触发模型下载。
'''
    nb = dict(nbformat=4, nbformat_minor=5,
        metadata={'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
        cells=[cell('code', "from google.colab import drive\ndrive.mount('/content/drive')\n", 'drive-mount'),
               cell('markdown', intro, 'scope'), cell('code', setup, 'source-and-fixed-task'),
               cell('markdown', '## 准备实际依赖和模型文件\n', 'setup-title'),
               cell('code', environment, 'environment'), cell('markdown', '## 执行与完整回交\n', 'run-title'),
               cell('code', execute, 'run-and-report')])
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return dict(notebook=str(NOTEBOOK), companion=str(ZIP), files=len(companion_files()))


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
