"""Stdlib Colab bootstrap; reserve before full-source/dependency preparation."""
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import types
import urllib.request
import zipfile

REPOSITORY = 'RICHAAARC/SC-SSTW'
SOURCE_REF = 'dev/local-joint-state-payload-v1'
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Relative-Time-Payload-CDC32-V1/fixed-r01-v1/attempt01')
SOURCE = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Relative-Consistency-V1/fresh-r01-v1/attempt01'
WORK = Path('/content/video-wm-relative-payload-v1')


def source_text(path, source_directory, source_ref):
    if source_directory: return (Path(source_directory)/path).read_text()
    return urllib.request.urlopen(f'https://raw.githubusercontent.com/{REPOSITORY}/{source_ref}/{path}', timeout=120).read().decode('utf-8')


def safe_extract(archive, destination):
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = member.filename.replace('\\', '/')
            target = (destination/name).resolve()
            if name.startswith('/') or any(':' in part for part in name.split('/')) or not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('unsafe source archive path: '+name)
        zipped.extractall(destination)


def show_results(output=OUTPUT):
    output = Path(output)
    print('保留结果目录:', output)
    for name in ('result.json','counts.json','resources.json','quality.json','raw_terminal_summary.json',
                 'evaluation_summary.json','preparation_failure.json'):
        path = output/name
        if path.is_file():
            try: value = json.loads(path.read_text())
            except (OSError, ValueError) as error: value = dict(read_error=str(error))
            print(name, json.dumps(value, ensure_ascii=False, indent=2))
    print('完整逐槽/逐引用记录: post/、plans/、local/、final/；原始投影及两key缓存: bank/。')


def install_dependencies(source):
    import importlib.util
    setup_path = source/'scripts/b_framewise_qim_v1_setup.py'
    spec = importlib.util.spec_from_file_location('relative_payload_dependency_setup', setup_path)
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
    logs = OUTPUT/'preparation'; logs.mkdir(parents=True, exist_ok=True)
    setup.ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; '
        'from diffusers import WanPipeline,AutoencoderKLWan; import runtime.wan.b_relative_time_payload_backend_v1',
        ['numpy','torch','psutil','accelerate','safetensors','huggingface-hub','diffusers','transformers','sentencepiece','ftfy'],
        'actual E41 payload imports', source, logs/'setup.log')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        setup.command(['apt-get','-qq','update'], 'Prepare media dependency', source, logs/'setup.log')
        setup.command(['apt-get','-qq','install','-y','ffmpeg'], 'Install media dependency', source, logs/'setup.log')


def run_colab(source_directory=None, source_ref=SOURCE_REF):
    # Reopening never imports scientific modules or re-runs preparation/post.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        show_results(); return dict(status='READ_ONLY', output=str(OUTPUT))
    records_path = 'runtime/wan/b_relative_time_payload_records_v1.py'
    records = types.ModuleType('standalone_relative_payload_records')
    exec(compile(source_text(records_path, source_directory, source_ref), records_path, 'exec'), records.__dict__)
    config = dict(model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers', source=SOURCE, output=str(OUTPUT))
    if not records.reserve(OUTPUT, config):
        show_results(); return dict(status='READ_ONLY', output=str(OUTPUT))
    preparation_start = time.monotonic()
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        if source_directory:
            source = Path(source_directory).resolve()
        else:
            from urllib.parse import quote
            archive = WORK/'source.zip'
            urllib.request.urlretrieve(f'https://codeload.github.com/{REPOSITORY}/zip/refs/heads/{quote(source_ref,safe="/")}', archive)
            destination = WORK/'available_source'
            safe_extract(archive, destination)
            choices = [p for p in destination.iterdir() if (p/'runtime/wan/b_relative_time_payload_runner_v1.py').is_file()]
            if len(choices) != 1: raise ValueError('downloaded source must contain one usable project root')
            source = choices[0]
        sys.path.insert(0, str(source))
        os.environ.update(HF_HOME=str(WORK/'cache/huggingface'), TORCH_HOME=str(WORK/'cache/torch'))
        # Reuse the original import/install helper, without loading unrelated generation routes.
        install_dependencies(source)
        records.write(OUTPUT/'preparation.json', dict(status='RETURNED', seconds=time.monotonic()-preparation_start,
                                                      source_directory=str(source), source_ref=source_ref))
        from runtime.wan.b_relative_time_payload_runner_v1 import execute_reserved
        result = execute_reserved(OUTPUT)
    except BaseException as error:
        result = dict(status='STOPPED_FIXED_ATTEMPT', failure=f'{type(error).__name__}: {error}',
                      preparation_seconds=time.monotonic()-preparation_start, consumed=True, scientific_pass=None)
        records.write(OUTPUT/'preparation_failure.json', result)
        if not (OUTPUT/'blind_terminal.json').exists(): records.finalize_blind(OUTPUT, result['failure'])
        records.write(OUTPUT/'result.json', result)
    show_results()
    return dict(result, output=str(OUTPUT))
