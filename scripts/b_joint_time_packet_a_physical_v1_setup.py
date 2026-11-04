"""Import-driven dependency preparation, no model construction or media work."""
from pathlib import Path
import importlib.util
import shutil


def setup_dependencies(root,logs,stage='gpu1'):
    root,logs=Path(root),Path(logs);logs.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('joint_a_import_setup',root/'scripts/b_framewise_qim_v1_setup.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    if stage in ('cpu_media','post'):
        old.ensure_imports('import numpy,torch,psutil; import experiments.wan_state_clock.b_joint_time_packet_a_'+
                          ('cpu_media' if stage=='cpu_media' else 'post')+'_v1'+
                          ('; import runtime.wan.b_joint_time_packet_a_media_v1' if stage=='cpu_media' else ''),
                          ['numpy','torch','psutil'],'actual joint A '+stage+' imports',root,logs/'setup.log')
        if stage=='cpu_media':
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                old.command(['apt-get','-qq','update'],'Prepare media dependency',root,logs/'setup.log')
                old.command(['apt-get','-qq','install','-y','ffmpeg'],'Install FFmpeg',root,logs/'setup.log')
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                raise RuntimeError('ffmpeg/ffprobe unavailable; see '+str(logs/'setup.log'))
        return
    old.ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan',
                      ['numpy','torch','psutil','accelerate','safetensors','huggingface-hub','diffusers'],
                      'joint A VAE dependencies',root,logs/'setup.log')
    old.ensure_imports('import runtime.wan.b_joint_time_packet_a_native_v1; import experiments.wan_state_clock.b_joint_time_packet_a_gpu_v1',
                      ['transformers','sentencepiece','ftfy','diffusers','accelerate','safetensors','huggingface-hub','numpy','torch','psutil'],
                      'actual joint A GPU imports',root,logs/'setup.log')
