"""Import-driven dependency preparation, no model construction or media work."""
from pathlib import Path
import importlib.util


def setup_dependencies(root,logs):
    root,logs=Path(root),Path(logs);logs.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('joint_a_import_setup',root/'scripts/b_framewise_qim_v1_setup.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan',
                      ['numpy','torch','psutil','accelerate','safetensors','huggingface-hub','diffusers'],
                      'joint A VAE dependencies',root,logs/'setup.log')
    old.ensure_imports('import runtime.wan.b_joint_time_packet_a_native_v1; import experiments.wan_state_clock.b_joint_time_packet_a_gpu_v1',
                      ['transformers','sentencepiece','ftfy','diffusers','accelerate','safetensors','huggingface-hub','numpy','torch','psutil'],
                      'actual joint A GPU imports',root,logs/'setup.log')
