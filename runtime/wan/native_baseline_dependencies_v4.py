"""Actual import repair reused from the validated public Baseline setup."""
import os, json, shutil, subprocess, sys
from pathlib import Path

def _install_dependencies(source, vendor):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(source), str(vendor / "videoseal"), str(vendor / "watermark-anything")])
    mapping = dict(numpy="numpy", torch="torch", torchvision="torchvision", omegaconf="omegaconf",
                   pandas="pandas", einops="einops", lpips="lpips", timm="timm", pycocotools="pycocotools",
                   tqdm="tqdm", pytorch_msssim="pytorch-msssim", tensorboard="tensorboard", calflops="calflops",
                   transformers="transformers", safetensors="safetensors", scipy="scipy", av="av",
                   huggingface_hub="huggingface-hub", ftfy="ftfy", sentencepiece="sentencepiece",
                   accelerate="accelerate", cv2="opencv-python-headless", pywt="PyWavelets",
                   skimage="scikit-image", sklearn="scikit-learn", matplotlib="matplotlib", PIL="pillow",
                   yaml="PyYAML", ffmpeg="ffmpeg-python", requests="requests", decord="decord")
    probe = """import json
try:
 import numpy, torch, torchvision, videoseal.utils.cfg
 import watermark_anything.models, watermark_anything.data.metrics
 import importlib.util
 from pathlib import Path
 p=Path(__import__('sys').argv[1])/'watermark-anything/notebooks/inference_utils.py'
 s=importlib.util.spec_from_file_location('wam_import_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 print(json.dumps({'ok':True}))
except ModuleNotFoundError as e:
 print(json.dumps({'missing':e.name}));raise SystemExit(3)
"""
    installed = set()
    for _ in range(len(mapping)+1):
        checked = subprocess.run([sys.executable, "-c", probe, str(vendor)], cwd=vendor / "videoseal",
                                 env=environment, capture_output=True, text=True)
        if checked.returncode == 0:
            break
        if checked.returncode != 3:
            raise RuntimeError("Actual dependency import failed: " + checked.stderr[-6000:])
        missing = json.loads(checked.stdout.splitlines()[-1])["missing"].split(".")[0]
        if missing not in mapping or missing in installed:
            raise RuntimeError("Unresolved dependency import: " + missing + " " + checked.stderr[-2000:])
        subprocess.run([sys.executable, "-m", "pip", "install", mapping[missing]], check=True)
        installed.add(missing)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
    return environment
