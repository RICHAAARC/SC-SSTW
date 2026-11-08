"""Passive native adapters. Importing this module never imports/loads a model."""
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def finite_tensor(value, shape=None):
    import torch
    if shape is not None and list(value.shape) != list(shape):
        raise ValueError(f"native output shape {list(value.shape)} != {list(shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("native model returned non-finite values")
    return value


def quantize(value):
    finite_tensor(value)
    return value.detach().clamp(0, 1).mul(255).round().to(__import__('torch').uint8).permute(0, 2, 3, 1).cpu().numpy()


class NativeAdapter:
    def __init__(self, method, vendor_root, device="cuda"):
        self.method = method
        self.vendor = Path(vendor_root)
        self.device = device
        self.model = None
        self.handles = []
        self.counts = {"embedder_calls": 0, "embedder_frames": 0,
                       "detector_calls": 0, "detector_frames": 0}

    def load(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for this user-run GPU batch")
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        repo = self.vendor / ("watermark-anything" if self.method == "wam" else "videoseal")
        sys.path.insert(0, str(repo))
        with working_directory(repo):
            if self.method == "pixelseal":
                import videoseal
                self.model = videoseal.load("pixelseal").float().eval().to(self.device)
                expected = dict(chunk_size=32, step_size=8, scaling_w=0.2, scaling_i=1.0,
                                img_size=256, video_mode="repeat", clamp=True)
                actual = {k: getattr(self.model, k, None) for k in expected}
                if actual != expected:
                    raise ValueError(f"PixelSeal actual method settings differ from first-batch spec: {actual}")
                details = dict(card="pixelseal", native_bits=256, parameters=actual,
                               attenuation=type(self.model.attenuation).__name__,
                               embed_antialias=True, extract_antialias=True,
                               weight_url="https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth")
                if self.model.attenuation is None:
                    raise ValueError("PixelSeal first batch requires native JND")
            elif self.method == "wam":
                checkpoint = repo / "checkpoints/wam_mit.pth"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise FileNotFoundError(f"Unreadable WAM weight: {checkpoint}")
                params = json.loads((repo / "checkpoints/params.json").read_text())
                expected = dict(nbits=32, img_size=256, img_size_extractor=256,
                                scaling_w=2.0, scaling_i=1.0, attenuation="jnd_1_3_blue")
                if any(params.get(k) != v for k, v in expected.items()):
                    raise ValueError("WAM native parameters differ from first-batch method settings")
                # Same official constructors, but do not inherit the demo helper's
                # missing-checkpoint print or swallowed JND construction exception.
                from omegaconf import OmegaConf
                from watermark_anything.models import Wam, build_embedder, build_extractor
                from watermark_anything.augmentation.augmenter import Augmenter
                from watermark_anything.modules.jnd import JND
                from watermark_anything.data.transforms import normalize_img, unnormalize_img
                from watermark_anything.data.metrics import msg_predict_inference
                embed_cfg = OmegaConf.load(params["embedder_config"])
                extract_cfg = OmegaConf.load(params["extractor_config"])
                augmentation_cfg = OmegaConf.load(params["augmentation_config"])
                attenuation_cfg = OmegaConf.load(params["attenuation_config"])
                embedder = build_embedder(params["embedder_model"], embed_cfg[params["embedder_model"]], params["nbits"])
                extractor = build_extractor(extract_cfg.model, extract_cfg[params["extractor_model"]], params["img_size"], params["nbits"])
                attenuation = JND(**attenuation_cfg[params["attenuation"]], preprocess=unnormalize_img, postprocess=normalize_img)
                self.model = Wam(embedder, extractor, Augmenter(**augmentation_cfg), attenuation,
                                 params["scaling_w"], params["scaling_i"])
                self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                self.model = self.model.float().eval().to(self.device)
                self.normalize = normalize_img
                self.unnormalize = unnormalize_img
                self.native_message = msg_predict_inference
                details = dict(native_bits=32, parameters=expected, image_video_extension=True,
                               native_postprocess="semihard: sigmoid(mask)>0.5; selected mean logits>0.5",
                               weight_url="https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth")
            else:
                raise ValueError("unsupported method; no automatic fallback")
        def hook(name):
            def observe(module, args):
                self.counts[name + "_calls"] += 1
                self.counts[name + "_frames"] += int(args[0].shape[0])
            return observe
        self.handles = [self.model.embedder.register_forward_pre_hook(hook("embedder")),
                        self.model.detector.register_forward_pre_hook(hook("detector"))]
        details.update(device=self.device, precision="FP32", inference_mode=True,
                       autocast=False, seconds=time.monotonic()-started)
        return details

    def tensor(self, rgb):
        import numpy as np
        import torch
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("native input must be uint8 THWC RGB")
        return torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().to(self.device) / 255

    def embed(self, rgb, message, rectangle):
        import torch
        x = self.tensor(rgb)
        msg = torch.tensor([message], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            if self.method == "wam":
                if len(rgb) != 1:
                    raise ValueError("WAM adapter uses one image per call")
                y = self.unnormalize(self.model.embed(self.normalize(x), msgs=msg)["imgs_w"])
                finite_tensor(y, x.shape)
                y0, x0, y1, x1 = rectangle
                composed = x.clone()
                composed[:, :, y0:y1, x0:x1] = y[:, :, y0:y1, x0:x1]
                y = composed
            else:
                y = self.model.embed(x, msgs=msg, is_video=True, lowres_attenuation=False,
                                     interpolation=dict(mode="bilinear", align_corners=False, antialias=True))["imgs_w"]
            finite_tensor(y, x.shape)
            return quantize(y)

    def detect(self, rgb):
        """Only received pixels; no message/roster/clean image/seed/ROI arguments."""
        import numpy as np
        import torch
        x = self.tensor(rgb)
        with torch.inference_mode():
            if self.method == "wam":
                preds = finite_tensor(self.model.detect(self.normalize(x))["preds"], (1, 33, 256, 256))
                mask = preds[:, :1].sigmoid()
                selected = mask > 0.5
                count = int(selected.sum().item())
                sums = (preds[:, 1:] * selected).double().sum(dim=(2, 3))[0]
                means = None if count == 0 else preds[:, 1:].masked_select(selected.expand_as(preds[:, 1:])).view(32, -1).mean(dim=-1, dtype=float)
                bits = self.native_message(preds[:, 1:], mask, method="semihard")[0]
                # Native empty-mask mean is NaN; retain undefined rather than its bool cast.
                return dict(bits=None if count == 0 else bits.int().cpu().tolist(),
                            selected_pixels=count, sums=sums.cpu().numpy(),
                            selected_means=None if means is None else means.cpu().tolist(),
                            mask=np.packbits(selected[0, 0].cpu().numpy().reshape(-1)),
                            mask_shape=[256, 256], native_threshold=0.5)
            preds = self.model.detect(x, is_video=True,
                                     interpolation=dict(mode="bilinear", align_corners=False, antialias=True))["preds"]
            preds = finite_tensor(preds)
            if list(preds.shape) != [len(rgb), 257]:
                raise ValueError(f"Expected native frame logits Fx257; got {list(preds.shape)}")
            return dict(logits=preds.cpu().numpy(), bits=(preds[:, 1:] > 0).int().cpu().tolist(),
                        video_bits=(preds[:, 1:].mean(0) > 0).int().cpu().tolist())

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model = None
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()

    def resources(self):
        import torch
        return dict(device=self.device, peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device == "cuda" else None)
