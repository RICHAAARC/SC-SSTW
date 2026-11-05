"""Shared uint8 RGB / H264 channel, invoked only by the user-run runner."""
import json
from pathlib import Path
import subprocess


class Media:
    def read_source(self, root, source, shape, fps):
        import numpy as np
        path = Path(root) / source["relative_rgb"]
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text())
        if meta.get("dtype") != "uint8" or meta.get("shape") != shape or float(meta.get("fps", -1)) != fps:
            raise ValueError(f"Invalid raw RGB sidecar: {sidecar}")
        count = int(np.prod(shape))
        if path.stat().st_size != count:
            raise ValueError(f"RGB byte length mismatch: {path}")
        return np.fromfile(path, dtype=np.uint8).reshape(shape)

    def encode(self, rgb, path, fps):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = rgb.shape[1:3]
        args = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1", str(path)]
        subprocess.run(args, input=rgb.tobytes(), check=True, capture_output=True)
        return dict(path=str(path), bytes=path.stat().st_size, codec="h264", crf=18)

    def probe(self, path, shape, fps):
        from fractions import Fraction
        value = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames", "-of", "json", str(path)],
                               check=True, capture_output=True, text=True)
        stream = json.loads(value.stdout)["streams"][0]
        if (stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p"
                or [stream.get("height"), stream.get("width")] != shape[1:3]
                or Fraction(stream["r_frame_rate"]) != fps or int(stream.get("nb_frames", -1)) != shape[0]):
            raise ValueError(f"Actual H264 stream differs from declared media semantics: {stream}")
        return stream

    def decode(self, path, shape):
        import numpy as np
        value = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path),
                                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                               check=True, capture_output=True)
        if len(value.stdout) != int(np.prod(shape)):
            raise ValueError("Decoded RGB byte count mismatch; no padding or truncation")
        return np.frombuffer(value.stdout, dtype=np.uint8).reshape(shape).copy()


def quality(left, right):
    import numpy as np
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Quality requires same-content uint8 RGB pairs")
    error = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(error ** 2))
    return dict(frames=len(left), pixels=int(np.prod(left.shape[:-1])), mse_rgb255=mse,
                psnr_db=None if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
                psnr_infinite=mse == 0, quality_threshold=None)
