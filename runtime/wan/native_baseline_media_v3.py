"""Actual AVI preparation and H264 channel. Invoked only by the user-run CLI."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import urllib.request


def crop_box(width, height):
    # Exact largest integer 8:5 central crop; standard 320x240 -> (0,20,320,200).
    scale = min(width//8, height//5)
    if scale < 1:
        raise ValueError('Source too small for an 8:5 crop')
    w, h = 8*scale, 5*scale
    return (width-w)//2, (height-h)//2, w, h


class MediaV3:
    def command(self, args, *, data=None):
        return subprocess.run(args, input=data, check=True, capture_output=True, timeout=600).stdout

    def download(self, source, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError('Source prepare must not overwrite an existing media artifact')
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'Video-WM-native-baseline-v3'})
        received = 0
        with urllib.request.urlopen(request, timeout=120) as response, path.open('xb') as target:
            while chunk := response.read(1024*1024):
                target.write(chunk); received += len(chunk)
            target.flush(); __import__('os').fsync(target.fileno())
        if received == 0:
            raise ValueError('Empty source download')
        return dict(url=source['url'], bytes=received, download_calls=1,
                    http_requests=None, http_request_measurement='NOT_MEASURED: urllib may follow redirects')

    def source_probe(self, path):
        raw = self.command(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', str(path)])
        s = json.loads(raw)['streams'][0]
        if Fraction(s['r_frame_rate']) != 25 or int(s.get('nb_frames', -1)) < 181:
            raise ValueError(f'Actual AVI format is not 25fps with >=181 frames: {s}')
        crop_box(int(s['width']), int(s['height']))
        return s

    def normalize(self, path, metadata):
        import numpy as np
        x, y, w, h = crop_box(int(metadata['width']), int(metadata['height']))
        raw = self.command(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-noautorotate',
                            '-i', str(path), '-map', '0:v:0', '-an', '-vf',
                            f'crop={w}:{h}:{x}:{y},scale=512:320:flags=bilinear',
                            '-frames:v', '181', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
        if len(raw) != 181*320*512*3:
            raise ValueError('Actual normalized source lacks exactly 181 real frames; no padding/replacement')
        return np.frombuffer(raw, dtype=np.uint8).reshape(181, 320, 512, 3).copy()

    def encode(self, rgb, path, fps):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        h, w = rgb.shape[1:3]
        self.command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                      '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', str(path)], data=rgb.tobytes())
        return dict(path=str(path), bytes=path.stat().st_size, codec='h264', crf=18)

    def probe(self, path, shape, fps):
        s = json.loads(self.command(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                    'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames', '-of', 'json', str(path)]))['streams'][0]
        if (s.get('codec_name') != 'h264' or s.get('pix_fmt') != 'yuv420p'
                or [s.get('height'), s.get('width')] != list(shape[1:3])
                or Fraction(s['r_frame_rate']) != fps or int(s.get('nb_frames', -1)) != shape[0]):
            raise ValueError(f'Actual result media differs from declared format: {s}')
        return s

    def decode(self, path, shape):
        import numpy as np
        raw = self.command(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-noautorotate', '-i', str(path),
                            '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
        if len(raw) != int(np.prod(shape)):
            raise ValueError('Decoded byte count mismatch; no padding or truncation')
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
