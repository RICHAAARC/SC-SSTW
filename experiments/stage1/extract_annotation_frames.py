"""Decode only: export actual sampled grayscale frames, never observe q."""
import argparse
import json
from pathlib import Path
import struct
import zlib

from runtime.stage1.observation import decode_video
from experiments.stage1.evaluate_relation import COORDINATES


def png_gray(width, height, pixels):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xffffffff)
    scanlines = b"".join(b"\0" + pixels[y * width:(y + 1) * width] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def extract(path, output_dir, *, sample_id, position_definition, decode_config):
    output = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    if output == repository or repository in output.parents:
        raise ValueError("annotation exports must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    width, height, frames = decode_video(path, **decode_config)
    rows = []
    for i, frame in enumerate(frames):
        filename = f"sample_{i:04d}.png"
        (output / filename).write_bytes(png_gray(width, height, frame))
        rows.append({"sample_index": i, "time_seconds": i / decode_config["sample_hz"],
                     "frame_file": filename, "p": None, "p_pixel": None,
                     "uncertainty_pixels": None, "reason": "UNANNOTATED"})
    result = {"sample_id": sample_id, "position_definition": position_definition,
              "coordinate_system": COORDINATES, "frame_width": width, "frame_height": height,
              "source_path": str(path), "decode_config": decode_config,
              "annotation_status": "UNANNOTATED_DO_NOT_EVALUATE_AS_TRUTH", "rows": rows}
    (output / "annotations_template.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--position-definition", required=True)
    parser.add_argument("--decode-config", required=True, help="JSON file containing the frozen decode object only")
    args = parser.parse_args()
    extract(args.video, args.output, sample_id=args.sample_id, position_definition=args.position_definition,
            decode_config=json.loads(Path(args.decode_config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
