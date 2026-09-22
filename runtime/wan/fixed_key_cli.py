"""Command-line interface for the reusable SC-SSTW fixed-key core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.wan import fixed_key_control
from runtime.wan.fixed_key_core import generate_video, load_protocol, receive_mp4
from runtime.wan.io import dump


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    writer = sub.add_parser("generate")
    writer.add_argument("--prompt", required=True); writer.add_argument("--seed", type=int, required=True)
    writer.add_argument("--key", required=True)
    writer.add_argument("--protocol", required=True); writer.add_argument("--output", required=True)
    writer.add_argument("--arm", choices=("SINGLE46", "MULTI44_46"), default="MULTI44_46")
    writer.add_argument("--objective", choices=fixed_key_control.OBJECTIVES,
                        default=fixed_key_control.DEFAULT_OBJECTIVE)
    receiver = sub.add_parser("receive")
    receiver.add_argument("--input-mp4", required=True); receiver.add_argument("--key", required=True)
    receiver.add_argument("--protocol", required=True); receiver.add_argument("--calibration")
    receiver.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "generate":
        result = generate_video(args.prompt, args.seed, args.key.encode(), args.output, protocol, args.arm,
                                objective=args.objective)
        dump(Path(args.output).with_suffix(".writer.json"), result)
    else:
        result = receive_mp4(args.input_mp4, args.key.encode(), protocol, None if args.calibration is None else _json(args.calibration))
        dump(Path(args.output), result)
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
