"""Explicit saved-file reporting only; no model, attack or recovery subcommands."""
import argparse
from pathlib import Path

from .report import build_report, load_json, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_report(load_json(args.protocol), base_dir=args.protocol.parent)
    for path in write_report(report, args.output):
        print(path)


if __name__ == "__main__":
    main()
