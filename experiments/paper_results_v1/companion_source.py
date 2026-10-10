"""Build the ordinary source companion used by the two Paper Results notebooks.

The archive is a convenience distribution, not an execution identity.  It has
no embedded manifest or digest contract.  Notebook extraction still rejects
absolute and parent-traversing member paths before writing any source file.
"""
from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/paper_results_v1_companion.zip"


def companion_files():
    paths = [
        ROOT / "experiments/__init__.py",
        ROOT / "main/__init__.py",
        ROOT / "runtime/__init__.py",
        ROOT / "experiments/paper_results_v1/real_eval.adopted.json",
        ROOT / "experiments/paper_results_v1/attack_eval.adopted.json",
        ROOT / "experiments/paper_results_v1/source_identity_audit.json",
    ]
    paths.extend(sorted((ROOT / "experiments/paper_results_v1").glob("*.py")))
    paths.extend(sorted((ROOT / "main/tube_state").rglob("*.py")))
    paths.extend(sorted((ROOT / "runtime/wan").rglob("*.py")))
    unique = {path.relative_to(ROOT).as_posix(): path for path in paths}
    return [(name, unique[name]) for name in sorted(unique)]


def build_companion_zip(output=OUTPUT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as archive:
        for name, path in companion_files():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    temporary.replace(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "files": len(companion_files()),
    }


if __name__ == "__main__":
    print(build_companion_zip())
