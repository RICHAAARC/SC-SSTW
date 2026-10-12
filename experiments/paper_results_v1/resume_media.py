"""Restore disposable RGB caches from the already published Drive MP4."""
from pathlib import Path
from experiments.paper_results_v1.resume_state import unit_attempt


def published_path(receipt, artifacts, read_json):
    seen = set()
    while receipt.get("full_alias_of"):
        name = receipt["full_alias_of"]
        if name in seen:
            raise ValueError("cyclic FULL media alias")
        seen.add(name)
        receipt = next(row for row in artifacts if row["artifact_id"] == name)
    if receipt.get("source_mp4"):
        return Path(receipt["source_mp4"])
    if isinstance(receipt.get("mp4"), dict) and receipt["mp4"].get("path"):
        return Path(receipt["mp4"]["path"])
    if receipt.get("codec_record"):
        record = read_json(receipt["codec_record"]["path"])
        return Path(record["mp4"]["path"])
    raise FileNotFoundError("No saved publication MP4 for " + str(receipt.get("artifact_id")))


def load_rgb(store, receipt, *, artifacts=None, loader=None, decode=None, read_json=None):
    if loader is None or read_json is None:
        from experiments.paper_results_v1 import attack_eval
        loader = loader or attack_eval._load_rgb8
        read_json = read_json or attack_eval.read_json
    try:
        return loader(receipt)
    except (OSError, ValueError) as exc:
        cache_error = f"{type(exc).__name__}: {exc}"
    if artifacts is None:
        artifacts = store.data["artifacts"]
    target = store.temp_root / "restored" / Path(receipt["artifact_id"]) / "received.rgb8"
    # Confirmation reads a fresh attack-state copy; retain the local cache even
    # when that source receipt still names a vanished /content path.
    if target.is_file():
        try:
            value = loader({**receipt, "path": str(target)})
        except (OSError, ValueError):
            pass
        else:
            receipt["path"] = str(target)
            return value
    mp4 = published_path(receipt, artifacts, read_json)
    if decode is None:
        from runtime.wan.variable_rgb_media import decode_published
        decode = decode_published
    with unit_attempt(store, "restore-rgb/" + receipt["artifact_id"],
                      operation="decode_existing_mp4", source=str(mp4), cache_error=cache_error):
        _array, saved = decode(mp4, target, expected_frames=int(receipt["shape"][0]))
        receipt.update(path=saved["path"], source_mp4=str(mp4))
        store.save()
        return loader(receipt)
