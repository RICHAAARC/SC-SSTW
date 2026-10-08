"""Runtime source identity for Git checkouts and detached release directories.

The bundled content manifest identifies source bytes, not an invented Git SHA.
No model input, receiver decision or method threshold depends on this metadata.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess

SOURCE_FILES = ('experiments/__init__.py', 'experiments/wan_state_clock/__init__.py', 'experiments/wan_state_clock/configs/grow_video_reference_v1.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_internal_single_deletion_v1.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_internal_single_deletion_v1_oracle.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_internal_single_deletion_v1_posthoc.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_internal_single_deletion_v1_preparation.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_receiver_estimated_align_v1.json', 'experiments/wan_state_clock/configs/historical/video_trajectory_receiver_estimated_align_v1_posthoc.json', 'experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1.json', 'experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1_oracle.json', 'experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1_posthoc.json', 'experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1_preparation.json', 'experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1.json', 'experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1_posthoc.json', 'experiments/wan_state_clock/configs/video_trajectory_receiver_estimated_align_v1.json', 'experiments/wan_state_clock/configs/video_trajectory_receiver_estimated_align_v1_posthoc.json', 'experiments/wan_state_clock/grow_video_reference_run.py', 'experiments/wan_state_clock/receiver_records.py', 'experiments/wan_state_clock/requirements-grow-video-reference.txt', 'experiments/wan_state_clock/video_trajectory_internal_single_deletion_v1_run.py', 'experiments/wan_state_clock/video_trajectory_payload_framewise_sync_m05_v1_run.py', 'experiments/wan_state_clock/video_trajectory_receiver_estimated_align_v1_run.py', 'main/__init__.py', 'main/tube_state/__init__.py', 'main/tube_state/grow_video_reference.py', 'main/tube_state/payload_reader.py', 'main/tube_state/video_trajectory_internal_single_deletion_v1.py', 'main/tube_state/video_trajectory_payload_framewise_sync_v1.py', 'main/tube_state/video_trajectory_receiver_estimated_align_v1.py', 'main/tube_state/video_trajectory_receiver_origin_v1.py', 'runtime/__init__.py', 'runtime/wan/__init__.py', 'runtime/wan/fixed_rgb_media.py', 'runtime/wan/framewise_autoencoder_kl.py', 'runtime/wan/generation.py', 'runtime/wan/grow_video_reference.py', 'runtime/wan/io.py', 'runtime/wan/provenance.py', 'runtime/wan/rgb8_source.py', 'runtime/wan/trajectory.py', 'runtime/wan/vae.py', 'runtime/wan/video_trajectory_internal_single_deletion_v1.py', 'runtime/wan/video_trajectory_payload_framewise_sync_v1.py', 'runtime/wan/video_trajectory_payload_gt_v1.py', 'runtime/wan/video_trajectory_receiver_estimated_align_v1.py', 'runtime/wan/video_trajectory_receiver_origin_v1.py')

CONDITIONAL_FILES = (
    "experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1.json",
    "experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1_oracle.json",
    "experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1_posthoc.json",
    "experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1_preparation.json",
    "experiments/wan_state_clock/video_trajectory_conditional_joint_v1_prepare.py",
    "experiments/wan_state_clock/video_trajectory_conditional_joint_v1_run.py",
    "main/tube_state/video_trajectory_conditional_joint_v1.py",
    "runtime/wan/conditional_joint/__init__.py",
    "runtime/wan/conditional_joint/fixed_rgb_media.py",
    "runtime/wan/conditional_joint/generation.py",
    "runtime/wan/conditional_joint/quality.py",
    "runtime/wan/conditional_joint/trajectory.py",
    "runtime/wan/conditional_joint/vae.py",
    "runtime/wan/video_trajectory_conditional_joint_v1.py",
)
ATTRIBUTION_FILES = (
    "experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json",
    "experiments/wan_state_clock/video_trajectory_attribution_v1_prepare.py",
    "experiments/wan_state_clock/video_trajectory_attribution_v1_protocol.py",
    "experiments/wan_state_clock/video_trajectory_attribution_v1_run.py",
    "main/tube_state/video_trajectory_attribution_v1.py",
    "runtime/wan/video_trajectory_attribution_v1.py",
)
ATTRIBUTION_UNCERTAINTY_FILES = (
    "experiments/wan_state_clock/configs/video_trajectory_attribution_uncertainty_v1.json",
    "experiments/wan_state_clock/video_trajectory_attribution_uncertainty_v1_protocol.py",
    "experiments/wan_state_clock/video_trajectory_attribution_uncertainty_v1_run.py",
)
SOURCE_FILES = tuple(sorted((
    *SOURCE_FILES,
    *CONDITIONAL_FILES,
    *ATTRIBUTION_FILES,
    *ATTRIBUTION_UNCERTAINTY_FILES,
)))



def file_hashes(root):
    root = Path(root)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}

def content_id(files):
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def source_identity(root):
    root = Path(root).resolve()
    actual = file_hashes(root)
    receipt = dict(kind="unversioned_directory", git_commit=None,
        git_status=None, files=actual, content_sha256=content_id(actual),
        manifest_status="ABSENT", manifest_content_sha256=None,
        manifest_sha256=None, changed_files=[], release_id=None)
    manifest_path = root / "release_manifest.json"
    if manifest_path.is_file():
        receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        try:
            m = json.loads(manifest_path.read_text())
            files = m["files"]
            if (m["schema"] != 1 or set(files) != set(SOURCE_FILES)
                or any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h)
                       for h in files.values())
                or m["content_sha256"] != content_id(files)
                or not isinstance(m["release_id"], str)):
                raise ValueError("manifest schema/content identity mismatch")
            receipt.update(kind="release_manifest", release_id=m["release_id"],
                manifest_content_sha256=m["content_sha256"],
                changed_files=[name for name in SOURCE_FILES if files[name] != actual[name]])
            receipt["manifest_status"] = "MODIFIED" if receipt["changed_files"] else "MATCH"
        except (ValueError, KeyError, TypeError) as exc:
            receipt.update(manifest_status="INVALID", manifest_error=str(exc))
    # Never inherit a parent workspace's unrelated .git.
    if (root / ".git").exists():
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                stderr=subprocess.DEVNULL, text=True).strip()
            if not re.fullmatch("[0-9a-f]{40}", sha):
                raise ValueError("invalid Git commit response")
            status = subprocess.check_output(["git", "status", "--porcelain", "--",
                *SOURCE_FILES, "release_manifest.json"], cwd=root,
                stderr=subprocess.DEVNULL, text=True).splitlines()
            receipt.update(kind="git_checkout", git_commit=sha, git_status=status)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            receipt["git_error"] = str(exc)
    return receipt
