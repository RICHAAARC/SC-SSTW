"""Runtime source identity for Git checkouts and detached release directories.

Source location is optional; no per-file integrity inventory is constructed.
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
SOURCE_FILES = tuple(sorted((*SOURCE_FILES, *CONDITIONAL_FILES)))



def file_hashes(root):
    """Legacy optional record field; no source-byte inventory is required."""
    return {}


def content_id(files):
    return None


def source_identity(root):
    """Best-effort location only; never reads a manifest or hashes source files."""
    root = Path(root).resolve()
    receipt = dict(kind="source_directory", root=str(root), git_commit=None,
        git_status=None, files={}, content_sha256=None, manifest_status="NOT_USED",
        manifest_content_sha256=None, manifest_sha256=None, changed_files=[], release_id=None)
    try:
        if (root / ".git").exists():
            receipt["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip() or None
    except Exception as exc:
        receipt["record_issue"] = str(exc)
    return receipt
