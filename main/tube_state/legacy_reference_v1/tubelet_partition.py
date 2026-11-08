
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .digest import compute_object_digest
from .tensor_artifact import FloatTensorArtifact


@dataclass(frozen=True)
class TubeletPartitionConfig:

    tubelet_length: int
    spatial_patch_size: tuple[int, int]
    temporal_stride: int
    spatial_stride: tuple[int, int]
    allow_partial_tubelet: bool = False


@dataclass(frozen=True)
class TubeletDescriptor:

    temporal_index: int
    tubelet_index: int
    frame_start: int
    frame_stop: int
    height_start: int
    height_stop: int
    width_start: int
    width_stop: int
    flat_indices: tuple[int, ...]


def build_tubelet_partition_config(
    tubelet_length: int,
    spatial_patch_size: tuple[int, int] = (4, 4),
    temporal_stride: int | None = None,
    spatial_stride: tuple[int, int] | None = None,
    allow_partial_tubelet: bool = False,
) -> TubeletPartitionConfig:
    if not isinstance(tubelet_length, int) or tubelet_length < 1:
        raise ValueError("tubelet_length must be a positive integer")
    if (
        not isinstance(spatial_patch_size, tuple)
        or len(spatial_patch_size) != 2
        or any(not isinstance(size, int) or size < 1 for size in spatial_patch_size)
    ):
        raise ValueError("spatial_patch_size must contain two positive integers")

    resolved_temporal_stride = tubelet_length if temporal_stride is None else temporal_stride
    if not isinstance(resolved_temporal_stride, int) or resolved_temporal_stride < 1:
        raise ValueError("temporal_stride must be a positive integer")

    resolved_spatial_stride = spatial_patch_size if spatial_stride is None else spatial_stride
    if (
        not isinstance(resolved_spatial_stride, tuple)
        or len(resolved_spatial_stride) != 2
        or any(not isinstance(size, int) or size < 1 for size in resolved_spatial_stride)
    ):
        raise ValueError("spatial_stride must contain two positive integers")

    if not isinstance(allow_partial_tubelet, bool):
        raise TypeError("allow_partial_tubelet must be a boolean")

    return TubeletPartitionConfig(
        tubelet_length=tubelet_length,
        spatial_patch_size=spatial_patch_size,
        temporal_stride=resolved_temporal_stride,
        spatial_stride=resolved_spatial_stride,
        allow_partial_tubelet=allow_partial_tubelet,
    )


def build_tubelet_descriptors(
    latent_shape: tuple[int, int, int, int],
    partition_config: TubeletPartitionConfig,
) -> list[TubeletDescriptor]:
    if (
        not isinstance(latent_shape, tuple)
        or len(latent_shape) != 4
        or any(not isinstance(size, int) or size < 1 for size in latent_shape)
    ):
        raise ValueError("latent_shape must be a 4-tuple of positive integers")
    if not isinstance(partition_config, TubeletPartitionConfig):
        raise TypeError("partition_config must be a TubeletPartitionConfig instance")

    return list(_build_cached_tubelet_descriptors(latent_shape, partition_config))


@lru_cache(maxsize=32)
def _build_cached_tubelet_descriptors(
    latent_shape: tuple[int, int, int, int],
    partition_config: TubeletPartitionConfig,
) -> tuple[TubeletDescriptor, ...]:
    frame_count, _, height, width = latent_shape
    patch_height, patch_width = partition_config.spatial_patch_size
    stride_height, stride_width = partition_config.spatial_stride
    temporal_starts = _build_axis_starts(
        frame_count,
        partition_config.tubelet_length,
        partition_config.temporal_stride,
        partition_config.allow_partial_tubelet,
    )
    height_starts = _build_axis_starts(
        height,
        patch_height,
        stride_height,
        partition_config.allow_partial_tubelet,
    )
    width_starts = _build_axis_starts(
        width,
        patch_width,
        stride_width,
        partition_config.allow_partial_tubelet,
    )

    descriptors: list[TubeletDescriptor] = []
    tubelet_index = 0
    for frame_start in temporal_starts:
        frame_stop = min(frame_count, frame_start + partition_config.tubelet_length)
        for height_start in height_starts:
            height_stop = min(height, height_start + patch_height)
            for width_start in width_starts:
                width_stop = min(width, width_start + patch_width)
                descriptors.append(
                    TubeletDescriptor(
                        temporal_index=frame_start,
                        tubelet_index=tubelet_index,
                        frame_start=frame_start,
                        frame_stop=frame_stop,
                        height_start=height_start,
                        height_stop=height_stop,
                        width_start=width_start,
                        width_stop=width_stop,
                        flat_indices=_build_flat_indices(
                            latent_shape,
                            frame_start,
                            frame_stop,
                            height_start,
                            height_stop,
                            width_start,
                            width_stop,
                        ),
                    )
                )
                tubelet_index += 1
    return tuple(descriptors)


def compute_tubelet_partition_digest(
    latent_shape: tuple[int, int, int, int],
    partition_config: TubeletPartitionConfig,
) -> str:
    return _build_cached_partition_digest(latent_shape, partition_config)


@lru_cache(maxsize=32)
def _build_cached_partition_digest(
    latent_shape: tuple[int, int, int, int],
    partition_config: TubeletPartitionConfig,
) -> str:
    descriptors = _build_cached_tubelet_descriptors(latent_shape, partition_config)
    return compute_object_digest(
        {
            "latent_shape": list(latent_shape),
            "partition_config": {
                "tubelet_length": partition_config.tubelet_length,
                "spatial_patch_size": list(partition_config.spatial_patch_size),
                "temporal_stride": partition_config.temporal_stride,
                "spatial_stride": list(partition_config.spatial_stride),
                "allow_partial_tubelet": partition_config.allow_partial_tubelet,
            },
            "descriptors": [
                {
                    "temporal_index": descriptor.temporal_index,
                    "tubelet_index": descriptor.tubelet_index,
                    "frame_start": descriptor.frame_start,
                    "frame_stop": descriptor.frame_stop,
                    "height_start": descriptor.height_start,
                    "height_stop": descriptor.height_stop,
                    "width_start": descriptor.width_start,
                    "width_stop": descriptor.width_stop,
                }
                for descriptor in descriptors
            ],
        }
    )


def extract_tubelet_values(
    tensor_artifact: FloatTensorArtifact,
    tubelet_descriptor: TubeletDescriptor,
) -> list[float]:
    if not isinstance(tensor_artifact, FloatTensorArtifact):
        raise TypeError("tensor_artifact must be a FloatTensorArtifact instance")
    if not isinstance(tubelet_descriptor, TubeletDescriptor):
        raise TypeError("tubelet_descriptor must be a TubeletDescriptor instance")

    values = [
        float(tensor_artifact.values[flat_index])
        for flat_index in tubelet_descriptor.flat_indices
    ]
    if not values:
        raise ValueError("tubelet extraction produced an empty payload")
    return values


def dot_tubelet_direction(
    tensor_artifact: FloatTensorArtifact,
    tubelet_descriptor: TubeletDescriptor,
    direction: list[float] | tuple[float, ...],
) -> float:
    if not isinstance(tensor_artifact, FloatTensorArtifact):
        raise TypeError("tensor_artifact must be a FloatTensorArtifact instance")
    if not isinstance(tubelet_descriptor, TubeletDescriptor):
        raise TypeError("tubelet_descriptor must be a TubeletDescriptor instance")
    if not isinstance(direction, (list, tuple)) or not direction:
        raise ValueError("direction must be a non-empty sequence")
    if len(direction) != len(tubelet_descriptor.flat_indices):
        raise ValueError("direction length does not match the tubelet payload size")

    return sum(
        float(tensor_artifact.values[flat_index]) * float(direction_value)
        for flat_index, direction_value in zip(
            tubelet_descriptor.flat_indices,
            direction,
        )
    )


def add_tubelet_delta_in_place(
    tensor_artifact: FloatTensorArtifact,
    tubelet_descriptor: TubeletDescriptor,
    direction: list[float] | tuple[float, ...],
    scale: float,
) -> None:
    if not isinstance(tensor_artifact, FloatTensorArtifact):
        raise TypeError("tensor_artifact must be a FloatTensorArtifact instance")
    if not isinstance(tubelet_descriptor, TubeletDescriptor):
        raise TypeError("tubelet_descriptor must be a TubeletDescriptor instance")
    if not isinstance(direction, (list, tuple)) or not direction:
        raise ValueError("direction must be a non-empty sequence")
    if not isinstance(scale, (int, float)):
        raise TypeError("scale must be numeric")

    if len(direction) != len(tubelet_descriptor.flat_indices):
        raise ValueError("direction length does not match the tubelet payload size")

    for flat_index, direction_value in zip(tubelet_descriptor.flat_indices, direction):
        tensor_artifact.values[flat_index] = float(
            tensor_artifact.values[flat_index]
            + (float(scale) * float(direction_value))
        )


def _build_axis_starts(
    axis_length: int,
    window_size: int,
    stride: int,
    allow_partial_tubelet: bool,
) -> list[int]:
    if window_size > axis_length and not allow_partial_tubelet:
        raise ValueError("window size exceeds axis length while partial tubelets are disabled")

    starts: list[int] = []
    current = 0
    while current < axis_length:
        end = current + window_size
        if end > axis_length and not allow_partial_tubelet:
            break
        starts.append(current)
        if end >= axis_length:
            break
        current += stride

    if not starts:
        raise ValueError("partition produced no valid tubelet starts")
    return starts


def _build_flat_indices(
    latent_shape: tuple[int, int, int, int],
    frame_start: int,
    frame_stop: int,
    height_start: int,
    height_stop: int,
    width_start: int,
    width_stop: int,
) -> tuple[int, ...]:
    _, channel_count, _, _ = latent_shape
    flat_indices: list[int] = []
    for frame_index in range(frame_start, frame_stop):
        for channel_index in range(channel_count):
            for height_index in range(height_start, height_stop):
                for width_index in range(width_start, width_stop):
                    flat_indices.append(
                        _flat_index(
                            latent_shape,
                            frame_index,
                            channel_index,
                            height_index,
                            width_index,
                        )
                    )
    if not flat_indices:
        raise ValueError("tubelet flat-index construction produced an empty payload")
    return tuple(flat_indices)


def _flat_index(
    latent_shape: tuple[int, int, int, int],
    frame_index: int,
    channel_index: int,
    height_index: int,
    width_index: int,
) -> int:
    frame_count, channel_count, height, width = latent_shape
    del frame_count
    return (
        (((frame_index * channel_count) + channel_index) * height + height_index) * width
        + width_index
    )


def _flat_index(
    latent_shape: tuple[int, ...],
    frame_index: int,
    channel_index: int,
    height_index: int,
    width_index: int,
) -> int:
    _, channel_count, height, width = latent_shape
    return (
        (((frame_index * channel_count) + channel_index) * height + height_index) * width
        + width_index
    )