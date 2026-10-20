"""Archived margin loop in memory; no file hashes/cache identity admission."""
from typing import Any
from .tubelet_partition import (TubeletPartitionConfig, build_tubelet_partition_config,
    build_tubelet_descriptors, dot_tubelet_direction, add_tubelet_delta_in_place)
from .codebook import build_tubelet_codebook, build_codebook_config
from .tensor_artifact import FloatTensorArtifact

def build_partition_config_from_method_config(
    method_config: dict[str, Any],
) -> TubeletPartitionConfig:
    if not isinstance(method_config, dict):
        raise TypeError("method_config must be a dictionary")

    tubelet_length = int(method_config.get("tubelet_length", 1))
    tubelet_partition = method_config.get("tubelet_partition", {})
    if tubelet_partition and not isinstance(tubelet_partition, dict):
        raise TypeError("tubelet_partition must be a dictionary")
    spatial_patch_size = tuple(tubelet_partition.get("spatial_patch_size", (4, 4)))
    if len(spatial_patch_size) != 2:
        raise ValueError("spatial_patch_size must contain two integers")
    spatial_stride_payload = tubelet_partition.get("spatial_stride", spatial_patch_size)
    if isinstance(spatial_stride_payload, dict):
        spatial_stride = (
            int(spatial_stride_payload.get("height", spatial_patch_size[0])),
            int(spatial_stride_payload.get("width", spatial_patch_size[1])),
        )
    else:
        spatial_stride = tuple(spatial_stride_payload)
    temporal_stride = int(tubelet_partition.get("temporal_stride", tubelet_length))
    return build_tubelet_partition_config(
        tubelet_length=tubelet_length,
        spatial_patch_size=(int(spatial_patch_size[0]), int(spatial_patch_size[1])),
        temporal_stride=temporal_stride,
        spatial_stride=(int(spatial_stride[0]), int(spatial_stride[1])),
        allow_partial_tubelet=bool(
            tubelet_partition.get("allow_partial_tubelet", True)
        ),
    )

def projection_margin(tensor, reference_id, partition_config, *, codebook_config=None,
                      enable_sync=False, embedding_margin=1.0):
    if embedding_margin <= 0:
        raise ValueError("positive archived embedding margin required")
    descriptors=build_tubelet_descriptors(tensor.shape,partition_config)
    book=build_tubelet_codebook(reference_id,descriptors,len(descriptors[0].flat_indices),
        codebook_config or build_codebook_config(),enable_sync=enable_sync)
    before=[];after=[];delta_norm=[]
    for descriptor in descriptors:
        direction=book.directions[descriptor.tubelet_index]
        code_sign=book.combined_codes[descriptor.tubelet_index]
        projection=code_sign*dot_tubelet_direction(tensor,descriptor,direction)
        before.append(projection)
        if projection < embedding_margin:
            delta_scale=embedding_margin-projection
            add_tubelet_delta_in_place(tensor,descriptor,direction,delta_scale*code_sign)
            after.append(projection+delta_scale*book.direction_norm_squares[descriptor.tubelet_index])
            delta_norm.append(delta_scale)
        else:
            after.append(projection);delta_norm.append(0.)
    return tensor,dict(embedding_rule="projection_margin",embedding_margin=embedding_margin,
        mean_projection_before=round(sum(before)/len(before),6),
        mean_projection_after=round(sum(after)/len(after),6),
        mean_embedding_delta_norm=round(sum(delta_norm)/len(delta_norm),6),
        before=before,after=after,current_B_net32_writer_called=False,
        scope="archived fixed signature reference, not current net32 payload writer")
