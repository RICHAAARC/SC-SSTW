"""Public-reference CPU control. No writer trace, answers or attack metadata input."""
from copy import deepcopy
from .evidence import SyntheticProbeEvidenceExtractor
from .schema import LatentSample
from .codebook import build_codebook_config
from .tensor_artifact import read_float_tensor_npy

class PublicReferenceExtractor(SyntheticProbeEvidenceExtractor):
    def _scale_search_enabled(self, sample=None):
        return bool(self._resolve_sync_search_config().get("enable_scale_search",False))

    def _filter_scale_candidates_for_speed_change(self,candidates,sample):
        return candidates  # all publicly declared scales, independent of true attack

    def _resolve_sync_search_score_rule(self,sample):
        return self._resolve_sync_search_config().get("search_score_rule","penalized_prior")


def extract_public_reference(received_latent_path, reference_id, reference_shape,
                             public_method_config, codebook_config=None):
    config=deepcopy(public_method_config)
    config.setdefault("score_calibration",{})["embedding_projection_support_weight"]=0.
    sample_shape=read_float_tensor_npy(received_latent_path).shape
    sample=LatentSample(sample_id=reference_id,split="",sample_role="",
        latent_shape=sample_shape,latent_tensor_digest_random="",latent_generation_seed_random=0,
        latent_backend_name="fixed_cpu_reference",latent_backend_status="no_model",
        latent_artifact_path=str(received_latent_path),
        mechanism_trace={"reference_latent_shape":list(reference_shape)},applied_attack_params={})
    extractor=PublicReferenceExtractor("tubelet_sync",config,{"tubelet":True,"sync":True},"sync_rescue_fusion")
    extractor._codebook_config=codebook_config or build_codebook_config()
    scores,trace=extractor.extract(sample)
    return dict(scores=scores,trace=trace,writer_support_enabled=False,
        reference_id_pre_registered=True,actual_message_used=False,attack_metadata_used=False,
        scope="CPU reference signature/alignment/fusion portability control, not Wan migration success")
