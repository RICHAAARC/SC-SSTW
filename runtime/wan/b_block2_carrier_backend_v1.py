"""Lazy real Wan/media backend; runner imports only after durable attempt."""
from runtime.wan.b_full_context_local_read_backend_v1 import FullBackend
from runtime.wan import b_block2_carrier_tensor_v1 as tensor
from main.tube_state import b_block2_carrier_v1 as method
from main.tube_state import b_legacy_position_margin_v1 as payload

class Block2Backend(FullBackend):
    def construct_variant(self,z,family,key):return tensor.construct(z,family,key)
    def read_variant(self,z,family,key):return method.raw_table(tensor.project_rows(z,family,key),family,key)
    def payload_spectrum(self,z):return payload.payload_spectrum(z)
    def payload_votes(self,value,key):return payload.payload_from_spectrum(value,key)
