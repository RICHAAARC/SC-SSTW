"""Writer-only ordinary432-bit diagnostic fixture. Receivers do not import this."""
def prepare():
    from main.tube_state import b_lfg_bch_reference_v1 as bch
    message=b'Video-WM LFG posthoc reference stage1 fixed message v1'
    bits=bch.int_to_bits(int.from_bytes(message,'big'),432)
    return dict(status='WRITER_FIXTURE',message_hex=message.hex(),information_bits=bits,codeword_bits=bch.encode(message),new_user_payload=False,capacity_claim=False,BCH_parameters=bch.parameters())
