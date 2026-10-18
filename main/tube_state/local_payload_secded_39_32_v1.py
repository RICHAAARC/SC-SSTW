"""Fixed systematic shortened extended Hamming (39,32,4); no truth input."""
PARITY_COORDINATES = (1, 2, 4, 8, 16, 32, 0)
DATA_COORDINATES = tuple(i for i in range(39) if i not in PARITY_COORDINATES)


def _bits(value, length, *, erasures=False):
    if len(value) != length or any(x is not None and (type(x) is not int or x not in (0, 1))
                                  or (x is None and not erasures) for x in value):
        raise ValueError(f"expected {length} binary bits" + (" or None" if erasures else ""))


def syndrome(word):
    _bits(word, 39)
    sigma = parity = 0
    for i, bit in enumerate(word):
        if bit:
            sigma ^= i
            parity ^= 1
    return sigma, parity


def encode(message):
    _bits(message, 32)
    word = [0] * 39
    for coordinate, bit in zip(DATA_COORDINATES, message):
        word[coordinate] = bit
    for parity in PARITY_COORDINATES[:-1]:
        value = 0
        for coordinate in DATA_COORDINATES:
            if coordinate & parity:
                value ^= word[coordinate]
        word[parity] = value
    word[0] = sum(word[1:]) % 2
    return word


def decode(received):
    """One SECDED rule, no erasure recovery, nearest-word search or soft input."""
    _bits(received, 39, erasures=True)
    out = dict(status="UNRESOLVED", received=list(received), syndrome=None,
               flipped_coordinate=None, rule_applied=False, returned_codeword=None,
               message=[None]*32, syndrome_reductions=0, truth_used=False)
    if any(bit is None for bit in received):
        out["reason"] = "RAW_SYMBOL_ERASURE"
        return out
    sigma, parity = syndrome(received)
    out.update(syndrome=dict(sigma=sigma, overall_parity=parity), syndrome_reductions=1)
    if sigma and not parity:
        out["status"] = "DETECTED_UNCORRECTABLE"
        return out
    if parity and sigma > 38:
        out["status"] = "INVALID_SHORTENED_SYNDROME"
        return out
    word = list(received)
    if parity:
        word[sigma] ^= 1  # sigma=0 means the overall parity coordinate.
        out.update(rule_applied=True, flipped_coordinate=sigma, syndrome_reductions=2)
        if syndrome(word) != (0, 0):
            out["status"] = "INVALID_CODEWORD"
            return out
    out.update(status="SINGLE_ERROR_RULE_APPLIED" if parity else "CODE_CONSISTENT",
               returned_codeword=word, message=[word[i] for i in DATA_COORDINATES])
    return out
