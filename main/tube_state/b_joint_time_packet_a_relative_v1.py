"""Conditional hard-code consistency inside unchanged F42 W; no presence rule."""
from copy import deepcopy

from . import b_joint_time_packet_a_relation_v1 as f42
from . import b_joint_time_packet_a_v1 as packet_code

ROLES = ("R0", "R1")
SLOTS = tuple((lane, role) for lane in range(5) for role in ROLES)


def counters():
    return dict(nodes=0, comparable_nodes=0, compared_candidates=0, hard_bit_comparisons=0,
                template_words_encoded=0, strict_winners=0, ties=0, conditional_edge_pairs=0,
                bch_decode_calls=0, model_calls=0, codec_calls=0)


def shared_hard_context(raw_node, unit_id, received_frame):
    """One original, uncorrected hard read, canonicalized by actual physical address."""
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("Explicit received unit address required")
    if not isinstance(raw_node, dict) or type(raw_node.get("received_frame")) is not int or raw_node["received_frame"] != received_frame:
        raise ValueError("Original packet received-frame address mismatch")
    if "unit_id" in raw_node and raw_node["unit_id"] != unit_id:
        raise ValueError("Original packet unit address mismatch")
    observations = raw_node.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Original packet observations required")
    slots = {}
    for obs in observations:
        if not isinstance(obs, dict):
            raise ValueError("Physical observation must be an object")
        if obs.get("role") == "F":
            continue  # Derived fusion never supplies a physical copy or an extra vote.
        lane, role = obs.get("lane"), obs.get("role")
        if type(lane) is not int or (lane, role) not in SLOTS or (lane, role) in slots:
            raise ValueError("Missing, duplicate or invalid physical lane/copy coordinate")
        if obs.get("physical_copy", True) is not True:
            raise ValueError("R0/R1 observation marked nonphysical")
        if "received_frame" in obs and (type(obs["received_frame"]) is not int or obs["received_frame"] != received_frame):
            raise ValueError("Physical observation belongs to another received frame")
        if "unit_id" in obs and obs["unit_id"] != unit_id:
            raise ValueError("Physical observation belongs to another unit")
        bits, known = obs.get("raw_bits"), obs.get("known")
        if not isinstance(bits, list) or len(bits) != 63 or any(type(b) is not int or b not in (0, 1) for b in bits):
            raise ValueError("63 original integer hard bits required; corrected words are not inputs")
        if not isinstance(known, list) or len(known) != 63 or any(type(k) is not bool for k in known) or not all(known):
            raise ValueError("All 630 original hard-known flags must be true")
        # UNRESOLVED_CODE/INVALID_PACKET do not discard valid physical hard bits.
        slots[lane, role] = tuple(bits)
    if set(slots) != set(SLOTS):
        raise ValueError("Exactly ten physical copy slots required; fusion cannot fill a slot")
    return dict(unit_id=unit_id, received_frame=received_frame,
                slots=[dict(lane=l, role=r, bits=slots[l, r]) for l, r in SLOTS],
                coordinate_order="lane0..4 / R0,R1 / bit_index0..62 (MSB62..LSB0)", known_bits=630)


def support(n):
    lanes = list(range(min(n + 1, 5)))
    return dict(candidate_n=n, active_lanes=lanes, roles=list(ROLES), bit_indices=[0, 62],
                support_bits=126*len(lanes), complete_630=len(lanes) == 5, distance=None)


def template(n, cache, counts):
    if n not in cache:
        words = []
        for lane in range(5):
            j = n - ((n-lane) % 5)
            age = n-j
            words.append(tuple(packet_code.encode_raw24(packet_code.message(j, age))))
            counts["template_words_encoded"] += 1
        cache[n] = tuple(words)
    return cache[n]


def compare_node(raw_node, original_node, unit_id, counts, cache=None):
    """Internal node seam. Production uses diagnose so W and raw share one packet."""
    cache = {} if cache is None else cache
    counts["nodes"] += 1
    r = original_node.get("received_frame")
    original = deepcopy(original_node.get("W"))
    result = dict(received_frame=r, original_W=original, original_f42_status=original_node.get("status"),
                  conditional_candidates=deepcopy(original), conditional_position=None,
                  comparison_performed=False, candidate_comparisons=[], shared_observation=None,
                  formal_output=None, presence_claim=False)
    if not original_node.get("evaluable") or not original_node.get("relation_complete") or original is None:
        return dict(result, status="ORIGINAL_UNRESOLVED", reason="Original F42 is not complete/evaluable")
    if not isinstance(original, list) or any(type(n) is not int or not 0 <= n < packet_code.N for n in original) or len(set(original)) != len(original):
        return dict(result, status="COMPARISON_UNRESOLVED", reason="Invalid original W; no replacement candidate set")
    if not original:
        return dict(result, status="UNCHANGED_EMPTY", reason="Original F42 empty handling retained")
    if len(original) == 1:
        return dict(result, conditional_position=original[0], status="UNCHANGED_SINGLETON",
                    reason="Original F42 singleton retained without a new distance/support gate")
    comparisons = [support(n) for n in original]
    result["candidate_comparisons"] = comparisons
    if not all(c["complete_630"] for c in comparisons):
        return dict(result, status="COMPARISON_UNRESOLVED",
                    reason="Candidate support is not the same complete 630 coordinates; all candidates retained")
    try:
        context = shared_hard_context(raw_node, unit_id, r)
    except (ValueError, TypeError, KeyError) as error:
        return dict(result, status="COMPARISON_UNRESOLVED", reason=str(error))
    result["shared_observation"] = {k: v for k, v in context.items() if k != "slots"}
    result["shared_observation"]["physical_slots"] = [dict(lane=l, role=role) for l, role in SLOTS]
    for candidate in comparisons:
        codewords = template(candidate["candidate_n"], cache, counts)
        distance = 0
        for observation in context["slots"]:
            distance += sum(a != b for a, b in zip(observation["bits"], codewords[observation["lane"]]))
        candidate["distance"] = distance
        counts["compared_candidates"] += 1
        counts["hard_bit_comparisons"] += 630
    counts["comparable_nodes"] += 1
    result["comparison_performed"] = True
    minimum = min(c["distance"] for c in comparisons)
    winners = [c["candidate_n"] for c in comparisons if c["distance"] == minimum]
    if len(winners) != 1:
        counts["ties"] += 1
        return dict(result, status="COMPARISON_TIE", reason="Integer minimum is tied; entire original W retained")
    counts["strict_winners"] += 1
    return dict(result, status="CONDITIONAL_STRICT_MINIMUM", conditional_candidates=winners,
                conditional_position=winners[0], reason="Unique strictly smallest integer Hamming within original W")


def conditional_edge(left, right, r, counts):
    a, b = left["conditional_candidates"], right["conditional_candidates"]
    complete = all(n["status"] != "ORIGINAL_UNRESOLVED" for n in (left, right)) and a is not None and b is not None
    pairs = []
    if complete:
        for x in a:
            for y in b:
                counts["conditional_edge_pairs"] += 1
                pairs.append(dict(left=x, right=y, delta=y-x, relation=f42.relation(x, y)))
    exact = complete and len(a) == len(b) == 1
    return dict(received_edge=r, status="CONDITIONAL_EXACT_" + pairs[0]["relation"] if exact
                else "CONDITIONAL_LIST" if pairs else "NO_ENDPOINT_PAIR" if complete else "ORIGINAL_UNRESOLVED",
                left_candidates=deepcopy(a), right_candidates=deepcopy(b), pairs=pairs, exact_singleton_pair=exact,
                left_endpoint_origin=left["status"], right_endpoint_origin=right["status"], formal_output=None,
                presence_claim=False)


def diagnose(packet, unit_id, check=lambda: None):
    """Fixed F42 and new conditional result from one packet, without mutating either."""
    if not isinstance(packet, dict) or not isinstance(packet.get("nodes"), list) or not packet["nodes"]:
        raise ValueError("Nonempty original received packet required")
    if "unit_id" in packet and packet["unit_id"] != unit_id:
        raise ValueError("Packet unit address mismatch")
    original_counts, counts = f42.counters(), counters()
    original = f42.diagnose(packet, len(packet["nodes"]), original_counts, check)
    nodes, cache = [], {}
    for raw, old in zip(packet["nodes"], original["nodes"]):
        check()
        nodes.append(compare_node(raw, old, unit_id, counts, cache))
    edges = [conditional_edge(a, b, r, counts) for r, (a, b) in enumerate(zip(nodes, nodes[1:]))]
    return dict(unit_id=unit_id, original_f42=original, original_f42_counts=original_counts,
                conditional_nodes=nodes, conditional_edges=edges, counters=counts,
                formal_output=None, presence_claim=False)
