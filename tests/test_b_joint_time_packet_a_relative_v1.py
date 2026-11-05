"""Artificial hard-bit and graph fixtures only; no saved research data."""
from copy import deepcopy
import inspect
import unittest

from main.tube_state import b_joint_time_packet_a_relative_v1 as method
from main.tube_state import b_joint_time_packet_a_v1 as code
from main.tube_state import b_joint_time_packet_a_relation_v1 as f42


def raw_node(n=11, r=0):
    observations = []
    for lane in range(5):
        j = n - ((n-lane) % 5)
        for role in method.ROLES:
            observations.append(dict(lane=lane, role=role, physical_copy=True,
                raw_bits=code.encode_raw24(code.message(j, n-j)), known=[True]*63,
                status="UNRESOLVED_CODE", msg=None, candidate_n=None))
    return dict(received_frame=r, observations=observations)


def original(candidates, r=0):
    return dict(received_frame=r, W=candidates, status="ARTIFICIAL_F42", evaluable=True, relation_complete=True)


class RelativeChecks(unittest.TestCase):
    def compare(self, raw, values):
        counts = method.counters()
        return method.compare_node(raw, original(values), "artificial/NEW/FULL", counts), counts

    def test_strict_all_candidates_and_uncorrected_physical_words(self):
        raw = raw_node()
        old = original([12, 11, 13])
        before = deepcopy((raw, old))
        counts = method.counters()
        result = method.compare_node(raw, old, "unit", counts)
        self.assertEqual(result["conditional_position"], 11)
        self.assertEqual(result["conditional_candidates"], [11])
        self.assertEqual(counts["hard_bit_comparisons"], 1890)
        self.assertEqual([c["candidate_n"] for c in result["candidate_comparisons"]], [12, 11, 13])
        self.assertEqual((raw, old), before)
        self.assertFalse(result["presence_claim"])
        self.assertIsNone(result["formal_output"])
        # Saved decoder status and corrected positions never replace the hard observation.
        raw["observations"][0].update(status="INVALID_PACKET", corrected_positions=list(range(63)), corrected_bits=[1]*63)
        self.assertEqual(self.compare(raw, [11, 12])[0]["conditional_position"], 11)

    def test_integer_tie_retains_entire_original_list(self):
        raw, other = raw_node(11), raw_node(12)
        for a, b in zip(raw["observations"], other["observations"]):
            if a["role"] == "R1":
                a["raw_bits"] = b["raw_bits"]
        result, counts = self.compare(raw, [12, 11])
        self.assertEqual(result["status"], "COMPARISON_TIE")
        self.assertEqual(result["conditional_candidates"], [12, 11])
        self.assertIsNone(result["conditional_position"])
        self.assertEqual(counts["compared_candidates"], 2)

    def test_shared_coordinates_order_and_physical_copy_boundaries(self):
        raw = raw_node()
        # Canonical reconstruction from legal reordered records, not writer flatten order.
        raw["observations"] = sorted(raw["observations"], key=lambda o: (o["role"], -o["lane"]))
        context = method.shared_hard_context(raw, "unit", 0)
        self.assertEqual([(o["lane"], o["role"]) for o in context["slots"]], list(method.SLOTS))
        result, _ = self.compare(raw, [11, 12])
        self.assertEqual(result["candidate_comparisons"][0]["distance"], 0)
        # Reverse one word only: a known non-palindromic coordinate corruption changes E.
        target = next(o for o in raw["observations"] if o["raw_bits"] != o["raw_bits"][::-1])
        target["raw_bits"] = target["raw_bits"][::-1]
        self.assertGreater(self.compare(raw, [11, 12])[0]["candidate_comparisons"][0]["distance"], 0)
        changes = [lambda x: x["observations"].append(deepcopy(x["observations"][0])),
                   lambda x: x["observations"].pop(),
                   lambda x: x["observations"][0].update(received_frame=1),
                   lambda x: x["observations"][0].update(unit_id="another"),
                   lambda x: x["observations"][0].update(role="F")]
        for change in changes:
            value = raw_node(); change(value)
            result, counts = self.compare(value, [11, 12])
            self.assertEqual(result["status"], "COMPARISON_UNRESOLVED")
            self.assertEqual(result["conditional_candidates"], [11, 12])
            self.assertEqual(counts["hard_bit_comparisons"], 0)

    def test_all_known_binary_startup_and_passthrough(self):
        for field, value in [("known", False), ("raw_bits", float("nan")), ("raw_bits", 2), ("raw_bits", True)]:
            raw = raw_node(); raw["observations"][4][field][17] = value
            self.assertEqual(self.compare(raw, [11, 12])[0]["status"], "COMPARISON_UNRESOLVED")
        result, counts = self.compare(raw_node(), [3, 4, 11])
        self.assertEqual(result["conditional_candidates"], [3, 4, 11])
        self.assertEqual([r["support_bits"] for r in result["candidate_comparisons"]], [504, 630, 630])
        self.assertEqual(counts["template_words_encoded"], 0)
        for values, status in [([], "UNCHANGED_EMPTY"), ([0], "UNCHANGED_SINGLETON"), (None, "ORIGINAL_UNRESOLVED")]:
            result, counts = self.compare(None, values)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["conditional_candidates"], values)
            self.assertEqual(counts["hard_bit_comparisons"], 0)

    def test_public_template_bit_order_boundaries_without_full_domain(self):
        for n in (4, 11, 927, code.N-1):
            counts = method.counters()
            templates = method.template(n, {}, counts)
            for lane, bits in enumerate(templates):
                j = n - ((n-lane) % 5)
                self.assertEqual(list(bits), code.encode_raw24(8*j+n-j))
                word = code.bits_to_word(list(bits))
                self.assertEqual(list(bits), [(word >> bit) & 1 for bit in range(62, -1, -1)])
            self.assertEqual(counts["template_words_encoded"], 5)

    def test_same_packet_f42_preserved_and_condition_edges_have_no_order_filter(self):
        raw = raw_node()
        for obs in raw["observations"]:
            n = 11 if obs["role"] == "R0" else 12
            j = n - ((n-obs["lane"]) % 5)
            obs.update(status="LEGAL_LOCAL_CANDIDATE", msg=8*j+n-j, candidate_n=n)
        packet = dict(coding="NEW", nodes=[raw])
        before = deepcopy(packet)
        expected = f42.diagnose(packet, 1, f42.counters())
        value = method.diagnose(packet, "unit")
        self.assertEqual(value["original_f42"], expected)
        self.assertEqual(packet, before)
        self.assertEqual(value["original_f42"]["nodes"][0]["W"], [11, 12])
        self.assertEqual(value["conditional_nodes"][0]["conditional_position"], 11)
        counts = method.counters()
        nodes = [method.compare_node(None, original([n], i), "unit", counts) for i, n in enumerate([10, 10, 11, 14, 12])]
        edges = [method.conditional_edge(a, b, r, counts) for r, (a, b) in enumerate(zip(nodes, nodes[1:]))]
        self.assertEqual([e["pairs"][0]["relation"] for e in edges], ["REPEAT", "NORMAL", "GAP", "INVERSION"])
        ambiguous = dict(nodes[0], conditional_candidates=[10, 11], conditional_position=None, status="COMPARISON_TIE")
        self.assertEqual(len(method.conditional_edge(ambiguous, ambiguous, 0, counts)["pairs"]), 4)
        self.assertNotIn("truth", inspect.signature(method.diagnose).parameters)


if __name__ == "__main__":
    unittest.main()
