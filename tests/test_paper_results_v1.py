"""Small stdlib saved-table tests; no scientific receiver execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.paper_results_v1.report import (
    LAYOUT, ReportFormatError, adapt_segmented_row, build_report,
    compare_saved_support, load_json, write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def slot(identity="s/v/a", **changes):
    return {"slot_id": identity, "saved_slot_id": identity, "binding_id": "fixture",
            "source_id": "s", "view": "v", "arm": "a", "message_id": "m",
            "information_condition": "BLIND", "evidence_role": "SYNTHETIC_NOT_A_RESULT", **changes}


def saved(identity="s/v/a", **changes):
    return {"slot_id": identity, "case_id": "s", "receiver": "a", "view": "v",
            "oracle": False, "planned_bits": 32, "planned_segments": 4,
            "status": "READ", "reason": None, "eligible_planned_bits": 24,
            "structural_missing_bits": 8, "scored_present_bits": 16,
            "missing_present_bits": 8, "wrong_present_bits": 1,
            "false_attribution_bits": 0, "decoded_bits": [0] * 16 + [None] * 16,
            "path_exact": False, "absolute_path_error": 2, **changes}


class Reports(unittest.TestCase):
    def report(self, rows, slots=None, **binding):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "rows.json").write_text(json.dumps(rows))
            protocol = {"schema": "segmented-saved-report-v1", "plan_status": "ADOPTED",
                        "payload_layout": LAYOUT, "slots": slots if slots is not None else [slot()],
                        "result_bindings": [{"binding_id": "fixture", "rows_path": "rows.json", **binding}]}
            return build_report(protocol, base_dir=path)

    def test_denominators_and_null(self):
        result = self.report([saved()])
        totals = result["summary"]
        self.assertEqual(totals["planned_bits"], 32)
        self.assertEqual(totals["structurally_present_bits"] + totals["structural_missing_bits"], 32)
        self.assertEqual(totals["read_present_bits"] + totals["present_unread_bits"], 24)
        self.assertEqual(totals["observed_ber"], 1 / 16)
        self.assertIsNone(totals["complete_present_ber"])
        self.assertEqual(result["general_source_cell_false_attribution"], "UNKNOWN")

    def test_failure_unresolved_unsupported_and_missing_retained(self):
        for state in ("FAILED", "UNRESOLVED", "UNSUPPORTED", "CONFLICT", "MISSING", "UNREAD"):
            result = self.report([saved(status=state, reason="specific reason", scored_present_bits=0,
                                        missing_present_bits=24, wrong_present_bits=0)])
            self.assertEqual(result["rows"][0]["status"], state)
            self.assertEqual(result["rows"][0]["saved_reason"], "specific reason")
            self.assertIsNone(result["summary"]["observed_ber"])
        result = self.report([])
        self.assertEqual(result["summary"]["planned_bits"], 32)
        self.assertEqual(result["rows"][0]["status"], "MISSING")
        self.assertIsNone(result["summary"]["read_present_bits"])

    def test_duplicate_unplanned_alias_and_no_automatic_roster(self):
        result = self.report([saved(), saved(), saved("extra")])
        self.assertEqual(result["summary"]["planned_rows"], 1)
        self.assertEqual(result["rows"][0]["status"], "CONFLICT")
        self.assertEqual(len(result["rows"][0]["raw_saved_rows"]), 2)
        self.assertEqual(len(result["unplanned_rows"]), 1)
        result = self.report([saved(physical_read_unit="unit", read_alias_of="elsewhere")])
        self.assertEqual(result["counts"]["saved_alias_rows"], 1)
        self.assertEqual(result["counts"]["physical_read_unit_labels"], 1)
        self.assertEqual(self.report([saved()], slots=[])["summary"]["planned_rows"], 0)

    def test_metadata_difference_and_optional_summary_do_not_gate(self):
        result = self.report([saved(source_sha="different")],
                             expected_top_level={"source_sha": "old", "content_sha256": "missing", "config_sha256": "old"},
                             summary_path="absent.json")
        self.assertEqual(result["rows"][0]["report_state"], "ADAPTED")
        self.assertTrue(result["inputs"][0]["summary_read_issue"])

    def test_real_format_numeric_identity_errors_are_not_science_failures(self):
        for changes in ({"scored_present_bits": 40}, {"eligible_planned_bits": 23},
                        {"wrong_present_bits": 17}, {"case_id": "wrong"}, {"oracle": True},
                        {"mapping_origin": "CALLER_SUPPLIED"},
                        {"decoded_bits": [1]}, {"absolute_path_error": -1}):
            result = self.report([saved(**changes)])
            row = result["rows"][0]
            self.assertEqual(row["saved_status"], "READ")
            self.assertEqual(row["report_state"], "INVALID_SAVED_FORMAT")
            self.assertIn("not a scientific failure", row["report_reason"])
            self.assertIsNone(result["summary"]["error_bits"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'x.json'
            for content in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
                path.write_text(content)
                with self.assertRaises(ReportFormatError):
                    load_json(path)

    def test_blind_oracle_caller_and_evidence_roles_are_separate(self):
        secret = {"expected_message": [1] * 32, "true_missing_segments": [3]}
        for condition, output in (("BLIND", "blind_output"), ("ORACLE", "oracle_output"), ("CALLER_SUPPLIED", "caller_map_output")):
            result = self.report([saved(oracle=condition == "ORACLE", **secret)], [slot(information_condition=condition)])
            row = result["rows"][0]
            self.assertIsNotNone(row[output])
            self.assertNotIn("expected_message", row[output])
            self.assertNotIn("true_missing_segments", row[output])
            self.assertEqual(row['posthoc']['true_missing_segments'], [3])
            self.assertEqual(row['posthoc']['expected_message'], [1] * 32)
            if condition != "BLIND":
                self.assertFalse(row[output]["blind_sync_performed"])
                self.assertIsNone(row["blind_output"])
        left = adapt_segmented_row(slot(), saved())
        right = copy.deepcopy(left)
        right["evidence_role"] = "CONFIRMATION"
        self.assertEqual(compare_saved_support(left, right)["status"], "INCOMPARABLE")

    def test_pair_requires_per_bit_support_and_preserves_coverage_losses(self):
        left = adapt_segmented_row(slot(), saved())
        right = copy.deepcopy(left)
        self.assertEqual(compare_saved_support(left, right)["status"], "UNKNOWN")
        a = [{"bit": i, "present": i < 24, "read": i < 16, "correct": i != 0 if i < 16 else None} for i in range(32)]
        b = [{"bit": i, "present": i < 24, "read": 8 <= i < 24, "correct": True if 8 <= i < 24 else None} for i in range(32)]
        right['metrics']['error_bits'] = 0
        right['blind_output']['decoded_bits'] = [None] * 8 + [0] * 16 + [None] * 8
        left["posthoc"]["bit_support"], right["posthoc"]["bit_support"] = a, b
        pair = compare_saved_support(left, right)
        self.assertEqual(pair["common_read_bits"], 8)
        self.assertEqual((pair["correct_coverage_retained"], pair["correct_coverage_lost"], pair["correct_coverage_added"]), (8, 7, 8))
        self.assertEqual(pair["left_common_ber"], 0)

    def test_partial_unknown_counts_cannot_hide_known_contradictions(self):
        base = saved(decoded_bits=None, true_missing_segments=None)
        conflicts = (
            dict(eligible_planned_bits=8, structural_missing_bits=24, scored_present_bits=16, missing_present_bits=None, wrong_present_bits=0),
            dict(eligible_planned_bits=8, structural_missing_bits=24, scored_present_bits=None, missing_present_bits=16, wrong_present_bits=0),
            dict(eligible_planned_bits=None, structural_missing_bits=24, scored_present_bits=16, missing_present_bits=None, wrong_present_bits=0),
            dict(eligible_planned_bits=8, structural_missing_bits=None, scored_present_bits=None, missing_present_bits=None, wrong_present_bits=9),
            dict(eligible_planned_bits=None, structural_missing_bits=None, scored_present_bits=None, missing_present_bits=25, wrong_present_bits=8),
        )
        for changes in conflicts:
            row = {**base, **changes}
            result = adapt_segmented_row(slot(), row)
            self.assertEqual(result['report_state'], 'INVALID_SAVED_FORMAT')
            self.assertEqual(result['saved_status'], 'READ')
            self.assertEqual(result['raw_saved_rows'], [row])
            self.assertIsNone(result['metrics'].get('observed_ber'))
        compatible = dict(base, eligible_planned_bits=24, structural_missing_bits=None,
                          scored_present_bits=16, missing_present_bits=None, wrong_present_bits=None)
        result = adapt_segmented_row(slot(), compatible)
        self.assertEqual(result['report_state'], 'ADAPTED')
        self.assertIsNone(result['metrics']['structural_missing_bits'])
        self.assertIsNone(result['metrics']['present_unread_bits'])
        self.assertIsNone(result['metrics']['error_bits'])
        self.assertIsNone(result['metrics']['observed_ber'])

    def test_support_conflicts_with_known_counts_or_masks_give_no_pair_score(self):
        support = [{'bit':i, 'present':True, 'read':True, 'correct':True} for i in range(32)]
        base = saved(eligible_planned_bits=32, structural_missing_bits=0, scored_present_bits=32,
                     missing_present_bits=0, wrong_present_bits=0, decoded_bits=None, bit_support=support)
        for changes in (dict(wrong_present_bits=32), dict(decoded_bits=[None]*32),
                        dict(true_missing_segments=[3], eligible_planned_bits=None, structural_missing_bits=None,
                             scored_present_bits=None, missing_present_bits=None, wrong_present_bits=None)):
            raw = {**base, **changes}
            row = adapt_segmented_row(slot(), raw)
            self.assertEqual(row['report_state'], 'INVALID_SAVED_FORMAT')
            self.assertEqual(row['raw_saved_rows'], [raw])
            pair = compare_saved_support(row, copy.deepcopy(row))
            self.assertEqual(pair['status'], 'INVALID_SAVED_FORMAT')
            self.assertNotIn('left_common_ber', pair)
        # Missing aggregate counts/masks are allowed; supporting records stay explicit.
        unknown = dict(base, **{source:None for source in ('eligible_planned_bits','structural_missing_bits',
                       'scored_present_bits','missing_present_bits','wrong_present_bits','false_attribution_bits')})
        row = adapt_segmented_row(slot(), unknown)
        self.assertEqual(row['report_state'], 'ADAPTED')
        self.assertIsNone(row['metrics']['error_bits'])
        self.assertEqual(compare_saved_support(row, copy.deepcopy(row))['left_common_ber'], 0)
        # Pair use also checks normalized records if altered after adaptation.
        row['metrics']['error_bits'] = 32
        self.assertEqual(compare_saved_support(row, copy.deepcopy(row))['status'], 'INVALID_SAVED_FORMAT')

    def test_invalid_physical_alias_labels_retain_fixed_rows_without_counting(self):
        for changes in (dict(physical_read_unit=[]), dict(read_alias_of={'fake':'alias'})):
            raw = saved(**changes)
            report = self.report([raw])
            self.assertEqual(report['summary']['planned_rows'], 1)
            self.assertEqual(report['rows'][0]['report_state'], 'INVALID_SAVED_FORMAT')
            self.assertEqual(report['rows'][0]['raw_saved_rows'], [raw])
            self.assertEqual(report['rows'][0]['saved_status'], 'READ')
            self.assertIsNone(report['counts']['saved_alias_rows'])
            self.assertEqual(report['counts']['physical_read_unit_labels'], 0)
        report = self.report([saved(physical_read_unit=None, read_alias_of=None)])
        self.assertEqual(report['rows'][0]['report_state'], 'ADAPTED')
        self.assertEqual(report['counts']['physical_read_unit_unknown_rows'], 1)
        self.assertEqual(report['counts']['saved_alias_rows'], 0)

    def test_adopted_plan_required_and_duplicate_plan_rejected(self):
        protocol = {"schema": "segmented-saved-report-v1", "plan_status": "PROPOSED",
                    "payload_layout": LAYOUT, "slots": [slot()],
                    "result_bindings": [{"binding_id": "fixture", "rows_path": "no.json"}]}
        with self.assertRaises(ReportFormatError):
            build_report(protocol)
        protocol["plan_status"] = "ADOPTED"
        protocol["slots"].append(slot())
        with self.assertRaises(ReportFormatError):
            build_report(protocol)

    def test_full_historical_ledger_conservation_and_outputs(self):
        for name, expected in (("original36", (36, 1152, 896, 256)), ("p2_196", (196, 6272, 3968, 2304))):
            directory = ROOT / 'docs/evidence/paper_results_v1' / name
            report = build_report(load_json(directory / 'protocol.json'), base_dir=directory)
            summary = report["summary"]
            self.assertEqual(tuple(summary[k] for k in ('planned_rows','planned_bits','structurally_present_bits','structural_missing_bits')), expected)
            self.assertEqual(summary["report_state_counts"], {"ADAPTED": expected[0]})
            originals = load_json(directory / 'row_results.json')
            self.assertEqual({r['slot_id']:r for r in originals}, {r['saved_slot_id']:r['raw_saved_rows'][0] for r in report['rows']})
            if name == 'p2_196':
                self.assertEqual(summary['status_counts'], {'READ':188,'UNRESOLVED':4,'UNSUPPORTED':4})
                self.assertEqual(report['counts']['physical_read_unit_labels'],98)
                self.assertEqual(report['counts']['logical_read_rows'],188)
                self.assertEqual(report['counts']['saved_alias_rows'],90)
            with tempfile.TemporaryDirectory() as output:
                files = write_report(report, output)
                self.assertEqual(len(files), 3)
                self.assertTrue(all(p.stat().st_size for p in files))
                with self.assertRaises(FileExistsError):
                    write_report(report, output)


if __name__ == '__main__':
    unittest.main()
