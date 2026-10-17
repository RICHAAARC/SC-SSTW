"""Post-read evaluation only. Never imported in the blind worker."""
import json
from pathlib import Path
from experiments.paper_results_v1.single_read_v1 import SLOT, write_json


def evaluate_saved(output):
    output = Path(output)
    state = json.loads((output / 'state.json').read_text())
    row = dict(slot_id=SLOT, status=state['status'], planned_rows=1, planned_segments=4,
               planned_bits=32, structural_missing_bits=16, eligible_planned_bits=16,
               scored_present_bits=0, missing_present_bits=16, wrong_present_bits=None,
               false_attribution_bits=0, observed_BER=None, complete_present_BER=None,
               path_exact=False, interval_exact=True, absolute_path_error=1,
               retained_path_difference=dict(received_index=8, true_position=45, estimated_position=44),
               error=state.get('error'), evaluation_basis='new persisted single-row decoded output only')
    result_path = output / 'read_result.json'
    if result_path.is_file():
        # A persisted blind result must exist before accessing any expected message.
        result = json.loads(result_path.read_text())
        if result.get('persisted_before_evaluation') is not True or result.get('slot_id') != SLOT:
            raise ValueError('saved single-row reader result required before evaluation')
        bits = result['decoded_bits']
        if (len(bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in bits[:16])
                or bits[16:] != [None] * 16):
            raise ValueError('saved fixed 16-present/16-missing result invalid')
        expected = [int(b) for b in f'{0xA6D39C5E:032b}']
        wrong = sum(a != b for a, b in zip(bits[:16], expected[:16]))
        row.update(status='READ', scored_present_bits=16, missing_present_bits=0,
                   wrong_present_bits=wrong, observed_BER=wrong / 16, complete_present_BER=wrong / 16,
                   decoded_bits=bits, truth_join_after_saved_read=True)
    summary = dict(status=row['status'], row=row, stages=state['stages'], chosen_input=state['chosen_input'],
                   original_P2='196 rows / 6272 bits retained without edits',
                   original_component='64 rows / 2048 bits retained without edits',
                   RAW_optional_overlay=dict(fixed_rows=32, planned_bits=1024, present_bits=640,
                       structural_missing_bits=384, old_known_bits=624, old_wrong_bits=0,
                       new_known_bits=row['scored_present_bits'], new_wrong_bits=row['wrong_present_bits'],
                       complete_present_BER=(row['wrong_present_bits'] / 640 if row['status'] == 'READ' else None)),
                   evidence_ceiling='Posthoc 16-bit gap only; path error1 remains; no independent confirmation or default change',
                   no_continuation=True)
    write_json(output / 'row_result.json', row)
    write_json(output / 'overlay.json', dict(original_slot=SLOT, original_status='NO_SAVED_READ_MATCH',
                                          supplemental_record='row_result.json', original_records_modified=False))
    write_json(output / 'summary.json', summary)
    return summary
