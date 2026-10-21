"""Card88 reviewed arithmetic reused by Card89; new tables only, never replay saved280."""
from collections import Counter
import math

STARTS=tuple(range(2,36))
VARIANTS=("raw_pooled","centered_pooled","centered_min")

def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x)


def candidate(s):
    cells = [dict(local_row=r, source_cell=s+r-1, group=(s+r-2)//11) for r in range(1, 11)]
    return dict(source_start=s, status='NOT_ENTERED', hypothesis_only=True, cells=cells,
        source_occupancy=[dict(source_cell=j, local_rows=[x['local_row'] for x in cells if x['source_cell'] == j]) for j in range(1, 45)],
        decoded_bits=[None]*32, segments=[dict(group=g, bits=[dict(slot=8*g+b,
            status='NOT_ENTERED', decoded=None, count=0, ones=None, tie=None,
            zero_coefficients=None) for b in range(8)]) for g in range(4)])


def blank_table(starts=STARTS):
    return dict(status='NOT_ENTERED', truth_inputs=False, message_inputs=False,
        centering=None, scores=[dict(source_start=s, status='NOT_ENTERED', denominator=10,
            coordinates=[dict(local_row=r, source_column=s+r-1) for r in range(1, 11)],
            F=None, V=None, raw_pooled=None, centered_pooled=None, centered_min=None,
            raw_terms=None, F_terms=None, V_terms=None) for s in starts],
        maxima={v:dict(status='NOT_ENTERED', exact_max_ties=None, maximum=None,
            provisional_start=None, authenticated=False) for v in VARIANTS},
        candidates=[candidate(s) for s in starts])


def score(q, out, counters):
    values = q['roi_q']
    row_means = [[math.fsum(values[r][k])/44 for k in range(4)] for r in range(10)]
    column_means = [[math.fsum(values[r][k][j] for r in range(10))/10 for j in range(44)] for k in range(4)]
    grand = [math.fsum(values[r][k][j] for r in range(10) for j in range(44))/440 for k in range(4)]
    centered = [[[values[r][k][j]-row_means[r][k]-column_means[k][j]+grand[k]
                  for j in range(44)] for k in range(4)] for r in range(10)]
    if any(not finite_number(x) for row in centered for roi in row for x in roi):
        raise ValueError('nonfinite centered Q')
    out['centering'] = dict(row_means=row_means, column_means=column_means,
        grand_means=grand, centered_q=centered, truth_inputs=False)
    counters['centered_values'] += 1760
    for item in out['scores']:
        s = item['source_start']
        item['status'] = 'STARTED'
        raw = [q['pooled_q'][r][s+r-1] for r in range(10)]
        ft = [(centered[r][0][s+r-1]+centered[r][1][s+r-1])/2 for r in range(10)]
        vt = [(centered[r][2][s+r-1]+centered[r][3][s+r-1])/2 for r in range(10)]
        f, v, original = math.fsum(ft)/10, math.fsum(vt)/10, math.fsum(raw)/10
        counters['half_means'] += 2
        counters['half_terms'] += 20
        counters['raw_means'] += 1
        counters['raw_terms'] += 10
        avg, minimum = (f+v)/2, min(f, v)
        if any(not finite_number(x) for x in (f, v, original, avg, minimum)):
            raise ValueError('nonfinite score')
        item.update(status='RETURNED', F=f, V=v, raw_pooled=original,
            centered_pooled=avg, centered_min=minimum, raw_terms=raw, F_terms=ft, V_terms=vt)
        counters['derived_centered_scores'] += 1
        counters['min_statistics'] += 1
    for variant in VARIANTS:
        maximum = max(row[variant] for row in out['scores'])
        ties = [row['source_start'] for row in out['scores'] if row[variant] == maximum]
        out['maxima'][variant] = dict(status='COMPLETE', maximum=maximum,
            exact_max_ties=ties, provisional_start=ties[0] if len(ties) == 1 else None,
            authenticated=False, identity_status='UNCALIBRATED', phase_status='UNCALIBRATED')


def reduce_all(votes, out, counters):
    signed, zeros = votes['signed_votes'], votes['zero_coefficient_mask']
    for item in out['candidates']:
        item['status'] = 'STARTED'
        counters['candidate_reducers_started'] += 1
        for g, segment in enumerate(item['segments']):
            eligible = [x['local_row']-1 for x in item['cells'] if x['group'] == g]
            segment.update(local_rows=[r+1 for r in eligible], observed_cells=len(eligible))
            for b, row in enumerate(segment['bits']):
                row['status'] = 'STARTED'
                seq = [int(signed[c][r][k] > 0) for c in range(4) for r in eligible for k in range(b, 240, 8)]
                counters['candidate_vote_terms'] += len(seq)
                if seq:
                    counters['group_bit_counters_started'] += 1
                    value = Counter(seq).most_common(1)[0][0]
                    counters['group_bit_counters_returned'] += 1
                    item['decoded_bits'][8*g+b] = value
                    row.update(status='UNCALIBRATED_CONDITIONAL_READ', decoded=value,
                        count=len(seq), ones=sum(seq), tie=sum(seq)*2 == len(seq),
                        zero_coefficients=sum(zeros[c][r][k] for c in range(4) for r in eligible for k in range(b, 240, 8)))
                else:
                    row.update(status='UNREAD_LOCATION', decoded=None, count=0, ones=0, tie=None, zero_coefficients=0)
                counters['candidate_message_rows_returned'] += 1
        item['status'] = 'CONDITIONAL_REDUCED_UNCALIBRATED'
        counters['candidate_reducers_returned'] += 1


def mark_unfinished(out, section, status, root):
    if section == 'score':
        for row in out['scores']:
            if row['status'] in ('NOT_ENTERED', 'STARTED'):
                row.update(status=status, unavailable_from=root)
        for obj in out['maxima'].values():
            if obj['status'] != 'COMPLETE':
                obj.update(status=status, unavailable_from=root)
    else:
        for row in out['candidates']:
            if row['status'] in ('NOT_ENTERED', 'STARTED'):
                row.update(status=status, unavailable_from=root)
                for segment in row['segments']:
                    for bit in segment['bits']:
                        if bit['status'] in ('NOT_ENTERED', 'STARTED'):
                            bit.update(status=status, unavailable_from=root)
