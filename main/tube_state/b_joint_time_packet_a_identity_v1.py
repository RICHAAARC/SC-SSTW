"""Fixed raw identity addresses, with no cell selection or aggregation."""


def empty(unit,reason):
    return [dict(source=unit['source'],arm=unit['arm'],view=unit['view'],query=unit['query'],phase=unit['phase'],
                 region=region,row=row,bit=bit,value=None,status='UNRESOLVED',cause=reason,raw=None)
            for region in ('R0','R1') for row in range(1,11) for bit in range(8)]


def addressed(decoded,unit,expected_id):
    """Preserve original decoder votes/ties. Validate cache address, never re-decode."""
    if decoded.get('key')!='key00':raise ValueError('original key00 identity cache required')
    rows=empty(unit,'MISSING_RAW_RECORD');lookup={(x['region'],x['row'],x['bit']):x for x in rows}
    seen=set()
    for region in ('R0','R1'):
        cells=decoded['regions'][region]['local_cells']
        if len(cells)!=10:raise ValueError('complete ten-row DCT2 cache required')
        for cell in cells:
            row=cell['local_row'];eid=f'{expected_id}/row{row}/{region}'
            if type(row) is not int or row not in range(1,11) or cell['evidence_id']!=eid:
                raise ValueError('old/new identity cache does not match fixed actual window address')
            if len(cell['bits'])!=8:raise ValueError('complete eight-bit raw cell required')
            for bit in cell['bits']:
                key=(region,row,bit['local_bit'])
                if key not in lookup or key in seen or bit.get('evidence_id')!=eid:raise ValueError('invalid or repeated raw bit address')
                seen.add(key);value=bit['value']
                if value is not None and (type(value) is not int or value not in (0,1)):raise ValueError('invalid raw identity bit')
                if bit.get('planned_count')!=8:raise ValueError('original eight coefficient votes required')
                lookup[key].update(value=value,status=bit['status'],cause=bit.get('cause_set',[]),raw=bit)
    if len(seen)!=160:raise ValueError('fixed 160 raw identity slots required')
    return rows
