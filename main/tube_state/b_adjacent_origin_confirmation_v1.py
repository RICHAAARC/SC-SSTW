"""Three-reader composition; frozen rank domains and original CDC full-C unchanged."""
from copy import deepcopy
from main.tube_state import b_rank3r_adjacent_origin_v1 as adj
from main.tube_state import b_rank3_four_source_confirmation_v1 as original
from main.tube_state import b_m3_time_path_v1 as geometry
from main.tube_state import b_cdc32_aggregation_v1 as cdc
READERS=('ADJ','RANK3R','CDC_M')
TAU=0.49221603782894896

def plan_task(key_id,families,views):
    """Only public key and anonymous observations; no source/arm/fixture labels."""
    corrected=adj.plan_task(key_id,families,views)
    base=deepcopy(corrected['original_selection'])
    record=dict(selection=base,original_selection=deepcopy(base),membership_checks=0,
        correction=dict(entered=False,status='ORIGINAL_BRANCH_PRESERVED',selected_window=None))
    original_rank=adj.plan_task(key_id,families,views,record)
    ranked={'ADJ':corrected,'RANK3R':original_rank}
    selection=original.select(families,'CDC_M');plans=[]
    for window,w in geometry.WINDOWS.items():
        ids=views[window][0]
        raw=dict(status='SCORES_COMPLETE' if not families[window]['issues'] else 'SCORE_UNRESOLVED',
            candidates=families[window]['members'],input_id=ids['input_id'],key_id=key_id)
        local=geometry.prepare_local(raw,dict(status='FROZEN',tau=TAU),ids['query_id'],w)
        plan=cdc.plan_local(local,views[window]);plan['window']=window;plans.append(plan)
    ranked['CDC_M']=dict(selection=selection,locals=cdc.reconcile_plans(plans),message_used=False,physical_calls=0)
    return {reader:ranked[reader] for reader in READERS}
