"""Three-reader composition; frozen rank domains and original CDC full-C unchanged."""
from main.tube_state import b_rank3r_complete_repeat_v1 as rank
from main.tube_state import b_rank3_four_source_confirmation_v1 as original
from main.tube_state import b_m3_time_path_v1 as geometry
from main.tube_state import b_cdc32_aggregation_v1 as cdc
READERS=('RANK3R','RANK3','CDC_M')
TAU=0.49221603782894896

def plan_task(key_id,families,views):
    """Only public key and anonymous observations; no source/arm/fixture labels."""
    ranked=rank.plan_task(key_id,families,views)
    selection=original.select(families,'CDC_M');plans=[]
    for window,w in geometry.WINDOWS.items():
        ids=views[window][0]
        raw=dict(status='SCORES_COMPLETE' if not families[window]['issues'] else 'SCORE_UNRESOLVED',
            candidates=families[window]['members'],input_id=ids['input_id'],key_id=key_id)
        local=geometry.prepare_local(raw,dict(status='FROZEN',tau=TAU),ids['query_id'],w)
        plan=cdc.plan_local(local,views[window]);plan['window']=window;plans.append(plan)
    ranked['CDC_M']=dict(selection=selection,locals=cdc.reconcile_plans(plans),message_used=False,physical_calls=0)
    return {reader:ranked[reader] for reader in READERS}
