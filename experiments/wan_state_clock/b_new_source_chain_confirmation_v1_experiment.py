"""One consumed c02 A/B -> time -> payload chain; both posts follow whole blind save."""
from pathlib import Path
import json
import time
import sys
import threading
from . import b_new_source_chain_confirmation_v1_records as records


class StageMeasurements:
    """Observational sampling only; each inherited stage still owns its real limits."""
    def __init__(self):
        self.values={};self.stop=threading.Event();self.thread=None;self.process=None
        try:
            import psutil
            self.process=psutil.Process()
        except Exception as error:self.values['measurement_error']=str(error)
    def sample(self):
        try:
            if self.process:
                rss=self.process.memory_info().rss
                self.values['peak_observed_resident_bytes']=max(rss,self.values.get('peak_observed_resident_bytes',0))
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                allocated=torch.cuda.memory_allocated()
                self.values['peak_observed_cuda_allocated_bytes']=max(allocated,self.values.get('peak_observed_cuda_allocated_bytes',0))
        except Exception as error:self.values['measurement_error']=str(error)
    def poll(self):
        while not self.stop.wait(.1):self.sample()
    def __enter__(self):
        self.sample();self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=1);self.sample()
        self.values.update(scope='100ms observed process RSS/CUDA allocation; sampling can miss transients; native measurements retained',
                           enforcement='inherited stage only; this sampler imposes no resource gate')


def run(config,output=None,*,backend_factories=None,receiver_backend_factory=None,payload_backend_factory=None,
        phase_runner=None,receiver_runner=None,payload_runner=None,time_evaluator=None,payload_evaluator=None):
    output=Path(output or config['output']).expanduser().resolve()
    if records.output_state(output)=='CONSUMED':return records.recover(output)
    if records.output_state(output)=='CONFLICT':return dict(status='OUTPUT_CONFLICT',physical_calls_this_invocation=0)
    check=records.preflight(config,output)
    if check['status']!='PREFLIGHT_READY':return check
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED_WHOLE_CHAIN',source_id='b2-m3-c02',source=str(config['source']),
                plan=records.plan(),budgets=records.budgets(),denominators=records.denominators(),
                recovery_layout=records.recovery_layout(),rerun='READ_ONLY_ALL_PHASES_AND_POSTS'),stream,indent=2,allow_nan=False)
    except FileExistsError:return records.recover(output)
    journal=records.Journal(output);results={};errors={};start=time.monotonic();interrupted=False;measurements={}
    def measured(name,fn):
        monitor=StageMeasurements()
        try:
            with monitor:return journal.call(name,fn)
        finally:
            measurements[name]=monitor.values
            records.dump(output/'stage_measurements.json',measurements)
    try:
        from .b_conditional_time_validation_v1_experiment import run as chain_run
        if phase_runner is None:
            from .b_minimal_l_materialization_v1_experiment import run_stage as phase_runner
        if receiver_runner is None:
            from .b_conditional_time_output_v1_experiment import run as receiver_runner
        if payload_runner is None:
            from .b_time_payload_cdc32_v1_experiment import run as payload_runner
        def material(source,folder,stage,**kwargs):
            return measured(stage,lambda:phase_runner(source,folder,stage,**kwargs))
        def receiver(*args,**kwargs):
            return measured('time',lambda:receiver_runner(*args,**kwargs,defer_post=True))
        results['time_chain']=chain_run(config,output/'time_chain',backend_factories=backend_factories,
            receiver_backend_factory=receiver_backend_factory,phase_runner=material,receiver_runner=receiver)
        if results['time_chain']['status']=='PACKAGE_INTERRUPTED':raise KeyboardInterrupt('time chain interrupted')
        if records.operation_not_executed(journal.rows['time']):
            errors['dependency']='time receiver was not entered; see earliest saved stage error'
        else:
            terminal=records.load(output/'time_chain/receiver/blind_terminal.json')
            if terminal.get('status')!='ALL_BLIND_ROWS_TERMINAL':raise ValueError('time blind terminal unavailable')
            results['payload']=measured('payload',lambda:payload_runner(dict(source=str(output/'time_chain'),
                output=str(output/'payload'),model_id=config['model_id']),backend_factory=payload_backend_factory,defer_post=True))
            if results['payload'].get('status')=='INTERRUPTED':interrupted=True
    except BaseException as error:
        interrupted |= not isinstance(error,Exception);errors['execution']=f'{type(error).__name__}: {error}'
    journal.terminalize(('A','B','time','payload'))
    blind_saved=False;time_rows={};time_evaluation=payload_evaluation=None;recovered={};counts={}
    try:
        recovered=records.recover(output)
        # Only confirmed unentered time phases may supply a NOT_STARTED fallback.
        # A called receiver with missing receipts stays UNKNOWN.
        fallback=records.chain.receiver.plan('media') if records.operation_not_executed(journal.rows['time']) else {}
        time_rows=recovered['stages']['time'].get('state',fallback)
        stage_budgets=dict(A=records.budgets()['materialization_A'],B=records.budgets()['materialization_B'],
                           time=records.budgets()['time'],payload=records.budgets()['payload'])
        counts={k:v.get('counts',dict(status='NOT_EXECUTED' if journal.rows[k]['started']==0 else 'UNRESOLVED_RECEIPT',
            operations={kind:dict(planned=cap,started=0 if journal.rows[k]['started']==0 else None,
                                  returned=0 if journal.rows[k]['started']==0 else None) for kind,cap in stage_budgets[k].items()}))
            for k,v in recovered['stages'].items()}
        resources={k:dict(phase=journal.rows[k],native=v.get('resources',{}),observed=measurements.get(k,dict(status='NOT_EXECUTED')))
                   for k,v in recovered['stages'].items()}
        records.dump(output/'counts.json',counts)
        records.dump(output/'blind_terminal.json',dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL',
            phases={k:journal.rows[k] for k in ('A','B','time','payload')},time_rows=time_rows,
            recovered_artifacts=recovered['recovered_artifacts'],denominators=records.denominators(),errors=dict(errors)))
        blind_saved=True
    except BaseException as error:
        interrupted |= not isinstance(error,Exception);errors['blind_save']=f'{type(error).__name__}: {error}';resources={}
    evaluation=dict(status='EVIDENCE_INCOMPLETE')
    if blind_saved:
        from . import b_new_source_chain_confirmation_v1_evaluation as post
        truth=output/'time_chain/receiver/private/attack_truth.json'
        def time_post():
            nonlocal time_evaluation
            evaluator=time_evaluator
            if evaluator is None:
                from .b_conditional_time_output_v1_evaluation import evaluate as evaluator
            value=evaluator(output/'time_chain/receiver',time_rows,truth)
            # A later receipt-write failure cannot erase an already observed counterexample.
            time_evaluation=value
            records.dump(output/'time_evaluation.json',value);return value
        def payload_post():
            nonlocal payload_evaluation
            terminal=records.load(output/'payload/blind_terminal.json')
            if terminal.get('status')!='ALL_BLIND_SLOTS_TERMINAL':raise ValueError('payload blind terminal unavailable')
            fixture=post.private_message(config['source'],output)
            evaluator=payload_evaluator
            if evaluator is None:
                from .b_time_payload_cdc32_v1_evaluation import evaluate as evaluator
            value=evaluator(output/'payload',truth,message_bits=fixture['message_bits'],message_source=fixture['message_source'])
            payload_evaluation=value
            records.dump(output/'payload_evaluation.json',value);return value
        if records.operation_not_executed(journal.rows['time']):
            journal.terminalize(('time_post',))
        else:
            try:time_evaluation=journal.call('time_post',time_post)
            except BaseException as error:
                errors['time_post']=f'{type(error).__name__}: {error}'
                interrupted |= not isinstance(error,Exception)
        # Physical interruption has already terminalized the blind package. Post itself never resumes.
        if records.operation_not_executed(journal.rows['payload']):
            journal.terminalize(('payload_post',))
        elif journal.rows['time_post']['status']!='INTERRUPTED':
            try:payload_evaluation=journal.call('payload_post',payload_post)
            except BaseException as error:
                errors['payload_post']=f'{type(error).__name__}: {error}';interrupted |= not isinstance(error,Exception)
        try:
            def summary():
                nonlocal evaluation
                evaluation=post.summarize(time_evaluation,payload_evaluation,time_rows,results,errors)
                records.dump(output/'evaluation.json',evaluation);return evaluation
            evaluation=journal.call('summary',summary)
        except BaseException as error:
            errors['summary']=f'{type(error).__name__}: {error}';interrupted |= not isinstance(error,Exception)
            evaluation.update(evidence_complete=False,errors=dict(errors))
            if evaluation.get('status')!='COUNTEREXAMPLE':evaluation['status']='EVIDENCE_INCOMPLETE'
    journal.terminalize()
    resources.update(total_wall_seconds=time.monotonic()-start,scope='native per-stage limits; no full-chain payload limit',
                     post_phases={k:journal.rows[k] for k in ('time_post','payload_post','summary')})
    records.dump(output/'resources.json',resources)
    result=dict(status='PACKAGE_INTERRUPTED' if interrupted else 'PACKAGE_FIXED_ATTEMPT_TERMINATED',source_id='b2-m3-c02',
        evaluation_status=evaluation['status'],phases=journal.rows,stage_results=results,errors=errors,
        known_counterexamples=evaluation.get('counterexamples',[]),
        denominators=records.denominators(),budgets=records.budgets(),wall_seconds=time.monotonic()-start,
        scientific_pass=None,blind_terminal_saved=blind_saved)
    result['execution_diagnostics']=records.execution_diagnostics(result,counts,recovered.get('stages'))
    records.dump(output/'result.json',result);return result
