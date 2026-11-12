"""Read-only arithmetic over saved JSON/CSV; no receiver, model or query runner."""
from pathlib import Path
from collections import Counter
from main.tube_state import paper_evidence_supplement_v1 as p
from main.tube_state import paper_suite_v1 as prior
from main.tube_state.paper_fpr_claims_v1 import at_least, role
from . import paper_evidence_supplement_records_v1 as rec


def decision_counts(group, field):
    """Only JSON booleans are decisions; null, absent and invalid stay distinct."""
    states = Counter()
    for row in group:
        saved = row.get(field)
        if not isinstance(saved, dict) or 'decision' not in saved:
            states['MISSING'] += 1
        elif saved['decision'] is None:
            states['NULL'] += 1
        elif type(saved['decision']) is not bool:
            states['INVALID_TYPE'] += 1
        else:
            states['ACCEPT' if saved['decision'] else 'NONACCEPT'] += 1
    return dict(planned=len(group), accept=states['ACCEPT'], nonaccept=states['NONACCEPT'],
                unknown=sum(states[k] for k in ('MISSING', 'NULL', 'INVALID_TYPE')), states=dict(states))


def video_messages(rows, method):
    """Keep all sixty view rows, including methods with no video-word metric."""
    output = []
    for row in rows:
        payload = row.get('payload')
        value = payload.get('native_video_target_agreement') if isinstance(payload, dict) else None
        applicable = method in ('pixelseal', 'videoseal')
        valid = (isinstance(value, dict) and type(value.get('bits')) is int and value['bits'] > 0
                 and type(value.get('errors')) is int and 0 <= value['errors'] <= value['bits']
                 and type(value.get('full_target_message')) is bool
                 and value['full_target_message']==(value['errors']==0))
        output.append(dict(source=row['source'], view=row['view'], method=method,
                           status='SAVED' if valid else 'UNKNOWN' if applicable else 'N/A', saved=value))
    known = [r['saved'] for r in output if r['status'] == 'SAVED']
    bits = sum(r['bits'] for r in known); errors = sum(r['errors'] for r in known)
    return output, dict(planned_views=60, applicable=method in ('pixelseal', 'videoseal'),
        observed_views=len(known), unknown_views=sum(r['status']=='UNKNOWN' for r in output),
        not_applicable_views=sum(r['status']=='N/A' for r in output),
        complete_messages=sum(r['full_target_message'] for r in known), observed_bits=bits,
        error_bits=errors, observed_ber=errors/bits if bits else None,
        scope='saved native target-query agreement; views are not independent sources')


def payload_totals(rows,method):
    """BER uses valid paired numerator/denominator records, never imputed zeros."""
    capacity=32 if method=='wam' else 256
    keys=('observed_bits','error_bits','missing_bits','erased_bits','full_message_frames','full_owner_frames')
    totals={key:0 for key in keys};unknown=[];valid=0
    lengths=dict(zip(p.VIEWS,prior.LENGTHS))
    for row in rows:
        value=row.get('payload');n=lengths[row['view']];planned=n*capacity
        okay=isinstance(value,dict) and all(type(value.get(k)) is int for k in (*keys,'planned_bits','planned_frames'))
        okay=okay and value['planned_bits']==planned and value['planned_frames']==n
        okay=okay and 0<=value['error_bits']<=value['observed_bits']<=planned
        okay=okay and all(0<=value[k]<=planned for k in ('missing_bits','erased_bits'))
        okay=okay and all(0<=value[k]<=n for k in ('full_message_frames','full_owner_frames'))
        if not okay:
            unknown.append(dict(source=row['source'],view=row['view'],planned_bits=planned,
                                reason='missing, invalid or inconsistent saved payload counts'))
            continue
        valid+=1
        for key in keys: totals[key]+=value[key]
    totals.update(planned_frames=sum(prior.LENGTHS)*12,planned_bits=sum(prior.LENGTHS)*12*capacity,
        payload_records=valid,unknown_payload_records=len(unknown),unknown_payload_planned_bits=sum(r['planned_bits'] for r in unknown),
        unknown_records=unknown,observed_ber=totals['error_bits']/totals['observed_bits'] if totals['observed_bits'] else None,
        ber_scope='only saved records with valid paired error_bits and observed_bits; fixed planned denominator retained')
    return totals


def event_totals(saved_rows,planned):
    # The saved edge schema partitions NORMAL, GAP and REPEAT events.
    categories=('adjacent','gap','repeat')
    details=Counter();invalid=0
    for row in saved_rows:
        edges=row.get('edges');n=row.get('planned_edges')
        valid=isinstance(edges,dict) and type(n) is int and n>=0
        valid=valid and all(type(v) is int and v>=0 for v in edges.values())
        valid=valid and sum(edges.get(k+'_planned',0) for k in categories)==n
        valid=valid and all(sum(edges.get(k+'_'+suffix,0) for suffix in ('correct','incorrect','unknown'))==edges.get(k+'_planned',0) for k in categories)
        if valid: details.update(edges)
        else: invalid+=1
    observed=sum(details[k+'_planned'] for k in categories)
    return dict(details,planned=planned,observed_planned=observed,
        correct=sum(details[k+'_correct'] for k in categories),
        incorrect=sum(details[k+'_incorrect'] for k in categories),
        unknown=sum(details[k+'_unknown'] for k in categories),technical_unknown_planned=planned-observed,
        invalid_saved_records=invalid)


LIMITS='''# Evidence scope and interpretation

These are twelve fixed Wan sources and five existing temporal edits. Views, arms and keys are correlated observations, not independent sources. CROP is a temporal slice; DELETE3 deletes the fixed three frames. No new source, attack, watermark receiver, random query or historical post-evaluation was executed to assemble these tables.

The main writer acts during generation/native decoding; external baselines write after the first media stage. Native capacities differ. Tables are a same-domain comparison under the recorded media recipes, not equal-distortion/equal-capacity rankings. FrameIndex is the adapted identity248+time8 reference, not a dedicated temporal-watermark SOTA. Methods without time output have N/A time columns.

Exact frame positions, exact endpoint pairs, event categories, time-supported payload, native frame BER, native aggregated messages and QMatch are separate measurements. MATCH compares a fixed claimed word, not authenticated time. MISMATCH records contradictory evidence. UNRESOLVED and INPUT_UNRESOLVED are nonacceptance, not proof of no watermark. P0 contains identity payload and is not OFF.

The random-query counting theorem assumes Q is conditionally uniform and independent of all frozen, query-free evidence E. At most one complete32 word is accepted per view/key. A source/arm family has ten slots, d distinct known words and u technical unknowns: [d,min(10,d+u)]/2^32. Logical unresolved evidence contributes an empty accept set. Unknown saved input contributes an unknown slot. For Q!=M remove M and use 2^32-1; OFF keeps 2^32. Fixed q0/q1 observations are outside this model. The 36 families and 360 opportunities define a probability model, not executed random draws. A conditional bound below 1e-6 is not empirical fixed-identity population FPR.

Compact12 is a separate 500-source public-domain experiment. Each method's 0/500 and one-sided95 reference upper bound are individual statements, not a four-method simultaneous confidence guarantee. Original thresholds are retained; no shared FPR curve or test-set tuning is introduced.

PSNR/MSE measure pixel deviation from the recorded reference, not perception. Saved wall times cover different stages and do not support a speed ratio. The sole new quality protocol is fixed16-frame MUSIQ-SPAQ imaging quality: it does not quantify flicker, transient artifacts, temporal coherence, aesthetics, prompt agreement or comprehensive video quality. Quality has no PASS/leading-score stopping threshold.
'''


def assemble(main_assembly,wan_root,frozen_root,output,*,compact_root=None):
    main_assembly,wan_root,frozen_root=map(Path,(main_assembly,wan_root,frozen_root))
    inputs=[main_assembly,wan_root,frozen_root]+([Path(compact_root)] if compact_root else [])
    output=rec.separate_output(output,inputs);output.mkdir(parents=True,exist_ok=True);errors=[]
    packed=rec.optional(main_assembly/'inputs/saved_records.json',errors)
    records=packed.get('records',{}) if isinstance(packed,dict) else {}
    def main(sid,name):
        value=records.get(sid+'/'+name)
        if value is None: errors.append(dict(path=sid+'/'+name,error='saved main record missing'))
        return value
    payload_path=main_assembly/'tables/payload_by_source_view.csv'
    payload_rows=rec.csv_rows(payload_path) if payload_path.is_file() else []
    fixed=[];time_payload=[];wan_rows=[];frozen_slots=[];families=[];cost=[];qualities=[]
    for sid in p.SOURCES:
        evaluation=main(sid,'evaluation_summary.json');evaluation=evaluation if isinstance(evaluation,dict) else {}
        decisions=main(sid,'decisions.json');decisions=decisions if isinstance(decisions,list) else []
        writer=rec.optional(frozen_root/sid/'attempt01/private/writer.json',errors)
        actual=writer.get('bits') if isinstance(writer,dict) else None
        terminal=rec.optional(frozen_root/sid/'attempt01/blind_terminal.json',errors)
        closed=isinstance(terminal,dict) and terminal.get('status')=='ALL_BLIND_TERMINAL'
        for a,arm in enumerate(p.ARMS):
            evs=[]
            for vi,view in enumerate(p.VIEWS):
                uid=f'input{5*a+vi}'
                for key in p.KEYS:
                    path=frozen_root/sid/'attempt01/evidence'/uid/(key+'.json');ev=rec.optional(path,errors)
                    if not closed: ev=None
                    word,status=prior.evidence_word(ev);evs.append(ev)
                    frozen_slots.append(dict(source=sid,arm=arm,view=view,key=key,state=status,word=word,input_path=str(path),blind_closed=closed))
                    for claim in ('q0','q1'):
                        matches=[r for r in decisions if isinstance(r,dict) and r.get('input_id')==uid and r.get('key_id')==key and r.get('claim_id')==claim]
                        row=matches[0] if len(matches)==1 else {};verdict=row.get('verdict')
                        usable=row.get('status')=='RETURNED' and verdict in p.VERDICTS
                        fixed.append(dict(source=sid,arm=arm,view=view,key=key,claim=claim,
                            verdict=verdict if usable else 'INPUT_UNRESOLVED',saved_verdict=verdict,
                            observation_status='SAVED_RETURNED' if usable else 'MISSING_OR_INVALID',
                            case_role=role(arm,key,claim),comparisons=row.get('comparisons'),valid_count=row.get('valid_count')))
            families.append(dict(source=sid,arm=arm,**p.family_counts(evs,actual_message=actual,arm=arm)))
        for vi,view in enumerate(p.VIEWS):
            uid=f'input{10+vi}';tm=evaluation.get('time',{}).get(uid,{})
            comparisons=evaluation.get('views',{}).get(view,{})
            matches=[r for r in payload_rows if r.get('source')==sid and r.get('view')==view]
            saved_payload=matches[0] if len(matches)==1 else {}
            payload={}
            for name in ('B_correct_supported','U_correct_supported','P_PAIRED_correct_supported_lower',
                         'P_PAIRED_correct_supported_upper','common_readable_bits','added_support','lost_support','unknown'):
                try:
                    value=int(saved_payload[name])
                    if not 0<=value<=32: raise ValueError('outside fixed32 slots')
                    payload[name]=value
                except (KeyError,ValueError,TypeError): payload[name]=None
            # Already saved arithmetic summaries; no literal or support reevaluation.
            time_payload.append(dict(source=sid,view=view,planned_nodes=prior.LENGTHS[vi],planned_edges=prior.LENGTHS[vi]-1,
                node_statuses=tm.get('node_statuses',{'TECHNICAL_UNKNOWN':prior.LENGTHS[vi]}),
                exact_endpoint_pair_statuses=tm.get('edge_statuses',{'TECHNICAL_UNKNOWN':prior.LENGTHS[vi]-1}),
                main_saved_comparisons=comparisons,**payload,
                fixed_payload_slots=32,record_status='SAVED' if tm else 'UNKNOWN'))
            for method in p.METHODS:
                for arm in ('OFF','WM'):
                    path=wan_root/sid/'attempt01/post'/method/arm/(view+'.json');value=rec.optional(path,errors)
                    valid=isinstance(value,dict)
                    value=value if valid else {}
                    wan_rows.append(dict(source=sid,method=method,arm=arm,view=view,record_status='SAVED' if valid else 'UNKNOWN',
                        payload=value.get('payload'),ownership=value.get('ownership'),wrong_query=value.get('wrong_query'),
                        time=value.get('time'),wrong_query_collision=value.get('wrong_query_collision'),input_path=str(path)))
        for family,record in [('main_generation','generation_resources.json'),('main_identity','identity_backend_resources.json'),('main_chain','chain_resources.json')]:
            value=records.get(sid+'/'+record)
            cost.append(dict(source=sid,stage=family,saved=value,status='SAVED' if value is not None else 'UNKNOWN',speed_ratio=None))
        result=rec.optional(wan_root/sid/'attempt01/result.json',errors)
        cost.append(dict(source=sid,stage='wan_saved_continuation_excludes_upstream_generation',saved=result,status='SAVED' if result else 'UNKNOWN',speed_ratio=None))
        q=main(sid,'quality.json')
        if isinstance(q,dict):
            qualities.extend(dict(source=sid,method='main',pair=key,measurement=value,interpretation='reference-relative pixel deviation') for key,value in q.items())
        q=rec.optional(wan_root/sid/'attempt01/quality.json',errors)
        if isinstance(q,list): qualities.extend(dict(source=sid,method=row.get('method'),pair='WM_OFF/'+str(row.get('view')),measurement=row,interpretation='reference-relative pixel deviation') for row in q if isinstance(row,dict))
    summary=[];native_video=[]
    for method in p.METHODS:
        rows=[r for r in wan_rows if r['method']==method and r['arm']=='WM'];off=[r for r in wan_rows if r['method']==method and r['arm']=='OFF']
        totals=payload_totals(rows,method)
        video,video_summary=video_messages(rows,method);native_video.extend(video)
        time=[r['time'] for r in rows if isinstance(r['time'],dict)]
        fi_nodes=Counter();fi_events=event_totals(time,sum(n-1 for n in prior.LENGTHS)*12)
        for row in time: fi_nodes.update(row.get('nodes',{}))
        if method=='frameindex':
            fi_nodes['technical_unknown_planned']=sum(prior.LENGTHS)*12-fi_nodes['target_planned']
        summary.append(dict(method=method,sources=12,views=60,positive_detection=decision_counts(rows,'ownership'),
            off_detection=decision_counts(off,'ownership'),wrong_query=decision_counts(rows,'wrong_query'),native_payload=totals,
            native_video_summary=video_summary,time_nodes=dict(fi_nodes) if method=='frameindex' else 'N/A',
            event_categories=dict(fi_events) if method=='frameindex' else 'N/A',exact_endpoint_pairs='N/A',
            capacity='identity248+time8' if method=='frameindex' else 32 if method=='wam' else 256))
    fixed_summary=[]
    for panel,sids in [('ALL',p.SOURCES),('TEST-A',p.SOURCES[:4]),('TEST-B',p.SOURCES[4:])]:
        for arm in p.ARMS:
            for key in p.KEYS:
                for claim in ('q0','q1'):
                    group=[r for r in fixed if r['source'] in sids and r['arm']==arm and r['key']==key and r['claim']==claim]
                    any5=[at_least([r['verdict'] for r in group if r['source']==sid],1) for sid in sids]
                    fixed_summary.append(dict(panel=panel,arm=arm,key=key,claim=claim,planned_sources=len(sids),planned_views=5*len(sids),
                        verdict_counts={v:sum(r['verdict']==v for r in group) for v in p.VERDICTS},
                        source_any5_accept=sum(v is True for v in any5),source_any5_nonaccept=sum(v is False for v in any5),source_any5_unknown=sum(v is None for v in any5),
                        random_query_theorem_applies=False,nonaccept_is_proof_of_absence=False))
    previous=rec.optional(main_assembly/'summary.json',errors)
    previous=previous if isinstance(previous,dict) else {}
    event_rows=rec.csv_rows(main_assembly/'tables/event_type_confusion.csv') if (main_assembly/'tables/event_type_confusion.csv').is_file() else []
    compact=rec.optional(Path(compact_root)/'VERIFICATION_SUMMARY.json',errors) if compact_root else None
    compact_rows=[dict(row,domain='500-source public Compact12, separate from Wan',confidence_scope='individual method; not simultaneous four-method coverage')
        for row in (compact.get('ownership',[]) if isinstance(compact,dict) else []) if row.get('arm')=='OFF' and row.get('channel')=='FAMILY12' and row.get('verification')=='query']
    for name,rows in [('wan_saved_posts',wan_rows),('main_time_payload',time_payload),('baseline_comparison',summary),
        ('main_fixed_claims',fixed),('fixed_four_state_summary',fixed_summary),('frozen_accept_slots',frozen_slots),
        ('conditional_families',families),('main_event_categories',event_rows),('saved_pixel_quality',qualities),
        ('saved_stage_cost',cost),('native_video_messages',native_video),('compact12_individual_reference',compact_rows),
        ('quality_execution_plan',p.streams()),('baseline_quality_references',rec.baseline_references(p.ROOT,wan_root))]: rec.table(output,name,rows)
    theory=p.protocol_bounds();theory['saved_family_mass_union_upper']=min(1.,sum(r['uniform32_upper'] for r in families))
    theory['saved_unknown_slots']=sum(r['u'] for r in families)
    theory['saved_wrong_query_union_upper']=min(1.,sum(r['wrong_query_upper'] for r in families))
    theory['saved_wrong_query_scope']='conditional on this frozen panel E, exclude the true word for P0/NEW; not empirical population FPR'
    theory['historical_random_receipts']='NOT_FOUND_IN_THIS_LOCAL_INPUT_SET; not a claim that no historical run exists'
    rec.write(output/'theory.json',theory)
    result=dict(status='SAVED_RECORD_ARITHMETIC_COMPLETE',fixed_counts=dict(wan_posts=len(wan_rows),main_views=len(time_payload),
        fixed_claims=len(fixed),frozen_evidence_slots=len(frozen_slots),conditional_families=len(families),quality_streams=84,quality_frames=1344),
        raw_read_errors=errors,main_saved_summary=previous,baseline_summaries=summary,theory=theory,
        new_execution=dict(model=0,codec=0,receiver=0,post_evaluation=0,random_queries=0),real_quality='NOT_EXECUTED_USER_COLAB_REQUIRED',scientific_pass=None)
    rec.write(output/'summary.json',result);(output/'LIMITATIONS.md').write_text(LIMITS,encoding='utf-8')
    write_report(output,result,time_payload)
    return result


def write_report(output, result, time_payload):
    lines=['# 保存证据联合表（未执行真实质量评分）', '',
        '十二个固定 Wan 源 × 五个既有时间编辑；480 个原始 baseline 后评只读。派生表不重跑接收器或后评。', '',
        '| 方法 | WM接受/60 | OFF接受/60 | 错查询接受/60 | 判定未知 | 原生逐帧错位/已读位 | 载荷记录未知 | 完整原生视频词/60 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['baseline_summaries']:
        a,o,q=(row[k] for k in ('positive_detection','off_detection','wrong_query'))
        native=row['native_payload'];video=row['native_video_summary']
        word=f"{video['complete_messages']}/60 (未知{video['unknown_views']})" if video['applicable'] else 'N/A'
        lines.append(f"| {row['method']} | {a['accept']} | {o['accept']} | {q['accept']} | {a['unknown']+o['unknown']+q['unknown']} | {native['error_bits']}/{native['observed_bits']} | {native['unknown_payload_records']} | {word} |")
    nodes=Counter();edges=Counter()
    for row in time_payload:
        nodes.update(row['node_statuses']);edges.update(row['exact_endpoint_pair_statuses'])
    b=sum(r['B_correct_supported'] or 0 for r in time_payload);u=sum(r['U_correct_supported'] or 0 for r in time_payload)
    missing=sum(r['B_correct_supported'] is None or r['U_correct_supported'] is None for r in time_payload)
    theory=result['theory']
    fi=next(r for r in result['baseline_summaries'] if r['method']=='frameindex')['event_categories']
    lines += ['',f'主线节点：{dict(nodes)}，固定9864；精确端点对：{dict(edges)}，固定9804。',
        '事件类别另见 main_event_categories.csv；FrameIndex 的 adjacent_* 是事件类别，不能冒充精确端点对。',
        f"FrameIndex事件类别（adjacent+gap+repeat）：{fi['correct']}正确/{fi['incorrect']}错误/{fi['unknown']}未知；固定{fi['planned']}，技术缺失计划{fi['technical_unknown_planned']}。",
        f'主线逐视图时间支撑载荷 B={b}/1920、U={u}/1920，缺失视图={missing}；与原生逐帧 BER、原生视频词分列。', '',
        f"条件均匀随机查询包上界 360/2^32={theory['package_uniform32_upper']:.12g}；保存接受集 union 上界={theory['saved_family_mass_union_upper']:.12g}，未知槽={theory['saved_unknown_slots']}。",
        f"保存接受集的均匀查询计数含查询恰好等于真消息；P0/NEW剔除真消息后的固定E条件错查询union上界={theory['saved_wrong_query_union_upper']:.12g}。即使该值为0，也不表示经验总体FPR为0。",
        '新增随机抽样0；以上是给定独立均匀查询及冻结证据的条件界，不是固定身份总体经验FPR。固定四态另见 fixed_four_state_summary.csv。',
        'Compact12 单独列在 compact12_individual_reference.csv；其500源公共域与Wan域不同，各方法单侧95%界不是四方法同时置信界。', '',
        '质量计划为84逻辑流、1344固定帧槽。尚未运行MUSIQ、真实媒体解码或GPU；不存在新的质量得分或科学PASS。',
        'baseline 的旧质量记录只声明同代OFF，若没有明确的保存OFF路径，新的WM−OFF配对差值保持未知；绝对WM评分仍执行，不扩展84流。',
        'PSNR只是相对参考的像素偏离；保存耗时覆盖不同阶段，不作速度倍数或等扰动排名。详见 LIMITATIONS.md。', '']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
