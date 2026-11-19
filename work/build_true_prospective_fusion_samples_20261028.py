"""Inventory archived prematch pairs, retaining earliest declared versions only.

No network, fitting, LLM generation, or result refresh. Local packet/response
matching supports traceability; it cannot authenticate a model's full runtime.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/模型LLM融合研究_20261028'
EVALUATION = ROOT/'outputs/前瞻评价接续_20261027/分组前瞻记录与评价.json'
SINGLE = ROOT/'data/prematch-intelligence/20261024_multiline'
ORIGINAL = ROOT/'data/prematch-intelligence/20261024_batch/run_20261023T175817818024Z'
NEXT = ROOT/'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z'
LABELS = ('home', 'draw', 'away')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def at(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Need an aware archived timestamp')
    return result.astimezone(timezone.utc)


def prob(values):
    if (len(values) != 3 or any(isinstance(x, bool) or not isinstance(x, (int,float))
            or not math.isfinite(x) or x < 0 for x in values)
            or not math.isclose(sum(values),1,abs_tol=1e-12)):
        raise ValueError('Need archived finite H/D/A probabilities')
    return values


def fixture_key(fixture):
    return tuple(fixture[k] for k in ('match_id','home','away','competition','kickoff','neutral'))


def response_list(doc):
    seq = doc.get('forecasts',doc.get('records'))
    if seq is None:
        return [doc]
    return list(seq.values()) if isinstance(seq,dict) else seq


def audit_selected(cohort, row):
    fid = row['fixture']['match_id']
    if cohort == 'single_earliest':
        raw_path,packet_path = SINGLE/'independent_response.json',SINGLE/'independent_packet.json'
    elif cohort == 'original10':
        raw_path,packet_path = ORIGINAL/'independent_response.json',ORIGINAL/'independent_packets.json'
    elif cohort == 'next_original10':
        raw_path,packet_path = NEXT/'independent_response.json',NEXT/'independent_packets.json'
    elif cohort.startswith('serie_a_'):
        day = '20261027' if '_20261027_' in cohort else '20261028'
        raw_path = ROOT/f'data/serie_a_today_{day}/llm_response.json'
        packet_path = raw_path.parent/'facts_only_llm_packet.json'
    else:
        raise ValueError('Unexpected earliest source; review without selecting by result')
    raw,packet = read(raw_path),read(packet_path)
    if cohort == 'single_earliest':
        answer = raw
        selected_packet = packet
    else:
        answers = {x.get('fixture_id',x.get('fixture',{}).get('match_id')):x for x in response_list(raw)}
        answer = answers[fid]
        packets = packet.get('packets',packet.get('records'))
        seq = list(packets.values()) if isinstance(packets,dict) else packets
        selected_packet = next(x for x in seq if x['fixture']['match_id']==fid)
    track = row['tracks']['llm_independent']
    generated = answer.get('generated_at_utc',answer.get('generated_at',raw.get('generated_at_utc',raw.get('generated_at'))))
    if prob(answer['probabilities']) != track['probabilities'] or at(generated)!=at(track['generated_at']):
        raise ValueError('Earliest response and archived track differ')
    if fixture_key(selected_packet['fixture']) != fixture_key(row['fixture']):
        raise ValueError('Fact packet fixture differs')
    packet_as_of = selected_packet.get('as_of',packet.get('prepared_at',packet.get('assembled_at')))
    if at(packet_as_of)>at(generated):
        raise ValueError('LLM generation precedes its archived input packet')
    exposure = raw.get('exposure',{})
    answer_stat = answer.get('exposed_to_statistical',exposure.get('statistical_probabilities',exposure.get('statistics')))
    answer_market = answer.get('exposed_to_market',exposure.get('market_probabilities',exposure.get('market')))
    if answer_stat is not False or answer_market is not False:
        raise ValueError('Raw response lacks consistent no-statistical/no-market declaration')
    if selected_packet.get('statistical') is not None:
        raise ValueError('Independent packet unexpectedly includes model probabilities')
    model_id = answer.get('model_identifier',raw.get('model_id','unknown'))
    return {
        'quality':'declared_independent_raw_response_and_packet_matched_runtime_not_authenticated',
        'raw_response_path':str(raw_path), 'packet_path':str(packet_path),
        'raw_generation_at':generated,'packet_as_of':packet_as_of,
        'actual_model_id':model_id,'model_id_known':model_id not in ('unknown',None,''),
        'raw_probabilities_and_generation_match':True,'fixture_identity_matches_packet':True,
        'exposed_to_statistical':False,'exposed_to_market':False,
        'full_runtime_context_independently_verified':False,
        'qualification':'Archived packet and self-declaration support a declared-independent line; neither exact runtime identity nor the complete execution context is independently authenticated. Unknown identity is retained, not imputed.',
    }


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assembled = datetime.now(timezone.utc).isoformat()
    existing = read(EVALUATION)
    cohorts = dict(existing['cohort_records'])
    source_paths = dict(existing['source_cohorts'])
    source_paths.update({'next_simple10':str(EVALUATION),'next_rich10':str(EVALUATION)})
    cohorts['single_earliest'] = read(SINGLE/'multiline_case.json')['records']
    source_paths['single_earliest'] = str(SINGLE/'multiline_case.json')
    for day in ('20261027','20261028'):
        for route in ('goal_simple','independent_xg','rich_121_research'):
            name = f'serie_a_{day}_{route}'
            path = ROOT/f'outputs/今日意甲预测_{day}/前瞻记录_{route}.json'
            cohorts[name] = read(path)['records']
            source_paths[name] = str(path)
    # Source selection never consults labels. At exact simultaneous ties, keep
    # the primary goal route already declared in the original forecast report.
    tie_order = ['single_earliest','original10','next_original10',
        'serie_a_20261027_goal_simple','serie_a_20261028_goal_simple',
        'personnel_revision9','next_context_revision7','next_simple10','next_rich10',
        'next_allcomp_repaired10','serie_a_20261027_independent_xg',
        'serie_a_20261027_rich_121_research','serie_a_20261028_independent_xg',
        'serie_a_20261028_rich_121_research']
    order = {name:i for i,name in enumerate(tie_order)}
    candidates = defaultdict(list)
    identities = {}
    for cohort,rows in cohorts.items():
        for row in rows:
            f = row['fixture'];fid=f['match_id']
            if fid in identities and fixture_key(f)!=identities[fid]:
                raise ValueError('Same match ID has conflicting fixture identity')
            identities[fid]=fixture_key(f)
            stat,lm = row['tracks'].get('statistical'),row['tracks'].get('llm_independent')
            if stat is None or lm is None:
                continue
            prob(stat['probabilities']);prob(lm['probabilities'])
            if (lm.get('exposed_to_statistical') is not False or lm.get('exposed_to_market') is not False):
                continue
            generated=max(at(stat['generated_at']),at(lm['generated_at']))
            if generated>at(row['as_of']) or at(row['as_of'])>=at(f['kickoff']):
                raise ValueError('Archived pair is not validly before kickoff')
            candidates[fid].append((generated,at(row['as_of']),order.get(cohort,999),cohort,row))
    selected = {fid:min(seq,key=lambda x:x[:3]) for fid,seq in candidates.items()}
    # Only two already-settled stores are read. No pending result lookups.
    result_paths = [ROOT/'outputs/赛果对比_20261026/原10场_原预测与赛果.json',
                    ROOT/'outputs/昨日意甲复核_20261028/结算.json']
    results={}
    for path in result_paths:
        for fid,value in read(path).get('results',{}).items():
            if fid in results:
                raise ValueError('Unexpected duplicated result identity across the two stores')
            results[fid]=(value,path)
    records=[]
    for fid,selection in sorted(selected.items(),key=lambda item:(at(item[1][4]['fixture']['kickoff']),item[0])):
        generated,_,_,cohort,row=selection
        f=row['fixture'];stat,lm=row['tracks']['statistical'],row['tracks']['llm_independent']
        audit=audit_selected(cohort,row)
        result,result_path=results.get(fid,(None,None))
        if result is not None:
            known=at(result['available_at']);finished=at(result['finished_by'])
            if known<finished or known<=at(f['kickoff']) or known>at(assembled):
                raise ValueError('Archived result availability chronology differs')
            h,a=result['home_goals'],result['away_goals']
            outcome=0 if h>a else 2 if a>h else 1
        else:
            known=None;outcome=None
        alternatives=[{
            'cohort':alt[3],'pair_generated_at':alt[0].isoformat(),'as_of':alt[4]['as_of'],
            'statistical_model':alt[4]['tracks']['statistical'].get('model'),
            'statistical_id':alt[4]['tracks']['statistical']['id'],
            'llm_id':alt[4]['tracks']['llm_independent']['id'],
            'selection':'not_selected_as_additional_observation',
        } for alt in sorted(candidates[fid],key=lambda x:x[:3]) if alt is not selection]
        records.append({
            'match_id':fid,'fixture':f,'cohort':cohort,'source_path':source_paths[cohort],
            'forecast_generated_at':generated.isoformat(),'available_as_of':row['as_of'],
            'forecast_group_utc_date':generated.date().isoformat(),
            'kickoff_group_utc_date':at(f['kickoff']).date().isoformat(),
            'competition':f['competition'],
            'statistical_probabilities':stat['probabilities'],'llm_probabilities':lm['probabilities'],
            'statistical_model_family':stat.get('model',stat.get('description','unknown')),
            'statistical':stat,'llm_independent':lm,'llm_audit':audit,
            'market_used_for_learning':False,'market_probabilities_supplied_to_llm_declared':False,
            'actual_outcome':outcome,'result':result,'result_available_at':None if known is None else known.isoformat(),
            'result_source_path':None if result_path is None else str(result_path),
            'current_role':'available_exploratory_training' if result is not None else 'no_archived_result',
            'scheduled_future_at_inventory_time':at(f['kickoff'])>at(assembled),
            'schedule_status_note':'Archived kickoff only; this inventory does not reconfirm schedule or fetch results',
            'alternative_version_count':len(alternatives),'alternative_versions':alternatives,
        })
    settled=[r for r in records if r['actual_outcome'] is not None]
    pending=[r for r in records if r['actual_outcome'] is None]
    later_italy=[r for r in settled if r['competition']=='Serie A']
    italy_start=min(at(r['forecast_generated_at']) for r in later_italy)
    earlier_epl=[r for r in settled if at(r['result_available_at'])<italy_start]
    if len(records)!=28 or len(settled)!=13 or len(pending)!=15 or len(earlier_epl)!=10 or len(later_italy)!=3:
        raise ValueError('Unexpected prospective coverage; inspect without inferring missing results')
    data={
        'assembled_at_utc':assembled,'class_order':list(LABELS),'records':records,
        'counts':{'raw_variant_pairs':sum(map(len,candidates.values())),'unique_fixtures':len(records),
            'settled_declared_independent_with_raw_packet_match':len(settled),
            'unsettled_without_archived_results':len(pending),'independently_authenticated_runtime':0,
            'known_exact_llm_model_ids':sum(r['llm_audit']['model_id_known'] for r in records),
            'by_selected_cohort':dict(Counter(r['cohort'] for r in records)),
            'by_settled_competition':dict(Counter(r['competition'] for r in settled))},
        'selection_policy':{
            'unit':'one match_id, identity verified across versions',
            'sort':['max(statistical.generated_at,llm.generated_at)','archived_record.as_of','predeclared_primary_source_tie_order'],
            'earliest_pair_not_best_performing_version':True,
            'same_timestamp_primary_route':'original report goal_simple; no score-based source selection',
            'duplicate_manchester_united_bournemouth':'earlier standalone pair retained instead of later batch pair',
            'tie_order':tie_order},
        'temporal_research_prefix':{
            'training_match_ids':[r['match_id'] for r in earlier_epl],
            'later_evaluation_match_ids':[r['match_id'] for r in later_italy],
            'earliest_later_component_generation_at':italy_start.isoformat(),
            'latest_earlier_result_available_at':max(at(r['result_available_at']) for r in earlier_epl).isoformat(),
            'interpretation':'May fit on 10 archived EPL pairs and inspect three later Serie A pairs using archived forecasts; the joint predictor was not published before those fixtures. This is reconstructed temporal research, not an actual prospective joint result.',
            'migration':'EPL to Serie A; different historical coverage and LLM input policies. A weight learned here is exploratory and not established for all leagues.'},
        'current_fit_scope':{'known_before':assembled,'train_match_ids':[r['match_id'] for r in settled],
            'n':len(settled),'interpretation':'All 13 known results may fit an exploratory weight for a genuinely new future forecast. Its training score cannot validate the selected weight.'},
        'forward_group_rule':'For any reconstructed fold, every training result_available_at must precede the earliest forecast_generated_at of the whole target forecast group. Never feed a result from a simultaneous/batch target into the same group.',
        'source_evaluation':str(EVALUATION),'settled_source_paths':[str(p) for p in result_paths],
        'new_fits':0,'new_llm_calls':0,'new_result_lookups':0,'old_forecasts_modified':False,
        'all_new_files_on_g':True,
        'limitations':[
            'All exact LLM model IDs remain unknown; multiple prompt policies are not independent model vendors.',
            'Local self-declarations and archived packets do not independently authenticate full prior execution context or exclude pretrained outcome memory.',
            'Only 13 already-settled unique matches are available; repeated versions, statistical variants and duplicated LLM responses do not expand this denominator.',
            'Current xG/rich statistical variants are retained as alternatives, not selected using known outcomes.',
            'The parent task should distinguish exploratory weight fitting from independent validation and actual historical publication.',
        ],
    }
    (OUT/'真实前瞻融合样本.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 模型与LLM：真实赛前融合样本盘点','',
        f'盘点UTC：{assembled}。本程序只读取既存赛前概率、原回答、事实包和两份既有结算，不联网、不拟合、不查询新赛果，也不改旧预测。','',
        f'原始{data["counts"]["raw_variant_pairs"]}组版本配对，去重后28场不同比赛；13场已有归档赛果，15场没有归档赛果。可用于探索性学习的是13场，不能把线路和版本数当比赛数。','',
        '|集合|不同比赛|本轮角色|','|---|---:|---|',
        '|英超原轮|10|已有赛果；可作较早学习段|',
        '|意甲10月27日|3|已有赛果；可作较晚重建检验段|',
        '|下一轮英超|10|未查赛果；原赛程尚在未来，未重新核实调期|',
        '|意甲10月28日|5|本轮原预测尚待结果；可另存真正新联合预测|','',
        '## 如何去重','',
        '按match_id核对完整比赛身份。只在同场同时有统计与声明独立LLM的版本中，选两条概率实际生成完成时间较早者；再以原记录as_of及事先指定的主线作同时刻裁决。选源不读取赛果。意甲同刻三条统计线保留原报告明确的goal_simple主线；xG和121研究列列作替代版本。','',
        '曼联—伯恩茅斯已有更早的独立单场记录，保留10月23日17:29:45Z完成的配对，而不是18:03:16Z的后来整批版本；仍只算一场。','',
        '## 独立性与身份的真实边界','',
        '全部28场选中配对均能匹配原LLM回答、三概率、生成时间及赛前事实包；原回答声明未接触统计或市场概率，包中没有提供当前统计概率。可以称“声明独立、原回答与包可核”。','',
        '实际精确模型ID全部为unknown。已有文件不能独立认证完整运行上下文，也不能认证外部发布时间或彻底排除预训练记忆；因此另列“完整运行上下文独立认证0场”，不能把这些声明写成认证隔离或独立模型供应商。未知身份保留未知，不补写。','',
        '## 权重学习能够怎样推进','',
        f'10场英超结果最晚于{data["temporal_research_prefix"]["latest_earlier_result_available_at"]}归档可得，早于三场意甲最早原概率生成{data["temporal_research_prefix"]["earliest_later_component_generation_at"]}。可用原10场学权重，再在后3场重建检验；该联合是今天新学，并非昨天真实发布的联合命中。','',
        '这还是英超到意甲、不同描述信息政策的迁移，小样本结果不证明普遍权重。当前可用全部13场学习一个探索性权重，供真正尚未开始的新联合预测；13场训练得分只能说明拟合，不能用于证明权重效果。','',
        '前推必须按整批预测生成日分组：训练赛果available_at早于目标整批最早概率生成时间。不能用同批先踢完的一场结果回头给该批原先已生成的其他概率选权重；finished_by是已观察完场的上界，不是假造真实终场时刻。','',
        'JSON中每场保留原统计模型、原回答身份与暴露声明、两个生成时间、原记录as_of、结果首次可用时间、来源路径、所有未选版本和市场不参与学习的声明。所有新文件在G盘。','']
    (OUT/'样本说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'counts':data['counts'],'temporal_prefix':data['temporal_research_prefix']},ensure_ascii=False))


if __name__=='__main__':
    main()
