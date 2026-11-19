"""Fixed-structure xG replication across five leagues; no live result review."""
from datetime import date,datetime,timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.native_experiment import season_directories,load_seasons
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid,summarize
from football_model.xg_history import join_understat,DateTeamXgModel
from run_shot_increment_20261025 import score_details,write

ROOT=REPOSITORY_ROOT
SOURCE=ROOT/'data/understat/history_20261024T172015108538Z'
AUDIT=ROOT/'outputs/跨联赛xG源核对_20261024T172403681186Z/逐季源差异.json'
EPL=ROOT/'outputs/独立历史xG前推_20261024T171031805341Z/独立xG前推结果.json'
CORE=ROOT/'outputs/核心参数训练_20261023T115640092598Z'
SLUGS={'D1':'Bundesliga','SP1':'La_Liga','I1':'Serie_A','F1':'Ligue_1'}


def load_originals(records):
    source={r.match_id:r for r in records}
    rows=[]
    for period,arm in (('2024-25','regularized_poisson_h180_a0p01'),('2025-26','original')):
        for competition in SLUGS:
            path=CORE/period/arm/f'{competition}.json'
            trial=json.loads(path.read_text(encoding='utf-8'))
            if trial['failures'] or trial['update_interval_days']!=28:
                raise ValueError('Original failures or update interval conflict')
            fits={f['before_date']:f for f in trial['fits']}
            for row in trial['predictions']:
                record=source[row['match_id']]
                fit=fits[row['fit_before_date']]
                if (row['date']!=record.match_date.isoformat() or row['actual_score']!=list(record.full_time)
                        or row['home']!=record.home_label or row['away']!=record.away_label
                        or not fit['training_max_date']<fit['before_date']<=row['date']
                        or fit['regularization']!=.01 or fit['half_life_days']!=180
                        or record.goal_fit_exclusion is not None):
                    raise ValueError('Original identity, chronology or preset differs')
                p=list(summarize(score_grid(*row['rates'],rho=0))['one_x_two'].values())
                if not np.allclose(p,row['probabilities'],rtol=0,atol=1e-10):
                    raise ValueError('Original rate replay mismatch')
                rows.append({**row,'home':record.home,'away':record.away,'period':period})
    if len(rows)!=2722 or len({r['match_id'] for r in rows})!=len(rows):
        raise ValueError('Original four-league historical coverage changed')
    return sorted(rows,key=lambda r:(r['date'],r['match_id']))


def comparison(old,new):
    if len(old)!=len(new) or [r['match_id'] for r in old]!=[r['match_id'] for r in new]:
        raise ValueError('Comparison must use identical fixtures and order')
    before,before_loss=score_details(old)
    after,after_loss=score_details(new)
    order=list(after_loss)
    differences=np.column_stack([after_loss[k]-before_loss[k] for k in order])
    intervals={str(block):moving_block_means([r['date'] for r in old],differences,block_weeks=block)
               for block in (2,4,8)}
    return {'n':len(new),'original_scores':before,'xg_scores':after,'difference_order':order,
            'mean_loss_differences':dict(zip(order,differences.mean(axis=0).tolist())),
            'block_intervals':intervals,'common_match_ids':[r['match_id'] for r in new]}


def main():
    now=datetime.now(timezone.utc)
    target=ROOT/'outputs'/now.strftime('独立xG跨联赛复现_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    plan={'created_at':now.isoformat(),'purpose':'replicate the same independent xG structure outside EPL',
          'new_live_results_reviewed':False,'leagues_new':list(SLUGS),'epl':'reuse earlier 22 fits, no refit',
          'source_directory':str(SOURCE),'source_difference_audit':str(AUDIT),
          'preset':{'regularization':.01,'half_life_days':180,'update_days':28,'rho':0},
          'budget':{'new_xg_fits':88,'reused_epl_fits':22,'new_goal_fits':0,'fusion_coefficients':0,'parameter_searches':0},
          'primary':'actual complete-score NLL, per league/season plus calendar-paired pooled description',
          'secondary':['1X2 LL/Brier/Top1','total 0/1/2/3/4+ LL','away-minus-two three-way LL'],
          'source_policy':'exclude every ordinary date conflict recorded before fitting, do not correct either source',
          'special_policy':'retain and exclude original registered awarded/resumed fixtures',
          'ordinary_date_conflicts_expected':15,'registered_special_expected':3,
          'training_confound':'xG source excludes 15 ordinary date conflicts still present in original goal history; no pure causal information attribution',
          'availability':'only strict earlier calendar-date xG at the original goal fit cutoff, provider publication/revisions unknown',
          'uncertainty':'paired noncircular calendar blocks 2/4/8 weeks,2000 repeats; descriptive, no refit/selection bootstrap',
          'adoption':'no automatic replacement, fusion or winner selection; report all league/season results',
          'scope':'historical results already viewed; new xG inputs, not untouched blind validation'}
    write(target/'分析前计划.json',plan)
    records,_=load_seasons(season_directories(ROOT))
    audit=json.loads(AUDIT.read_text(encoding='utf-8'))
    xg_records,joins,ordinary_exclusions=[],{},[]
    for competition,slug in SLUGS.items():
        for year in (2022,2023,2024,2025):
            season=f'{year}/{str(year+1)[-2:]}'
            key=f'{competition}:{season}'
            discrepancies=[r for r in audit[key]['differences'] if not r['known_exclusion']]
            bad_ids={r['understat_id'] for r in discrepancies}
            bad_independent={r['football_data_match_id'] for r in discrepancies}
            for r in discrepancies:
                if r['football_data_match_id'] is None:
                    raise ValueError('Unknown fixture identity must be resolved explicitly')
            raw=json.loads((SOURCE/f'{slug}_{year}.json').read_text(encoding='utf-8'))['dates']
            filtered=[r for r in raw if r['id'] not in bad_ids]
            independent=[r for r in records if r.match_id not in bad_independent]
            parsed,report=join_understat(filtered,independent,season,competition)
            report.update({'full_understat_rows':len(raw),
                           'full_independent_rows':sum(r.competition==competition and r.season==season for r in records),
                           'ordinary_date_conflicts':discrepancies})
            if report['missing_independent_match_ids'] or report['missing_special_match_ids']:
                write(target/'逐季数据联结.json',joins|{key:report})
                raise ValueError('Unexplained xG coverage missingness')
            joins[key]=report
            xg_records.extend(parsed)
            ordinary_exclusions.extend({'competition':competition,'season':season,**r} for r in discrepancies)
    write(target/'逐季数据联结.json',joins)
    write(target/'普通日期冲突排除.json',ordinary_exclusions)
    if len(ordinary_exclusions)!=15 or sum(len(r['excluded_special_matches']) for r in joins.values())!=3:
        raise ValueError('Source audit/exclusion counts changed')
    originals=load_originals(records)
    available={r.match_id for r in xg_records}
    models,new,excluded={},[],[]
    for competition in SLUGS:
        local_history=[r for r in xg_records if r.competition==competition]
        for row in [r for r in originals if r['competition']==competition]:
            if row['match_id'] not in available:
                excluded.append({'match_id':row['match_id'],'period':row['period'],
                                 'competition':competition,'reason':'ordinary_source_date_conflict'})
                continue
            key=f'{competition}:{row["fit_before_date"]}'
            if key not in models:
                models[key]=DateTeamXgModel().fit(local_history,date.fromisoformat(row['fit_before_date']))
            model=models[key]
            if {row['home'],row['away']}-model.teams:
                excluded.append({'match_id':row['match_id'],'period':row['period'],
                                 'competition':competition,'reason':'unseen_xg_team'})
                continue
            rates=model.predict(row['home'],row['away'],competition,date.fromisoformat(row['date']))
            forecast=summarize(score_grid(*rates,rho=0))
            new.append({**row,'original_rates':row['rates'],'rates':list(rates),'forecast':forecast,
                        'probabilities':list(forecast['one_x_two'].values())})
        print(f'Completed {competition}: '+str(sum(k.startswith(competition+":") for k in models))+' fixed xG fits',flush=True)
    if len(models)!=88:
        raise ValueError('Unexpected fit count, no budget expansion')
    epl=json.loads(EPL.read_text(encoding='utf-8'))
    all_new=sorted(new+epl['new_predictions'],key=lambda r:(r['date'],r['match_id']))
    all_original=originals+epl['original_predictions']
    original_map={r['match_id']:r for r in all_original}
    by_league,seasons={},{}
    for competition in ('E0',*SLUGS):
        by_league[competition]={}
        for period in ('2024-25','2025-26'):
            selected=[r for r in all_new if r['competition']==competition and r['period']==period]
            old=[original_map[r['match_id']] for r in selected]
            entry=comparison(old,selected)
            entry['original_full_n']=sum(r['competition']==competition and r['period']==period for r in all_original)
            by_league[competition][period]=entry
    for period in ('2024-25','2025-26'):
        seasons[period]={}
        for label,selected in [('all_five',[r for r in all_new if r['period']==period]),
                               ('four_new',[r for r in all_new if r['period']==period and r['competition']!='E0'])]:
            seasons[period][label]=comparison([original_map[r['match_id']] for r in selected],selected)
    result={'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),
            'new_xg_fits':len(models),'reused_epl_fits':22,'new_goal_fits':0,'parameter_searches':0,
            'new_raw_source_matches':sum(r['full_understat_rows'] for r in joins.values()),
            'new_usable_source_matches':len(xg_records),'all_five_usable_source_matches':len(xg_records)+1520,
            'join_reports':joins,'ordinary_exclusions':ordinary_exclusions,'target_exclusions':excluded,
            'by_league':by_league,'pooled_seasons':seasons,
            'new_xg_parameters':{k:m.parameters() for k,m in models.items()},
            'reused_epl_source':str(EPL),'original_predictions':originals,'new_predictions':new,
            'adopted':False}
    write(target/'跨联赛xG复现结果.json',result)
    report=['# 独立xG模型：五大联赛固定结构复现', '',
            '本轮只优化模型，没有查询或复核当前赛果。英超22次拟合原样复用；其他四联赛新增88次xG拟合。固定正则.01、半衰180天、原28天截止，原进球拟合0次、融合系数0个、参数搜索0次。', '',
            '以较早逐场xG拟合攻防与主场均值，再用真实完整进球评分；xG目标采用非负小数Poisson准损失，同场forecast／赛季汇总没有作为输入。', '',
            f'五联赛四季原数据7082场，3场既有改判／续赛及15场普通日期冲突分别保留并排除，可用{result["all_five_usable_source_matches"]}场。普通日期未校正、未放宽匹配；排除规则在拟合前固定，不根据预测好坏选择。', '',
            '| 联赛／赛季 | 共同场数／原覆盖 | 比分NLL差 | 胜平负LL差 | 原Top1 | xG Top1 |',
            '|---|---:|---:|---:|---:|---:|']
    names={'E0':'英超','D1':'德甲','SP1':'西甲','I1':'意甲','F1':'法甲'}
    for comp,periods in by_league.items():
        for period,item in periods.items():
            report.append(f'| {names[comp]}／{period} | {item["n"]}/{item["original_full_n"]} | {item["mean_loss_differences"]["exact_score"]:+.6f} | {item["mean_loss_differences"]["one_x_two"]:+.6f} | {item["original_scores"]["one_x_two"]["accuracy"]:.2%} | {item["xg_scores"]["one_x_two"]["accuracy"]:.2%} |')
    report+=['', '差值均为xG减原模型，负值较好；完整比分是主指标。各联赛和两季全部保留，不能挑选有利联赛宣称总体有效。', '',
             '| 汇总范围／赛季 | 场数 | 比分NLL差 | 4周95%描述区间 | 胜平负LL差 | 4周95%描述区间 |',
             '|---|---:|---:|---|---:|---|']
    for period,ranges in seasons.items():
        for label,item in ranges.items():
            def ci(key):
                lo,hi=item['block_intervals']['4']['interval_95'][item['difference_order'].index(key)]
                return f'[{lo:+.6f}, {hi:+.6f}]'
            report.append(f'| {label}／{period} | {item["n"]} | {item["mean_loss_differences"]["exact_score"]:+.6f} | {ci("exact_score")} | {item["mean_loss_differences"]["one_x_two"]:+.6f} | {ci("one_x_two")} |')
    report+=['', '## 公平性及解释限制', '',
             '同场评分不代表训练观测完全相同：xG训练排除了15场普通日期冲突，而原进球模型的历史仍可能含它们。这是保守资料处理，也限制把差异纯因果归因于机会质量；本轮没有偷偷增加共同训练集进球对照或改参数。', '',
             '数据是今天取得的历史xG，原始发布时点和后来修订未知。只用日期级更早可用假设；原比分赛季以前已查看，不是新盲测。逐场xG来自同一数据商，跨联赛不是跨数据商独立验证。', '',
             '移动块采用共同日历权重，同一周不同联赛一同抽取，避免把它们完全当独立观察；区间仍只是条件于已保存预测的描述敏感性，不重演模型选择或全部参数不确定性。2／4／8周全部区间保留在结果文件。', '',
             '没有根据本次结果替换正式模型、融合线路或添加神经网络；局部联赛优势需另有更晚数据支持，不能依据本次目标季继续调参追成绩。未知球队保持缺失。赛果复核继续暂停，等用户提醒。', '']
    (target/'跨联赛xG复现说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'directory':str(target),'new_fits':len(models),'usable_source':len(xg_records)+1520,
        'target_exclusions':len(excluded),'by_league':{c:{p:{'n':i['n'],'score_diff':i['mean_loss_differences']['exact_score'],
        'outcome_diff':i['mean_loss_differences']['one_x_two']} for p,i in ps.items()} for c,ps in by_league.items()},
        'pooled':{p:{k:{'n':i['n'],'diffs':i['mean_loss_differences'],'4week_CI':i['block_intervals']['4']['interval_95']} for k,i in rs.items()} for p,rs in seasons.items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
