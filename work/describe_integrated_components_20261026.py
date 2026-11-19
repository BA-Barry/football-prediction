"""Locate changes in the completed integration using saved predictions only."""
import gzip
import json
from pathlib import Path

import numpy as np

from football_model.evaluation import probability_scores
from football_model.prematch_calibration import apply_calibrator


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/现有组件整合_20261026'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    result = read(OUT/'整合完整结果.json')
    rows = read(OUT/'较晚逐场联合预测.json')['rows']
    with gzip.open(OUT/'整合赛前输入.json.gz','rt',encoding='utf-8') as f:
        inputs = json.load(f)['rows']
    by_id = {r['match_id']:r for r in inputs}
    oof = read(OUT/'较早整合时间外预测.json')['rows']
    temp = read(OUT/'整合温度参数.json')
    p = np.asarray([r['probabilities'] for r in oof])
    y = np.asarray([r['actual_outcome'] for r in oof])
    q = apply_calibrator(p,temp)
    earlier_reference = np.asarray([by_id[r['match_id']]['probabilities_by_route']['rich_performance'] for r in oof])
    actual = np.asarray([r['actual_outcome'] for r in rows])
    selected_old = np.asarray([r['probabilities']['original_rich'] for r in rows]).argmax(axis=1)
    selected_new = np.asarray([r['probabilities']['integrated_calibrated'] for r in rows]).argmax(axis=1)
    transitions = np.zeros((3,3),dtype=int)
    np.add.at(transitions,(selected_old,selected_new),1)
    class_changes = {}
    for c,name in enumerate(('home','draw','away')):
        old_n = int(((selected_old==c)&(actual==c)).sum())
        new_n = int(((selected_new==c)&(actual==c)).sum())
        class_changes[name] = {'actual_n':int((actual==c).sum()),'original_correct':old_n,
            'integrated_correct':new_n,'net_correct':new_n-old_n}
    full = np.asarray([r['market_losses']['integrated_calibrated']['score_nll']-
        r['market_losses']['original_rich']['score_nll'] for r in rows])
    hda = np.asarray([-np.log(r['probabilities']['integrated_calibrated'][r['actual_outcome']])+
        np.log(r['probabilities']['original_rich'][r['actual_outcome']]) for r in rows])
    identity_error = float(np.abs(full-hda).max())
    correlations = {}
    early = [r for r in inputs if r['period']=='2024-25']
    names = list(early[0]['probabilities_by_route'])
    for axis,label in ((0,'log_home_away'),(1,'log_draw_away')):
        arrays = np.column_stack([[
            np.log(r['probabilities_by_route'][name][axis])-
            np.log(r['probabilities_by_route'][name][2]) for r in early] for name in names])
        correlations[label] = {'routes':names,'pearson_matrix':np.corrcoef(arrays,rowvar=False).tolist()}
    description = {
        'new_fits':0,'source':'saved completed integration only; no model/parameter changes',
        'class_correct_changes':class_changes,
        'selection_transition_old_rows_new_columns_HDA':transitions.tolist(),
        'later_actual_frequencies':(np.bincount(actual,minlength=3)/len(actual)).tolist(),
        'earlier_meta_oos':{'n':len(oof),'first_date':oof[0]['date'],'last_date':oof[-1]['date'],
            'reference_raw':probability_scores(earlier_reference,y),
            'stack_raw':probability_scores(p,y),
            'stack_temperature_fit_apparent_not_external_test':probability_scores(q,y),
            'temperature':temp['parameters']},
        'earlier_member_log_odds_correlations':correlations,
        'score_loss_identity_max_error':identity_error,
        'score_loss_identity': 'When P(score|outcome) is unchanged, delta score NLL equals delta outcome NLL; these are not independent improvements/failures.',
        'interpretation_limits':['class accounting locates where errors changed, not why each source caused them',
            'temperature transfer between early smaller meta fits and final full-data fit can differ; observed transfer failure does not identify its sole cause',
            'correlated forecasts are not independent evidence; correlation alone does not prove redundancy or uselessness',
            'no learned coefficient is interpreted as a causal player/tactical effect']}
    write(OUT/'整合变化定位.json',description)
    lines = ['# 整合后变化在哪里：不新增训练的定位','',
        '本文件只读已存预测，不改模型、参数或方法；用来提取可积累的认识。','',
        '## 命中损失具体发生在哪里','',
        '| 实际赛果 | 场数 | 原正确 | 整合正确 | 净变化 |','|---|---:|---:|---:|---:|']
    for name,label in [('home','主胜'),('draw','平局'),('away','客胜')]:
        v = class_changes[name]
        lines.append(f"| {label} | {v['actual_n']} | {v['original_correct']} | {v['integrated_correct']} | {v['net_correct']:+d} |")
    lines += ['', '主胜识别的损失超过平局和客胜新增的正确判断，是本轮单选净下降的直接数量分解。它不证明某个成员或人员变量造成损失。','',
        '最终平均H/D/A概率为'+str([round(v,4) for v in result['scores']['integrated_calibrated']['mean_probabilities']])+', 实际频率为'+str([round(v,4) for v in description['later_actual_frequencies']])+'. 不能仅因开始选平，就称平局改善已弥补其他类别的代价。','',
        '## 校准的实际作用','',
        f"较早联合时间外预测共{len(oof)}场，原始LL={description['earlier_meta_oos']['stack_raw']['log_loss']:.6f}；在这些概率上学习温度后，该校准拟合集LL={description['earlier_meta_oos']['stack_temperature_fit_apparent_not_external_test']['log_loss']:.6f}。后者是温度拟合内分数，不是另一份独立检验。",'',
        f"较晚季校准前LL={result['scores']['integrated_raw']['log_loss']:.6f}、校准后={result['scores']['integrated_calibrated']['log_loss']:.6f}；它没有改善这一季，也未改变首选。较早小样本联合器与最终全季联合器之间存在校准迁移风险，不能说一个共享温度必然有益。没有据此重新拟合温度。",'',
        '## 比分一致化学到了什么','',
        '本轮保持P(比分|主胜/平局/客胜)不变，因此完整比分NLL的变化等于胜平负NLL变化。这是该分解的数学性质，逐场已存结果最大差为'+f'{identity_error:.3g}'+'. 两项分数同时变差不是两份独立的失败证据。', '',
        '这一步的价值是概率一致：三类、比分、总球和让球出自同一分布。它不会纠正更差的胜平负输入，也不会凭空改善区域内比分结构；新均值和总球必须重算。','',
        '## 下一步需要守住的机制','',
        '- 联合学习、校准和比分一致化已经分开落地，可分别复用，避免下次重新从接口开始。',
        '- 保留各成员信息来源和覆盖；相同信息的多种意见不自动增加证据强度。较早成员相关矩阵保存在JSON，仅用于描述。',
        '- 这套组合改变了原有类别取舍，尚未学到足以迁移的整体改善。现有数据未单独确认是哪项输入或设置导致，不能靠现象直接作因果归因。',
        '- 当前到这一套完成的整合与定位为止，不追加删成员、调C或改时间窗的试验；后续工作应针对明确缺口继续。']
    (OUT/'变化定位与可复用结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'class_changes':class_changes,'earlier_oof':description['earlier_meta_oos'],
                     'score_loss_identity_max_error':identity_error},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
