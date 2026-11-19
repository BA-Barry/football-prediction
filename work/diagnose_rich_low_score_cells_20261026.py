"""Fixed low-score cells of saved current121 forecasts; no parameter fitting."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from football_model.forecast_diagnostics import moving_block_means
from diagnose_rich_1x2_bias_20261026 import read, write, indices, PERIODS


CELLS = ('0:0', '0:1', '1:0', '1:1', 'other_draw', 'other_decisive')


def cells(row):
    h,a = row['rich_rates']
    zero = math.exp(-h-a)
    p = np.asarray([zero, zero*a, zero*h, zero*h*a,
                    row['rich_probabilities'][1]-zero*(1+h*a), 0.])
    p[-1] = 1-p[:-1].sum()
    if np.any(p < 0) or abs(p.sum()-1)>1e-10:
        raise ValueError('Full-score Poisson cells and saved draw probability must agree')
    score = row['actual_score']
    key = f'{score[0]}:{score[1]}'
    outcome = CELLS.index(key) if key in CELLS else 4 if score[0]==score[1] else 5
    return p, np.eye(6)[outcome]


def describe(rows):
    p,y = (np.asarray(v) for v in zip(*(cells(r) for r in rows)))
    delta = (p-y)*100
    block = moving_block_means([r['date'] for r in rows], delta, block_weeks=4)
    return {'n':len(rows), 'cells': {name:{'mean_probability':float(p[:,i].mean()),
        'expected_n':float(p[:,i].sum()), 'observed_n':int(y[:,i].sum()),
        'actual_frequency':float(y[:,i].mean()), 'bias_pp':float(delta[:,i].mean()),
        'conditional_4week_interval_pp':block['interval_95'][i]} for i,name in enumerate(CELLS)},
        'cell_order':list(CELLS), 'conditional_blocks':block}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',type=Path,required=True)
    out=parser.parse_args().diagnostic.resolve()
    rows=read(out/'提前3日_已有预测与固定分组.json')['records']
    method={'created_at':datetime.now(timezone.utc).isoformat(), 'source':str(out),
        'purpose':'Explain current121 draw shortfall using fixed existing DC score cells before proposing any distribution change',
        'horizon_days':3,'periods':list(PERIODS),'cells':list(CELLS),
        'groups':['all','balanced','clear_favourite'],
        'probabilities':'Analytic existing independent-Poisson 00/01/10/11; residual other draws from saved draw probability',
        'stopping':'Exactly six fixed season/group summaries; no fits, correction, parameter search or target predictions',
        'scope':'Descriptive previously viewed history, no causal attribution or DC adoption decision',
        'new_fits':0,'predictions_replaced':False}
    write(out/'低比分逐格方法.json',method)
    result={'method':method,'by_period':{}}
    lines=['# 当前121模型：低比分逐格偏差', '',
        '沿用既定提前3日、两季和率比1.5分组；只分解已保存概率，不训练、校准或改变比分分布。'
        '偏差为预测减实际。其他平局包含2:2及更高平局；其余比分构成其他决胜。', '',
        '| 赛季／分组 | 场数 | 比分 | 预测场数 | 实际场数 | 偏差（百分点） | 4周条件区间 |',
        '|---|---:|---|---:|---:|---:|---|']
    for period in PERIODS:
        local=[r for r in rows if r['period']==period]
        result['by_period'][period]={}
        for group in ('all','balanced','clear_favourite'):
            part=[local[i] for i in indices(local,group)]
            v=describe(part)
            result['by_period'][period][group]=v
            for name,cell in v['cells'].items():
                ci=cell['conditional_4week_interval_pp']
                lines.append(f"| {period}／{group} | {len(part)} | {name} | {cell['expected_n']:.1f} | "
                    f"{cell['observed_n']} | {cell['bias_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ['', '标准DC修正对0:0与1:1施加相等绝对量的变动，对0:1与1:0施加相反的等量变动。'
              '单格缺口不证明该修正适合，也不能以本次实际缺口直接设置参数。', '',
              '本轮到六个固定汇总为止；未学习rho，没有改主客进球率、正式概率或LLM输出。'
              '四周区间为既有预测条件下的描述，不代表完整参数不确定性。']
    write(out/'低比分逐格结果.json',result)
    (out/'低比分逐格定位.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:24]),flush=True)


if __name__=='__main__':
    main()
