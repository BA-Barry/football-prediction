"""Plot the derived score-law bound and the existing rich-rate range."""
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0e

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/同日误差互补与结构定位_20261027'
data = json.loads((OUT / '互补与结构汇总.json').read_text(encoding='utf-8'))
critical = data['critical_total_mean_for_balanced_draw_equal_one_third']
observed = data['route_summary']['same_day_rich']
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
total = np.linspace(.15, 5.1, 700)
ax.plot(total, i0e(total), color='#2764a0', linewidth=2.5,
        label='固定总进球均值下，独立泊松允许的最大平局概率')
ax.axhline(1/3, color='#9d4b3b', linestyle='--', linewidth=1.5, label='1/3：平局成为首选的必要条件')
ax.axvline(critical, color='#9d4b3b', linestyle=':', linewidth=1.3)
ax.axvspan(observed['total_mean_min'], observed['total_mean_max'], color='#e9b158', alpha=.25,
           label='现有丰富模型1,732场总进球均值范围')
ax.annotate(f'边界约 {critical:.3f}', (critical, 1/3), xytext=(2.2, .48),
            arrowprops={'arrowstyle': '->', 'color': '#555555'})
ax.text(2.4, .63, '这批丰富模型预测全部位于边界右侧\n只重新分配主客进球率，平局仍无法首选',
        fontsize=11, color='#313131')
ax.set(xlim=(0, 5.1), ylim=(0, .9), xlabel='总进球均值 T = 主队均值 + 客队均值',
       ylabel='平局概率上界', title='当前单选局限来自输出结构的一部分，不能只靠调主客均值解决')
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
ax.legend(loc='upper right', bbox_to_anchor=(1.0, .48), frameon=False, fontsize=9)
ax.grid(axis='y', alpha=.15)
fig.savefig(OUT / '平局首选的结构边界.png', dpi=160)
plt.close(fig)
