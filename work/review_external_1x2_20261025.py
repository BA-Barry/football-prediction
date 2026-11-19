"""Save bounded source review and a numerical label-dependence example."""
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/胜平负核心调研_20261025'
OUT.mkdir(parents=True, exist_ok=True)


def cascade(p_home, p_draw_given_not_home):
    return [p_home, (1-p_home)*p_draw_given_not_home,
            (1-p_home)*(1-p_draw_given_not_home)]


example = {
    'same_features_and_fitted_stage_models': True,
    'stage1_home_probability': .4,
    'stage2_model_draw_given_not_home_probability': .9,
    'if_test_label_home': cascade(.4, .5),
    'if_test_label_draw_or_away': cascade(.4, .9),
    'correct_evaluation_for_all_rows': cascade(.4, .9),
    'meaning': 'Only changing the held-out label changes saved 1X2 probabilities in the source inference path; this cannot be used for a future fixture',
}
rows = [
    {
        'repo': 'lrivals/FootWork',
        'source_depth': 'raw hierarchical inference source and author documentation read; repository not reproduced',
        'sources': [
            'https://raw.githubusercontent.com/lrivals/FootWork/main/src/Models/Hierarchical/Football_Match_Hierarchical_Pipeline.py',
            'https://raw.githubusercontent.com/lrivals/FootWork/main/docs/hierarchical_pipeline.md',
        ],
        'useful': 'Conditional probabilities can be multiplied to form one coherent three-class vector',
        'finding': 'mask_test_s2 is computed from y_test_full; stage2 predictions applied only to realised non-home-win test rows, with .5 filled for actual home wins',
        'lines_observed': '174-192 and 200-207 in the retrieved source',
        'decision': 'Do not adopt pipeline or reported scores; legal cascade formula alone is not evidence of improved 1X2',
        'simulation': example,
        'license_status': 'not independently verified; no source code copied into model',
    },
    {
        'repo': 'dank0i/football-forecast',
        'source_depth': 'README methodology and pyproject read; source tree not fully fetched or reproduced',
        'sources': ['https://github.com/dank0i/football-forecast',
                    'https://raw.githubusercontent.com/dank0i/football-forecast/main/pyproject.toml'],
        'useful': 'Report all feature blocks and failed upgrades alongside market and probabilistic scores',
        'finding': 'Main feature set includes the eleven actual starters and an embedding neural network; its scope differs from the current no-lineup research',
        'limitation': 'A flat residual gap or a market accuracy in one dataset does not prove a causal information deficit or universal accuracy ceiling',
        'decision': 'Borrow comparison questions only; do not copy headline rate, lineup route, neural model or dependency bundle',
        'license_status': 'not independently verified; no source code copied into model',
    },
    {
        'repo': 'wenjin06/football-outcome-forecasting-benchmark',
        'source_depth': 'README and repository layout only; source traversal/API access unavailable',
        'sources': ['https://github.com/wenjin06/football-outcome-forecasting-benchmark'],
        'useful': 'Keep uncertainty and market comparisons distinct from prediction edge claims',
        'finding': 'README describes a submitted manuscript, not independently verified acceptance, and a closing-odds benchmark',
        'limitation': 'Closing prices do not establish equal early-publication information; a negative finite benchmark cannot prove no method can beat a market',
        'decision': 'Evidence lead only, not a verified reusable learner or transferable ceiling',
        'license_status': 'repository page lists MIT; no source code copied into model',
    },
    {
        'repo': 'wzk1015/WorldCupArena',
        'source_depth': 'README, arXiv abstract and raw validate.py read; benchmark not reproduced',
        'sources': ['https://github.com/wzk1015/WorldCupArena',
                    'https://arxiv.org/abs/2607.18084',
                    'https://raw.githubusercontent.com/wzk1015/WorldCupArena/main/src/pipeline/validate.py'],
        'useful': 'Compare supplied evidence with self-retrieval under the same prospective target and record actual source timing',
        'finding': 'README claims top scoreline outcome must agree with 1X2 argmax; current code enforces headline outcome agreement and detailed lineup/reasoning fields',
        'limitation': 'An unconditional modal scoreline can be a draw while the sum of all home-win scorelines is largest; a conditional headline pick is a separate editorial convention, not a probability identity',
        'decision': 'Do not import consistency rules, lineup requirements, source snapshot hashes or lifecycle gates; existing dated prediction records suffice for this research',
        'license_status': 'repository page lists MIT; no source code copied into model',
    },
]
result = {'as_of_client_date': '2026-10-25', 'reviewed_projects': rows,
          'new_fits': 0, 'new_current_results_requests': 0,
          'installed_dependencies': [], 'code_copied_into_prediction_model': False,
          'access_limit': 'GitHub unauthenticated API returned rate limit errors and one network abort; raw official source and web pages used where available; no full-repository reproduction claimed'}
(OUT/'外部概率项目核读与取舍.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 外部概率项目核读：可借什么，排除什么', '',
         '本轮只研究公开作者资料，没有运行外部项目、安装依赖或修改当前预测。'
         '下列README结果均未在本项目数据上重现，不能移植其命中率或声称比市场更好。', '',
         '## FootWork：实际测试标签参与概率计算，排除整条管线', '',
         '已读[真实推断源码]('+rows[0]['sources'][0]+')：`mask_test_s2`依据测试真实结果是否主胜生成，'
         'Stage2只在真实非主胜行预测；真实主胜行填0.5。结果未知时无法执行这一分支。'
         '条件概率相乘的数学结构合法，但这份实际推断实现含测试标签信息。', '',
         '固定同一场输入、同一组已拟合模型，设Stage1主胜0.4、Stage2条件平局0.9：', '',
         '| 唯一变化 | 最终主胜 | 最终平局 | 最终客胜 |',
         '|---|---:|---:|---:|',
         '| 测试标签写主胜，源码填0.5 | 0.40 | 0.30 | 0.30 |',
         '| 测试标签写平局或客胜，源码执行Stage2 | 0.40 | 0.54 | 0.06 |', '',
         '这是按已读源码路径做的数值示例，不是运行作者模型所得的新成绩。'
         '真实结果只应参与事后评分，不能决定赛前概率。该问题足以排除当前管线的直接复用；'
         '不据此判断作者所有其他模块均有问题。额外F1阈值优化也不等于概率质量提升。', '',
         '## football-forecast：有用的反例表，不照搬应用范围', '',
         '作者[项目说明]('+rows[1]['sources'][0]+')公开列出了多种未改善的改动和市场参照，'
         '适合借鉴完整报告的习惯。主方案含实际首发11人和神经网络，超出本轮暂停首发的范围。'
         '“各子组差距相近”不能独自证明缺哪种信息，“该数据集市场约53%”也不是足球普遍上限。'
         '只核读了说明和[依赖元数据]('+rows[1]['sources'][1]+')，没有重跑或完整源码审计。', '',
         '## football-outcome-forecasting-benchmark：保留线索，不当已核实算法', '',
         '[作者仓库]('+rows[2]['sources'][0]+')描述了市场比较和难度研究，但使用收盘赔率。'
         '文稿状态按仓库自述仅为已提交；本轮未核实接收。源码遍历受访问限制，'
         '不把README中的“不胜市场”扩张为所有模型的能力边界，也不把它当赛前1／3日的同信息比较。', '',
         '## WorldCupArena：可借LLM对照设计，不能复制全部规则', '',
         '[项目]('+rows[3]['sources'][0]+')及[预印本]('+rows[3]['sources'][1]+')将提供证据和自主检索分开比较。'
         '可借这一问题设计；它不证明本项目LLM更准。README把最高概率比分与最高1X2类别的一致性列为要求，'
         '当前[验证源码]('+rows[3]['sources'][2]+')检查的是headline结果一致、并强制阵容和长篇理由。'
         '若headline只是从最高1X2类别中挑一个比分，这是展示约定；若把它当全比分分布众数必须一致，数学上不成立。'
         '本项目不复制这类规则或另增hash／冻结contract／gate，也不恢复首发要求。', '',
         '## 本轮访问边界', '',
         '未登录GitHub API受到速率限制，并有一次连接中止。能读取的原始文件和官方页面已逐条列出；'
         '不能将只见README的项目写成源码已全面核实，更不将未访问到的内容补造出来。', '']
(OUT/'外部概率项目核读与取舍.md').write_text('\n'.join(lines), encoding='utf-8')
print('saved 4 bounded project reviews and label-dependence numerical example; no fit')
