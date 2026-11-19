"""Publish the assistant's source-aware interpretation, without averaging forecasts."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
DATA = ROOT/'data/serie_a_collaborative_20261028'
OUT = ROOT/'outputs/协作预测与大样本方案_20261028'
ORDER = ('home_win', 'draw', 'away_win')
JUDGMENTS = {
    'espn:ita.1:401874993': {
        'name': 'AC米兰—博洛尼亚', 'direction': '米兰主胜为首选，优势有限',
        'shared_basis': '三条统计估计主胜约49%至51%，当季机会创造及对手校正后的近期进攻也偏米兰；这些是共享资料的多种表述。',
        'incremental_context': 'B01—B03提供博洛尼亚训练与康复状态，M01补米兰全赛事安排。它们没有确认本场缺阵或实际轮换，不能被解释成主胜的已拟合加分。',
        'disagreement': '新协作LLM主胜49%，与主线48.88%很接近；旧独立LLM59%更积极。输入政策及所见统计不同，这个变化不能全部归因为新闻，更不能称统计与新LLM独立相互验证。',
        'strongest_counterargument': '博洛尼亚有限客场样本的防守较好，最近一场机会表现改善；米兰的日程与即将到来的德比增加用人未知。',
        'usage': '首选只是三项中最大的一项，49%仍低于50%。主胜单选有实质平、负风险，不作为高把握结论。',
        'evidence_ids': ['B01', 'B02', 'B03', 'M01']},
    'espn:ita.1:401874965': {
        'name': '帕尔马—乌迪内斯', 'direction': '综合略偏帕尔马，保留主客方向分歧',
        'shared_basis': '真实进球主线偏客39.46%，xG线却偏主37.80%；这不是同一方向的幅度差。两队战绩相同，但乌迪内斯近期防守机会损失更重，对手校正后仍较差。',
        'incremental_context': 'P01仅确认帕尔马征召范围，U01补乌迪内斯赛程安排，均不足以独立决定方向。',
        'disagreement': '我更重视xG线及当前过程支持的有限主场优势，与新LLM40%主胜的方向一致；此处没有拟合近期取代长期的权重。不会把31.25/29.29/39.46与40/29/31平均后声称冲突已解决。',
        'strongest_counterargument': '帕尔马自身进攻极弱，乌迪内斯近期恶化的持续性未知；对莱切的异常一场拉动残差均值，八场开局不足以确定长期实力已反转。',
        'usage': '五场中方向分歧最值得跟踪的一场。综合偏主属于低确信度主观取舍，不能列入高把握串关支点。三项概率始终分别呈现。',
        'evidence_ids': ['P01', 'U01']},
    'espn:ita.1:401874773': {
        'name': '威尼斯—国际米兰', 'direction': '国际米兰客胜，五场中相对强弱最清楚',
        'shared_basis': '绝对机会水平及相对较早预期的表现都支持国米。进球线客胜79.02%，xG70.15%，丰富候选68.01%；差异提示幅度不稳定，并非79%已被证实。',
        'incremental_context': 'I01补全赛事安排；V02是新教练适应期自述。V01本队大名单列入Busio，与对手分析冲突，应保留入选事实及健康未知。',
        'disagreement': '我保留国米客胜方向，但采用较克制的文字判断；新LLM68%的主观数值不等于修复或校准了原79%。威尼斯的机会创造需要被认真对待，不把低进球自动视为以后必然反弹。',
        'strongest_counterargument': '热门的真实轮换、分钟和可用性不完整；威尼斯的创造机会能力强于其战绩显示，仍可能通过个别进球改变结果。',
        'usage': '可作为方向较明确的研究场次。没有刷新当前市场价格，本文不把方向优势直接转写成投注价值。',
        'evidence_ids': ['I01', 'I02', 'V01', 'V02', 'V03']},
    'espn:ita.1:401874988': {
        'name': '热那亚—尤文图斯', 'direction': '尤文客胜为首选，平局风险突出',
        'shared_basis': '尤文绝对防守过程明显优于热那亚，统计客胜51.06%至55.33%。近期进攻低于自身预期不能直接抹去绝对实力差。',
        'incremental_context': 'J01恢复、J02阵型选项、J03考虑后续赛程和J04对对手的看法来自同一发布会；并非四项独立利好或利空。J05补安排。',
        'disagreement': '新LLM21/28/51与进球线接近。我同意保留有限客胜优势，并强调机会产出偏低时的平局路径；不会用Sarr恢复或可换阵就推定攻击已提升。',
        'strongest_counterargument': '尤文最近进攻偏弱，热那亚防守并非全面崩溃；本队官方可用性资料缺失，定位球和临场管理的量级未测。',
        'usage': '客胜是第一选择，51%与确定性相距很大；平局单列28%，不把客胜与平局相加冒充客胜命中率。',
        'evidence_ids': ['J01', 'J02', 'J03', 'J04', 'J05']},
    'espn:ita.1:401874799': {
        'name': '蒙扎—那不勒斯', 'direction': '那不勒斯客胜为首选，优势中等',
        'shared_basis': '那不勒斯近期机会创造和允许机会都较好，对手校正后连续正向偏离；蒙扎主场成绩不应掩盖小样本与较高转化。数据仍支持客队。',
        'incremental_context': 'N03是教练停赛自述，N04为身体问题描述，N02/N05共同确认Vergara回归；没有可用的各自概率效应。N06/N07是意图与对手评价。',
        'disagreement': '进球线客胜59.30%，xG53.71%，新LLM55%。我保留客胜方向，同时保留客场表现和比赛管理的限制，不将55%称为学得联合概率。',
        'strongest_counterargument': '蒙扎主场有进球能力，那不勒斯客场优势远不如主场鲜明；教练不能在席和人员状态的实际影响未被估计。',
        'usage': '可跟踪的客胜倾向，平24%、主21%仍是实质结果路径。不能按近期数据良好就消除这些风险。',
        'evidence_ids': ['N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08']},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def probabilities(value):
    p = [value[k] for k in ORDER] if isinstance(value, dict) else list(value)
    if len(p) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in p) or abs(sum(p)-1) > 1e-10:
        raise ValueError('A complete H/D/A vector is required')
    return p


def pct(p):
    return ' / '.join(f'{x*100:.2f}' for x in p)


def main():
    now = datetime.now(timezone.utc)
    stat_path = ROOT/'outputs/今日意甲预测_20261028/统计预测与比分分布.json'
    llm_path = DATA/'情境协作LLM回答.json'
    stat, llm, evidence = read(stat_path), read(llm_path), read(DATA/'补充赛前证据.json')
    prior = read(ROOT/'data/serie_a_today_20261028/llm_response.json')
    estimates = {r['fixture']['match_id']:r for r in stat['records']}
    llms = {r['fixture_id']:r for r in llm['records']}
    original_llms = {r['fixture']['match_id']:r for r in prior['records']}
    if estimates.keys() != llms.keys() or estimates.keys() != JUDGMENTS.keys():
        raise ValueError('Fixture coverage differs')
    claims = {r['claim_id']:r for r in evidence['claims']}
    if not datetime.fromisoformat(llm['generated_at_utc']) <= now:
        raise ValueError('LLM record is not yet available')
    rows = []
    for ident, s in estimates.items():
        if datetime.fromisoformat(s['fixture']['kickoff']) <= now:
            raise ValueError('Synthesis must precede kickoff')
        j = JUDGMENTS[ident]
        if any(claims[ref]['match_id'] != ident for ref in j['evidence_ids']):
            raise ValueError('Synthesis cited another fixture')
        # Preserve provenance for original, informed, and model-only predictions.
        rows.append({'fixture': s['fixture'], 'name':j['name'],
            'statistical_generated_at': stat['generated_at'],
            'statistical_probabilities': {k: probabilities(s[k]['probabilities']) for k in
                ('goal_simple', 'independent_xg', 'rich_121_research')},
            'original_independent_llm': {'generated_at': prior['generated_at_utc'],
                'probabilities': probabilities(original_llms[ident]['probabilities'])},
            'new_contextual_llm': {'generated_at': llm['generated_at_utc'],
                'probabilities': probabilities(llms[ident]['probabilities']),
                'exposure': llm['exposure'], 'reasoning': llms[ident]['reasoning']},
            'assistant_synthesis': {k:v for k,v in j.items() if k != 'name'},
            'assistant_probability_vector': None, 'original_forecasts_changed': False})
    report = {'generated_at_utc': now.isoformat(), 'class_order': list(ORDER),
        'scope': 'Five unstarted Oct28 Italy fixtures / Oct29 Beijing fixtures',
        'assistant_exposure': {'statistical':True, 'original_and_new_llm':True,
            'official_evidence':True, 'current_market':False, 'target_results':False},
        'method': 'Source-aware assistant synthesis; no fitted or arithmetic probability fusion',
        'source_files': {'statistical':str(stat_path), 'llm':str(llm_path),
            'evidence':str(DATA/'补充赛前证据.json'), 'evidence_review':str(DATA/'本轮来源复核.json')},
        'records':rows, 'new_unique_fixtures':0, 'same_five_fixture_new_analysis_versions':5,
        'limits': ['The informed LLM is not an independent confirmation of the statistical model.',
            'Subjective contextual probabilities are not calibrated or established improvements.',
            'No new injury, tactical or motivation effect was fitted from five games.',
            'No updated market quotes are used; this is not an expected-value assessment.',
            'The expanded historical research model was not used for these forecasts.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'本轮五场综合研判.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n',encoding='utf-8')
    md = ['# 五场统计、情境LLM与助手综合', '', f'综合生成UTC：{now.isoformat()}；北京时间2026-10-29凌晨开球。', '',
        '本轮新增分析版本，仍是同五场比赛。原独立LLM和统计概率保持；新LLM已见三条统计估计、当季/对手校正资料及官方情境，所以不能称独立确认。助手不再造一组“联合概率”，而是解释依据与取舍。', '',
        '下表各格顺序均为 **主胜 / 平局 / 客胜（%）**。新LLM为未校准主观研究概率。', '',
        '|比赛|真实进球主线|独立xG统计线|新情境LLM|助手综合倾向|', '|---|---|---|---|---|']
    for r in rows:
        md.append(f"|{r['name']}|{pct(r['statistical_probabilities']['goal_simple'])}|{pct(r['statistical_probabilities']['independent_xg'])}|{pct(r['new_contextual_llm']['probabilities'])}|{r['assistant_synthesis']['direction']}|")
    md += ['', '## 逐场解释', '']
    labels = {'shared_basis':'共同依据', 'incremental_context':'新增情境', 'disagreement':'分歧与我的取舍',
        'strongest_counterargument':'最强反证', 'usage':'如何使用'}
    for r in rows:
        md += [f"### {r['name']}", '']
        for key,label in labels.items():
            md += [f"**{label}**：{r['assistant_synthesis'][key]}", '']
    md += ['## 证据与边界', '',
        '16个官方来源、24项陈述在UTC13:29做了针对复核；四份名单不等于完整伤停覆盖，10队完整可用性仍有缺口。教练观点和战术意图不当成执行事实。', '',
        '- [博洛尼亚训练](https://www.bolognafc.it/seduta-tattica-a-2-da-milan-bologna/)；[米兰赛程](https://www.acmilan.com/it/news/articoli/agenda/2026-10-01/ottobre-mese-di-calendari-intensi)。',
        '- [威尼斯本场名单](https://www.veneziafc.it/news/i-convocati-per-venezia-fc-inter)；[国米赛程](https://www.inter.it/it/notizie/inter-calendario-biglietti-partite-ottobre-san-siro)。',
        '- [尤文发布会](https://www.juventus.com/it/news/articoli/genoa-juventus-le-parole-di-luciano-spalletti-alla-vigilia)；[那不勒斯发布会](https://sscnapoli.it/allegri-a-monza-per-risalire-la-classifica-dobbiamo-migliorare-sia-nelle-prestazioni-che-nei-risultati/)。', '',
        '完整来源、日期、冲突、未知项见[补充证据](G:/Codex/football-prediction-2026-10-22/data/serie_a_collaborative_20261028/补充赛前证据.json)和[来源复核](G:/Codex/football-prediction-2026-10-22/data/serie_a_collaborative_20261028/本轮来源复核.md)。', '',
        '本轮未查询目标赛果、未刷新市场价格、未训练新闻影响系数。概率同向不证明误差独立，分歧幅度不是置信区间；没有以原13场权重替代证据综合。扩样研究候选没有用于本表，也没有把旧预测改写为新生成。']
    (OUT/'本轮五场综合研判.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'generated_at':now.isoformat(),'fixtures':len(rows), 'assistant_probability_vector':None,
        'path':str(OUT/'本轮五场综合研判.md')},ensure_ascii=False))


if __name__ == '__main__':
    main()
