"""Execute the authorized single early-period fit; all artifacts stay on G."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import traceback

from football_model.persistent_goal_difference import fit_static_panel, fit_dynamics

OUT = Path('G:/Codex/football-prediction-2026-10-22/outputs/持续状态净胜球研究_20261028')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    payload = json.load(gzip.open(OUT / '输入资料.json.gz', 'rt', encoding='utf-8'))
    static_rows = [r for r in payload['matches'] if r['season'] == '2022/23']
    dynamic_rows = [r for r in payload['matches'] if r['season'] == '2023/24']
    if max(r['date'] for r in dynamic_rows) >= min(r['F'] for r in payload['targets']):
        raise ValueError('Global parameter estimation overlaps a target forecast cutoff')
    # A completed earlier stage is retained on an interrupted run; never refit
    # merely because its later evaluation or document step was interrupted.
    static_path, dynamic_path = OUT / '静态初始化.json', OUT / '动态参数拟合.json'
    if static_path.exists():
        static = json.loads(static_path.read_text(encoding='utf-8'))
        print('Reusing completed static fit', flush=True)
    else:
        print(f'Static fit starts: {len(static_rows)} initial-season matches', flush=True)
        static = fit_static_panel(static_rows)
        save(static_path.name, static)
        print(f'Static fit completed: {static["optimizer"]}', flush=True)
    if dynamic_path.exists():
        print('Reusing completed dynamic fit', flush=True)
    else:
        print(f'Dynamic fit starts: {len(dynamic_rows)} next-season matches', flush=True)
        dynamic = fit_dynamics(static, dynamic_rows)
        save(dynamic_path.name, dynamic)
        print(f'Dynamic fit completed: {dynamic["optimizer"]}; parameters={dynamic["parameters"]}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('执行失败记录.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                                 'error': str(error), 'traceback': traceback.format_exc()})
        raise
