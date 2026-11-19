"""Download author-published Transfermarkt historical games, not a live scraper."""
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import urllib.request

ROOT = REPOSITORY_ROOT
OUT = ROOT/'data/historical_context_20261028/coach_sources'
SOURCES = {
    'games.csv.gz': 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/games.csv.gz',
    'clubs.csv.gz': 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/clubs.csv.gz',
    'README.upstream.md': 'https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/README.md',
    'LICENSE.upstream.txt': 'https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/LICENSE',
    'dataset-metadata.upstream.json': 'https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/dataset-metadata.json',
}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, url in SOURCES.items():
        destination = OUT/name
        receipt_path = OUT/(name+'.receipt.json')
        if destination.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            receipts.append(receipt)
            print(json.dumps({'name': name, 'action':'reused','bytes': destination.stat().st_size}), flush=True)
            continue
        started = datetime.now(timezone.utc).isoformat()
        req = urllib.request.Request(url, headers={'User-Agent':'Football-personal-research/1.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            length = response.headers.get('Content-Length')
            if length and int(length)>100_000_000:
                raise ValueError('Requested individual file exceeds the intended small download scope')
            content = response.read(100_000_001)
            if len(content)>100_000_000:
                raise ValueError('Downloaded file exceeds scope')
            ended = datetime.now(timezone.utc).isoformat()
            headers = {k:response.headers.get(k) for k in ('Content-Type','Content-Length','Last-Modified')}
            receipt = {'url':url, 'observed_at_utc':ended, 'started_at_utc':started,
                'http_status':response.status, 'response_headers':headers,
                'bytes':len(content), 'stored_file':str(destination),
                'historical_first_publication_time':None,
                'note':'Current author-published archival snapshot; response last-modified is not each row first-publication time.'}
        destination.write_bytes(content)
        write(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({'name':name,'action':'downloaded','bytes':len(content)}),flush=True)
    with gzip.open(OUT/'games.csv.gz','rt',encoding='utf-8',newline='') as h:
        reader=csv.DictReader(h)
        fields=reader.fieldnames
        rows=list(reader)
    comps={'GB1':'E0','L1':'D1','FR1':'F1','IT1':'I1','ES1':'SP1'}
    selected=[r for r in rows if r['competition_id'] in comps and 2020<=int(r['season'])<=2025]
    from collections import Counter
    report={'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'upstream_repository':'https://github.com/dcaribou/transfermarkt-datasets',
        'upstream_status':'https://github.com/dcaribou/transfermarkt-datasets/discussions/383',
        'license':'CC0-1.0 as declared in repository LICENSE; upstream data rights not separately cleared',
        'games_n':len(rows),'games_columns':fields,
        'games_date_min':min(r['date'] for r in rows), 'games_date_max':max(r['date'] for r in rows),
        'target_seasons':'2020/21-2025/26 five leagues', 'target_coverage_n':len(selected),
        'target_by_competition_season':{f'{c}:{season}':sum(r['competition_id']==c and r['season']==str(season) for r in selected)
            for c in comps for season in range(2020,2026)},
        'target_coach_missing_sides':sum(not r[f'{side}_club_manager_name'] for r in selected for side in ('home','away')),
        'target_duplicate_ids':[key for key,count in Counter(r['game_id'] for r in selected).items() if count>1],
        'examples':[{k:r[k] for k in ('game_id','competition_id','season','date','home_club_id','away_club_id',
            'home_club_name','away_club_name','home_club_manager_name','away_club_manager_name','url')} for r in selected[:5]],
        'current_2026_27_usable':False,
        'safe_context_rule':'Only coaches attached to completed dates strictly before the forecast cutoff. Never read target-match coach, formation, position or outcome as a pre-match feature.',
        'limits':['Snapshot rows can have retrospective corrections; original per-row publication unavailable.',
            'Last observed coach is not guaranteed target coach: appointment between matches requires a separately dated source.',
            'Detected change date is first match observed under a different coach, not announcement or appointment date.',
            'Null coach remains unknown; club current coach fields are not historical values.',
            'No current match outcomes were queried; downloaded archive ends July2026.'],
        'receipts':receipts}
    write(OUT/'source_coverage.json',report)
    print(json.dumps({k:report[k] for k in ('games_n','games_date_max','target_coverage_n','target_coach_missing_sides','target_duplicate_ids')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
