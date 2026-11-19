"""Fetch official FPL observations without projecting historical match labels.

Raw API bytes remain unchanged in raw/. All derived fixture output is restricted
to fixtures after the actual observation time that have not started or finished.
Player status and source percentages are preserved separately from text labels.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'data/epl_personnel_live_20261029'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'model_v2/src'))
from football_model.personnel_news_semantics import extract_personnel_news


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fetch(name):
    attempts = []
    base = 'https://fantasy.premierleague.com/api/' + name
    # A maximum of one same-official-host alternate path, only after failure.
    for attempt, url in enumerate((base + '/', base)):
        rec = {'source_url': url, 'requested_at_utc': now(), 'attempt': attempt + 1}
        body = None
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0 FPL research observation',
                                        'Accept': 'application/json', 'Cache-Control': 'no-cache'})
            with urlopen(req, timeout=45) as response:
                body = response.read()
                rec.update(http_status=response.status, final_url=response.url,
                           response_headers={k: v for k, v in response.headers.items()
                                             if k.lower() in ('date', 'last-modified', 'age', 'etag',
                                                              'content-type', 'cache-control')})
            rec['received_at_utc'] = now()
            payload = json.loads(body)
            expected = isinstance(payload, dict) and isinstance(payload.get('elements'), list) and isinstance(payload.get('teams'), list) if name == 'bootstrap-static' else isinstance(payload, list)
            if not expected:
                raise ValueError('Unexpected official API JSON structure')
            target = OUT / 'raw' / (name + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            rec.update(status='downloaded', raw_file=str(target), bytes=len(body))
            attempts.append(rec)
            return payload, attempts
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            rec.update(received_at_utc=now(), status='failed', error=str(exc))
            if isinstance(exc, HTTPError):
                rec['http_status'] = exc.code
                body = exc.read()
            if body is not None:
                target = OUT / 'raw' / (name + '.attempt' + str(attempt + 1) + '.response')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                rec.update(raw_error_file=str(target), bytes=len(body))
            attempts.append(rec)
    return None, attempts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_started = now()
    with ThreadPoolExecutor(max_workers=2) as pool:
        bootstrap_job = pool.submit(fetch, 'bootstrap-static')
        fixtures_job = pool.submit(fetch, 'fixtures')
        bootstrap, bootstrap_requests = bootstrap_job.result()
        fixtures, fixture_requests = fixtures_job.result()
    observed_at = now()
    manifest = {'run_started_at_utc': run_started, 'observation_completed_at_utc': observed_at,
                'source_urls': ['https://fantasy.premierleague.com/api/bootstrap-static/',
                                'https://fantasy.premierleague.com/api/fixtures/'],
                'requests': bootstrap_requests + fixture_requests,
                'raw_may_contain_historical_labels': True,
                'historical_match_scores_projected_or_reported': False}
    if bootstrap is None or fixtures is None:
        manifest.update(status='official_source_unavailable', current_2026_27_verified=False)
        write(OUT / '获取记录.json', manifest)
        print(json.dumps({'status': manifest['status'], 'manifest': str(OUT / '获取记录.json'),
                          'requests': manifest['requests']}, ensure_ascii=False))
        return

    teams = {t['id']: t for t in bootstrap['teams']}
    events = bootstrap.get('events', [])
    event_dates = sorted(e['deadline_time'] for e in events if e.get('deadline_time'))
    # Only kickoff/date and start/finish flags are inspected from raw fixtures.
    fixture_dates = sorted(f['kickoff_time'] for f in fixtures if f.get('kickoff_time'))
    expected_start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    expected_end = datetime(2027, 8, 1, tzinfo=timezone.utc)
    event_season_matches = bool(event_dates) and all(expected_start <= utc(d) < expected_end for d in event_dates)
    fixture_season_matches = bool(fixture_dates) and all(expected_start <= utc(d) < expected_end for d in fixture_dates)
    cutoff = utc(observed_at)
    future = [f for f in fixtures if f.get('kickoff_time') and utc(f['kickoff_time']) > cutoff
              and f.get('started') is False and f.get('finished') is False]
    season_ok = event_season_matches and fixture_season_matches and len(teams) == 20
    verification = {'expected_season': '2026/27', 'verified_current_2026_27': season_ok,
                    'event_deadline_range': [event_dates[0], event_dates[-1]] if event_dates else None,
                    'fixture_kickoff_range': [fixture_dates[0], fixture_dates[-1]] if fixture_dates else None,
                    'event_dates_match_expected_season': event_season_matches,
                    'fixture_dates_match_expected_season': fixture_season_matches,
                    'teams_count': len(teams),
                    'teams': [{'source_team_id': t['id'], 'source_team_code': t.get('code'),
                               'source_team_name': t['name']} for t in bootstrap['teams']],
                    'current_event_ids': [e['id'] for e in events if e.get('is_current')],
                    'next_event_ids': [e['id'] for e in events if e.get('is_next')],
                    'future_unstarted_fixture_count_at_observation': len(future),
                    'interpretation': ('Official content dates are consistent with the expected current season.' if season_ok else
                                       'Official content dates do not verify the expected current season; do not use this as current personnel or upcoming target data.')}
    players = []
    for player in bootstrap['elements']:
        team = teams.get(player.get('team'), {})
        players.append({'source_player_code': player.get('code'), 'source_player_id': player.get('id'),
                        'source_team_id': player.get('team'), 'source_team_code': team.get('code'),
                        'source_team_name': team.get('name'),
                        'first_name': player.get('first_name'), 'second_name': player.get('second_name'),
                        'web_name': player.get('web_name'), 'source_status': player.get('status'),
                        'news': player.get('news'), 'news_added': player.get('news_added'),
                        'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                        'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                        'semantics': extract_personnel_news(player.get('news'), player.get('status'))})
    limits = ['Source status is a source label, not a confirmed health state.',
              'Source chance fields are not calibrated participation, starting or match-outcome probabilities.',
              'Text categories describe wording; no effect size or strength adjustment is fitted or applied.',
              'No archived player minutes, starts, future news frequency, or last-seen duration is exported.',
              'Current observation must not be used retroactively at an earlier forecast cutoff.']
    write(OUT / '当前球员新闻.json', {'observed_at_utc': observed_at, 'source_url': manifest['source_urls'][0],
                                    'season_verification': verification, 'usable_as_current_personnel': season_ok,
                                    'limits': limits, 'players': players})
    fixture_projection = []
    for fixture in sorted(future, key=lambda f: (f['kickoff_time'], f['id'])):
        home, away = teams.get(fixture['team_h'], {}), teams.get(fixture['team_a'], {})
        fixture_projection.append({'source_fixture_id': fixture['id'], 'event': fixture.get('event'),
                                   'kickoff_time': fixture['kickoff_time'],
                                   'home_source_team_id': fixture['team_h'], 'home_source_team_code': home.get('code'),
                                   'home_source_team_name': home.get('name'),
                                   'away_source_team_id': fixture['team_a'], 'away_source_team_code': away.get('code'),
                                   'away_source_team_name': away.get('name'),
                                   'started': fixture.get('started'), 'finished': fixture.get('finished')})
    write(OUT / '未来未开场赛程.json', {'observed_at_utc': observed_at, 'source_url': manifest['source_urls'][1],
                                      'season_verified_current_2026_27': season_ok,
                                      'usable_as_current_targets': season_ok,
                                      'filter': 'kickoff_time > actual observation UTC; started is false; finished is false',
                                      'fixtures': fixture_projection})
    news_dates = sorted(p['news_added'] for p in players if p['news_added'])
    manifest.update(status='downloaded_current_season' if season_ok else 'downloaded_season_mismatch',
                    current_2026_27_verified=season_ok, season_verification=verification,
                    player_count=len(players), source_status_counts=dict(Counter(p['source_status'] for p in players)),
                    nonempty_news_count=sum(bool(p['news'] and p['news'].strip()) for p in players),
                    source_news_added_range=[news_dates[0], news_dates[-1]] if news_dates else None,
                    semantics_category_counts=dict(Counter(c for p in players for c in p['semantics']['text_categories'])),
                    limits=limits)
    write(OUT / '获取记录.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'observation_completed_at_utc',
                                             'current_2026_27_verified', 'season_verification',
                                             'player_count', 'nonempty_news_count', 'source_news_added_range',
                                             'semantics_category_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
