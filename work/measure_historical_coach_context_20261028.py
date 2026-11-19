"""Join archived fixture identities and expose only earlier coach observations."""
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import unicodedata

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'data/historical_context_20261028/coach_sources'
TARGETS = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练/扩样合并特征.json.gz'
TM_COMP = {'E0':'GB1','D1':'L1','F1':'FR1','I1':'IT1','SP1':'ES1'}
# Explicit reviewed aliases, keyed by the project's existing source team ID.
# No score or approximate-string decision is used to resolve identities.
ALIASES = {
 'E0': """Arsenal=11;Aston Villa=405;Bournemouth=989;Brentford=1148;Brighton=1237;Burnley=1132;Chelsea=631;Crystal Palace=873;Everton=29;Fulham=931;Ipswich=677;Leeds=399;Leicester=1003;Liverpool=31;Luton=1031;Man City=281;Man United=985;Newcastle=762;Norwich=1123;Nott'm Forest=703;Sheffield United=350;Southampton=180;Sunderland=289;Tottenham=148;Watford=1010;West Ham=379;Wolves=543""",
 'D1': """Augsburg=167;Bayern Munich=27;Bielefeld=10;Bochum=80;Darmstadt=105;Dortmund=16;Ein Frankfurt=24;FC Koln=3;Freiburg=60;Greuther Furth=65;Hamburg=41;Heidenheim=2036;Hertha=44;Hoffenheim=533;Holstein Kiel=269;Leverkusen=15;M'gladbach=18;Mainz=39;RB Leipzig=23826;Schalke 04=33;St Pauli=35;Stuttgart=79;Union Berlin=89;Werder Bremen=86;Wolfsburg=82""",
 'F1': """Ajaccio=1147;Angers=1420;Auxerre=290;Bordeaux=40;Brest=3911;Clermont=3524;Le Havre=738;Lens=826;Lille=1082;Lorient=1158;Lyon=1041;Marseille=244;Metz=347;Monaco=162;Montpellier=969;Nantes=995;Nice=417;Paris FC=10004;Paris SG=583;Reims=1421;Rennes=273;St Etienne=618;Strasbourg=667;Toulouse=415;Troyes=1095""",
 'I1': """Atalanta=800;Bologna=1025;Cagliari=1390;Como=1047;Cremonese=2239;Empoli=749;Fiorentina=430;Frosinone=8970;Genoa=252;Inter=46;Juventus=506;Lazio=398;Lecce=1005;Milan=5;Monza=2919;Napoli=6195;Parma=130;Pisa=4172;Roma=12;Salernitana=380;Sampdoria=1038;Sassuolo=6574;Spezia=3522;Torino=416;Udinese=410;Venezia=607;Verona=276""",
 'SP1': """Alaves=1108;Almeria=3302;Ath Bilbao=621;Ath Madrid=13;Barcelona=131;Betis=150;Cadiz=2687;Celta=940;Elche=1531;Espanol=714;Getafe=3709;Girona=12321;Granada=16795;Las Palmas=472;Leganes=1244;Levante=3368;Mallorca=237;Osasuna=331;Oviedo=2497;Real Madrid=418;Sevilla=368;Sociedad=681;Valencia=1049;Valladolid=366;Vallecano=367;Villarreal=1050""",
}


def read_gzip(path):
    with gzip.open(path, 'rt',encoding='utf-8') as f:
        return json.load(f)


def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def label(value):
    """Whitespace and Unicode canonicalization only; no fuzzy person matching."""
    return ' '.join(unicodedata.normalize('NFC',value).split()) or None


def context_for(history, cutoff):
    before = history[:bisect_left([r['date'] for r in history], cutoff)]
    if any(r['date'] >= cutoff for r in before):
        raise ValueError('Non-earlier coach observation')
    empty = {'history_present':0, 'latest_coach_label_known':0,
        'latest_observation_age_days':None, 'last_known_coach_age_days':None,
        'consecutive_same_name_games':None, 'observed_change_age_days':None,
        'streak_left_censored':None, 'continuity_broken_by_missing':None}
    if not before:
        return {'features':empty,'last_observed_coach_name':None,'last_match_date':None,
            'source_game_ids':[],'streak_first_observed_date':None,
            'previous_different_coach_last_match_date':None, 'change_first_observed_date':None,
            'eligible_history_n':0,'max_source_date':None,'latest_record_state':'no_earlier_league_match'}
    latest = before[-1]
    latest_date = date.fromisoformat(latest['date'])
    date_counts = Counter(r['date'] for r in before)
    latest_same_day = [r for r in before if r['date']==latest['date']]
    known = next((r for r in reversed(before) if r['coach'] is not None),None)
    refs = {latest['game_id']}
    if known is not None:
        refs.add(known['game_id'])
    features = empty | {'history_present':1,
        'latest_observation_age_days':(date.fromisoformat(cutoff)-latest_date).days,
        'last_known_coach_age_days':None if known is None else
            (date.fromisoformat(cutoff)-date.fromisoformat(known['date'])).days}
    result = {'features':features, 'last_observed_coach_name':None if known is None else known['coach'],
        'last_match_date':latest['date'],'source_game_ids':[],
        'streak_first_observed_date':None, 'previous_different_coach_last_match_date':None,
        'change_first_observed_date':None, 'eligible_history_n':len(before),
        'max_source_date':latest['date'], 'latest_record_state':'known'}
    if latest['coach'] is None or len(latest_same_day)>1:
        features['continuity_broken_by_missing']=1
        result['latest_record_state']='missing_latest_coach' if latest['coach'] is None else 'ambiguous_same_date_matches'
        if len(latest_same_day)>1:
            result['last_observed_coach_name']=None
            features['last_known_coach_age_days']=None
        refs.update(r['game_id'] for r in latest_same_day)
        result['source_game_ids']=sorted(refs)
        return result
    features['latest_coach_label_known']=1
    count=0
    first=latest
    previous_different=None
    broken=False
    previous_season=int(latest['season'])
    for row in reversed(before):
        if (row['coach'] is None or previous_season-int(row['season'])>1 or date_counts[row['date']]>1):
            broken=True
            refs.add(row['game_id'])
            break
        if row['coach']!=latest['coach']:
            previous_different=row
            refs.add(row['game_id'])
            break
        count+=1
        first=row
        refs.add(row['game_id'])
        previous_season=int(row['season'])
    features['consecutive_same_name_games']=count
    features['continuity_broken_by_missing']=int(broken)
    features['streak_left_censored']=int(previous_different is None)
    if previous_different is not None:
        result['previous_different_coach_last_match_date']=previous_different['date']
        result['change_first_observed_date']=first['date']
        features['observed_change_age_days']=(date.fromisoformat(cutoff)-date.fromisoformat(first['date'])).days
    result['streak_first_observed_date']=first['date']
    result['source_game_ids']=sorted(refs)
    return result


def main():
    with gzip.open(OUT/'games.csv.gz','rt',encoding='utf-8',newline='') as f:
        raw=list(csv.DictReader(f))
    with gzip.open(OUT/'clubs.csv.gz','rt',encoding='utf-8',newline='') as f:
        clubs={r['club_id']:r for r in csv.DictReader(f)}
    targets=read_gzip(TARGETS)['rows']
    aliases={}
    for comp,text in ALIASES.items():
        for item in text.split(';'):
            name,club_id=item.split('=')
            if clubs[club_id]['domestic_competition_id'] != TM_COMP[comp]:
                raise ValueError('Explicit club alias has wrong competition')
            aliases[f'football-data:{comp}:{name}']={'club_id':club_id, 'source_name':clubs[club_id]['name'],
                'competition':comp, 'transfermarkt_competition':TM_COMP[comp],
                'source_url':clubs[club_id]['url'], 'mapping_method':'explicitly reviewed name alias; no fuzzy or score selection'}
    missing=sorted({r[side] for r in targets for side in ('home','away')}-aliases.keys())
    if missing:
        raise ValueError(f'Unmapped team identity: {missing}')
    write(OUT/'project_club_aliases.json',aliases)
    date_corrections=[{
        'game_id':'3886463','archive_date':'2022-12-30','canonical_date':'2022-12-29',
        'competition_id':'ES1','home_club_id':'13','away_club_id':'1531',
        'project_match_id':'football-data:SP1:2022/23:Ath Madrid|Elche',
        'official_source_url':'https://en.atleticodemadrid.com/postpartidos/atletico-de-madrid-elche-cf-31-12-2022-18-46',
        'official_source_url_2':'https://www.atleticodemadrid.com/prepartidos/atletico-de-madrid-elche-cf-31-12-2022-00-00',
        'basis':'Official match timeline explicitly dates final moments Thu Dec29 23:23+0100, matching FD date; original URL slug is not treated as event date.',
        'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'score_used':False,'raw_file_modified':False}]
    correction_ids={r['game_id'] for r in date_corrections}
    relevant=[dict(r) for r in raw if r['competition_id'] in TM_COMP.values()]
    for fix in date_corrections:
        matches=[r for r in relevant if r['game_id']==fix['game_id']]
        if len(matches)!=1 or any(matches[0][k]!=fix[k] for k in ('competition_id','home_club_id','away_club_id')):
            raise ValueError('Explicit date repair identity changed')
        if matches[0]['date']!=fix['archive_date']:
            raise ValueError('Raw archive date changed; revisit the documented correction')
        matches[0]['date']=fix['canonical_date']
    write(OUT/'verified_date_corrections.json',date_corrections)
    if len({r['game_id'] for r in relevant})!=len(relevant):
        raise ValueError('Repeated source game IDs')
    by_identity=defaultdict(list)
    histories=defaultdict(list)
    for row in relevant:
        by_identity[(row['competition_id'],row['date'],row['home_club_id'],row['away_club_id'])].append(row)
        for side in ('home','away'):
            histories[(row['competition_id'],row[f'{side}_club_id'])].append({
                'game_id':row['game_id'],'date':row['date'],'season':row['season'],
                'coach':label(row[f'{side}_club_manager_name'])})
    for rows in histories.values():
        rows.sort(key=lambda r:(r['date'],r['game_id']))
    output,issues,joined=[],[],[]
    cached={}
    for target in targets:
        comp=TM_COMP[target['competition']]
        home,away=(aliases[target[s]]['club_id'] for s in ('home','away'))
        candidates=by_identity.get((comp,target['date'],home,away),[])
        if len(candidates)!=1:
            issues.append({'match_id':target['match_id'],'date':target['date'],'home':target['home'],'away':target['away'],
                'reason':'missing_exact_fixture' if not candidates else 'ambiguous_exact_fixture',
                'source_game_ids':[r['game_id'] for r in candidates]})
        else:
            joined.append({'match_id':target['match_id'],'source_game_id':candidates[0]['game_id'],
                'date':target['date'],'home':target['home'],'away':target['away'],
                'home_club_id':home,'away_club_id':away})
        sides=[]
        cutoff=target['forecast_before_date']
        for team,club_id in ((target['home'],home),(target['away'],away)):
            key=(comp,club_id,cutoff)
            if key not in cached:
                cached[key]=context_for(histories[(comp,club_id)],cutoff)
            context=cached[key]
            if context['max_source_date'] is not None and not context['max_source_date']<cutoff:
                raise ValueError('Target or future match entered prior coach context')
            if len(candidates)==1 and candidates[0]['game_id'] in context['source_game_ids']:
                raise ValueError('Target coach record used in pre-match context')
            sides.append({'team':team,'club_id':club_id,**context})
        output.append({k:target[k] for k in ('match_id','date','forecast_before_date','home','away','competition','period')}
            |{'coach_context':sides,'target_source_game_id':candidates[0]['game_id'] if len(candidates)==1 else None,
                'fixture_join_status':('verified_date_corrected_unique' if candidates[0]['game_id'] in correction_ids else 'exact_unique') if len(candidates)==1 else 'unresolved'})
    document={'created_at_utc':datetime.now(timezone.utc).isoformat(),'rows':output,
        'scope':'Same-domestic-league archived coach labels on dates strictly before F=D-3. No target same-match manager fields.',
        'source_files':[str(OUT/'games.csv.gz'),str(OUT/'project_club_aliases.json')],
        'verified_date_corrections':date_corrections,
        'historical_first_publication_reconstructed':False,
        'numeric_semantics':{'history_present':'Earlier same-league match exists.',
            'latest_coach_label_known':'Most recent earlier match has exactly one nonempty coach label.',
            'latest_observation_age_days':'F minus latest earlier league match date.',
            'last_known_coach_age_days':'F minus latest earlier nonempty coach label date; label can be stale.',
            'consecutive_same_name_games':'Continuous observed prior league games with same source coach label; not actual tenure.',
            'observed_change_age_days':'F minus first match in current same-name streak following an explicitly different earlier label. None if left-censored/broken.',
            'streak_left_censored':'No immediately earlier different coach establishes observed beginning; not an appointment fact.',
            'continuity_broken_by_missing':'Empty coach label, same-date ambiguity or missing league-season breaks observed continuity.'},
        'limits':['Source stops July2026; not current 2026/27 coach information.',
            'Source labels lack unique person IDs. Spelling/name changes can masquerade as changes.',
            'Appointment, departure and announcement dates are not supplied; observed match coach is not guaranteed target coach.',
            'Missing values, ambiguous latest matches and absent league seasons break continuity; unknown is not zero.',
            'The opening source boundary is left-censored. Earliest match under a label is only an observed lower bound.',
            'Only domestic top-league matches; coaches may have changed outside observed competition or between matches.',
            'Archived snapshot may contain retrospective corrections; no claim all labels were published by historical F.',
            'No game goals, standings, formations, current club coach or target-match coach enter context features.']}
    with gzip.open(OUT/'coach_context_lead3.json.gz','wt',encoding='utf-8') as f:
        json.dump(document,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    write(OUT/'fixture_join.json',{'matched':joined,'unresolved':issues,'matching_fields':['competition','date','explicit_home_club_id','explicit_away_club_id'],
        'score_used_for_matching':False, 'target_manager_exported':False})
    summary={'created_at_utc':document['created_at_utc'],'target_n':len(targets),
        'matched_unique_n':len(joined),
        'exact_unique_join_n':sum(r['source_game_id'] not in correction_ids for r in joined),
        'verified_date_corrected_join_n':sum(r['source_game_id'] in correction_ids for r in joined),
        'unresolved_n':len(issues),'unresolved':issues, 'aliases_n':len(aliases),
        'side_context_n':2*len(output),'numeric_feature_n_per_side':8,
        'side_latest_known_n':sum(s['features']['latest_coach_label_known'] for r in output for s in r['coach_context']),
        'side_no_history_n':sum(not s['features']['history_present'] for r in output for s in r['coach_context']),
        'side_left_censored_n':sum(s['features']['streak_left_censored']==1 for r in output for s in r['coach_context']),
        'side_continuity_broken_n':sum(s['features']['continuity_broken_by_missing']==1 for r in output for s in r['coach_context']),
        'side_observed_change_known_n':sum(s['features']['observed_change_age_days'] is not None for r in output for s in r['coach_context']),
        'by_period':{period:{'n':sum(r['period']==period for r in output),
            'latest_coach_known_sides':sum(s['features']['latest_coach_label_known'] for r in output if r['period']==period for s in r['coach_context'])}
            for period in sorted({r['period'] for r in output})},
        'examples':output[:3], 'source_ids_strictly_earlier':True,'target_match_manager_used':False,
        'source_full_game_n':len(raw),'source_top_league_game_n':len(relevant),'new_model_fits':0}
    write(OUT/'coach_context_coverage.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('examples','unresolved','by_period')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
