from measure_historical_coach_context_20261028 import context_for


def row(day, coach, ident, season='2024'):
    return {'date':day, 'coach':coach, 'game_id':ident, 'season':season}


def test_target_and_cutoff_day_are_not_observations():
    past=[row('2024-09-01','A','p1'),row('2024-09-08','B','p2')]
    future=[row('2024-09-10','C','f1'),row('2024-09-13','D','target')]
    assert context_for(past+future,'2024-09-10')==context_for(past,'2024-09-10')
    future[-1]['coach']='Very different target coach'
    assert context_for(past+future,'2024-09-10')==context_for(past,'2024-09-10')


def test_observed_change_is_not_appointment():
    c=context_for([row('2024-08-01','A','1'),row('2024-08-10','B','2'),
                   row('2024-08-17','B','3')],'2024-08-20')
    assert c['features']['consecutive_same_name_games']==2
    assert c['change_first_observed_date']=='2024-08-10'
    assert c['previous_different_coach_last_match_date']=='2024-08-01'
    assert c['features']['observed_change_age_days']==10
    assert c['features']['streak_left_censored']==0
    first=context_for([row('2024-08-01','A','1')],'2024-08-03')
    assert first['features']['streak_left_censored']==1
    assert first['features']['observed_change_age_days'] is None


def test_missing_label_interrupts_observed_streak():
    c=context_for([row('2024-08-01','A','1'),row('2024-08-10',None,'2'),
                   row('2024-08-17','A','3')],'2024-08-20')
    assert c['features']['consecutive_same_name_games']==1
    assert c['features']['observed_change_age_days'] is None
    assert c['features']['continuity_broken_by_missing']==1
    assert c['features']['streak_left_censored']==1


def test_missing_latest_keeps_label_stale_not_current():
    c=context_for([row('2024-08-01','A','1'),row('2024-08-10',None,'2')],'2024-08-20')
    assert c['last_observed_coach_name']=='A'
    assert c['features']['last_known_coach_age_days']==19
    assert c['features']['latest_coach_label_known']==0
    assert c['features']['consecutive_same_name_games'] is None
    assert c['features']['observed_change_age_days'] is None


def test_ambiguous_same_day_interrupts_latest_and_older_streak():
    rows=[row('2024-08-01','A','1'),row('2024-08-01','A','2'),
          row('2024-08-10','A','3')]
    c=context_for(rows,'2024-08-20')
    assert c['features']['consecutive_same_name_games']==1
    assert c['features']['continuity_broken_by_missing']==1
    latest=context_for(rows[:2],'2024-08-20')
    assert latest['features']['latest_coach_label_known']==0
    assert latest['last_observed_coach_name'] is None


def test_absent_league_season_interrupts_streak():
    c=context_for([row('2022-05-01','A','1',season='2021'),
                   row('2024-08-17','A','3',season='2024')],'2024-08-20')
    assert c['features']['consecutive_same_name_games']==1
    assert c['features']['continuity_broken_by_missing']==1
    assert c['features']['streak_left_censored']==1


def test_no_history_does_not_imply_zero_tenure():
    c=context_for([],'2024-08-20')
    assert c['features']['history_present']==0
    assert c['features']['latest_observation_age_days'] is None
    assert c['features']['consecutive_same_name_games'] is None
    assert c['features']['observed_change_age_days'] is None
