import pytest

from football_model.personnel_news_semantics import extract_personnel_news


def test_status_is_not_a_text_diagnosis():
    r=extract_personnel_news('Unknown - Expected back 10 Oct','i')
    assert r['text_categories']==['unspecified_reason_text']
    assert r['normalized_return_date'] is None
    assert r['expected_return_wording']==['Expected back 10 Oct']
    assert r['clinical_state_confirmed'] is False


def test_a_transfer_is_not_counted_as_injury():
    r=extract_personnel_news('Permanent transfer to Borussia Dortmund','u')
    assert r['text_categories']==['transfer_reference']
    assert r['estimated_strength_loss'] is None


def test_parent_club_restriction_is_not_a_medical_problem():
    r=extract_personnel_news('Ineligible to face his parent club on 26/4.','u')
    assert r['text_categories']==['eligibility_reference']
    assert r['actual_target_absence_confirmed'] is False


def test_fitness_and_percentage_are_kept_separate():
    r=extract_personnel_news('Lack of match fitness - 75% chance of playing','d')
    assert r['text_categories']==['fitness_reference']
    assert r['source_percentage_mentions']==[{'source_span':'75%','number_as_written':'75'}]
    assert r['estimated_strength_loss'] is None


def test_empty_available_status_is_not_confirmed_health():
    for news in (None,'','  '):
        r=extract_personnel_news(news,'a')
        assert r['text_categories']==['no_news_text']
        assert r['clinical_state_confirmed'] is False


def test_negative_or_returning_wording_does_not_assert_current_injury():
    r=extract_personnel_news('No longer affected by a knee injury','a')
    assert 'medical_reference' in r['text_categories']
    assert r['change_or_negation_language'] is True
    assert r['actual_target_absence_confirmed'] is False


def test_injury_text_does_not_depend_on_status_label():
    for status in ('a','i','d','s','u','n',None):
        r=extract_personnel_news('Hamstring injury - Expected back 14 Sep',status)
        assert r['text_categories']==['medical_reference']
        assert r['raw_status']==status


@pytest.mark.parametrize('news', [
    'Has Joined Wigan Athletic on loan until the end of the season',
    'has joined Kocaelispor\u00a0 permanently.',
    'Has joined Sheffield United on a season-long',
    'Season long loan to \u00c7aykur Rizespor',
    'Season-long loan to  Ipswich Town',
    'On a season long loan to Rio Ave',
    'Half season loan to Carlisle United, expected back in January',
    'One year loan deal to Real Madrid',
    'Contract cancelled by mutual consent',
    'Contract ended',
    'Mutual termination of contract',
    'Left by mutual consent',
    'Left club by mutual agreement',
    'Has departed the club as a free agent.',
    'Loan ended early',
    'Loan terminated and has returned to  Leicester City',
    'Moved to Colorado Rapids on a permanent deal',
    'Signed by CD Tondela',
    'Transfer to Galatasaray on a permanent basis',
    'Recalled by Sevilla',
])
def test_explicit_transfer_lifecycle_wording_is_not_medical(news):
    r=extract_personnel_news(news,'u')
    assert r['text_categories']==['transfer_reference']
    assert r['raw_news']==news
    assert r['matched_source_spans'][0]['source_span'] in news
    assert r['actual_target_absence_confirmed'] is False


@pytest.mark.parametrize('news', [
    'Made unavailable for selection by his club. - Expected back 24 Aug',
    'Made unavailable for selection on 8/5. - Unknown return date',
    'Won\u2019t be considered for selection on 22/8. - Unknown return date',
    "Won't be considered for selection on 22/8.",
    'Will not be considered for selection on 22/8.',
])
def test_selection_restriction_has_no_assumed_medical_cause(news):
    r=extract_personnel_news(news,'n')
    assert r['text_categories']==['selection_restriction_reference']
    assert r['change_or_negation_language'] is True
    assert r['clinical_state_confirmed'] is False
    assert r['actual_target_absence_confirmed'] is False


@pytest.mark.parametrize('news', [
    'Season long to Almere City',
    'Loaened to Derby County',
    'Trasferred to Al Wahda',
    'U18 team',
    'Season-long training programme',
    'The club has an update',
])
def test_incomplete_or_ambiguous_text_is_not_filled_from_status(news):
    for status in ('i','u','a',None):
        r=extract_personnel_news(news,status)
        assert r['text_categories']==['unclassified_text']
        assert r['matched_source_spans']==[]
        assert r['raw_news']==news


@pytest.mark.parametrize('wording', [
    'Unknown return date', 'unknown return date', 'Not known return date',
    'Return date unknown', 'Return date not known',
    'Expected back 01 Jul', 'expected back in January', 'Out for two weeks',
    'Suspended until 01 Apr',
])
def test_timing_is_kept_as_written_without_normalization(wording):
    news='Unknown - '+wording
    r=extract_personnel_news(news,'i')
    assert r['expected_return_wording']==[wording]
    assert r['normalized_return_date'] is None
    assert r['text_categories']==['unspecified_reason_text']


@pytest.mark.parametrize('news', [
    'No longer suspended', 'Suspension lifted', 'Returned after a knee injury',
    'No injury reported', 'Not affected by illness',
])
def test_negation_and_change_do_not_confirm_any_active_condition(news):
    r=extract_personnel_news(news,'a')
    assert r['change_or_negation_language'] is True
    assert r['clinical_state_confirmed'] is False
    assert r['actual_target_absence_confirmed'] is False
    assert r['estimated_strength_loss'] is None


def test_suspension_spelling_and_until_date_are_text_only():
    for news in ('Suspension until 01 Apr','Suspended until 01 Apr'):
        r=extract_personnel_news(news,'a')
        assert r['text_categories']==['suspension_reference']
        assert r['expected_return_wording']==[news]
        assert r['normalized_return_date'] is None
        assert r['actual_target_absence_confirmed'] is False


def test_retirement_and_explicit_condition_can_have_two_reference_labels():
    news='Retired due to hereditary heart condition'
    r=extract_personnel_news(news,'u')
    assert r['text_categories']==['medical_reference','transfer_reference']
    assert r['clinical_state_confirmed'] is False
    assert r['estimated_strength_loss'] is None


def test_raw_spacing_and_decimal_percentage_are_preserved_without_probability():
    news='  Lack of match fitness - 12.5 % chance of playing  '
    r=extract_personnel_news(news,'a')
    assert r['raw_news']==news
    assert r['source_percentage_mentions']==[{'source_span':'12.5 %','number_as_written':'12.5'}]
    assert r['normalized_return_date'] is None
    assert r['actual_target_absence_confirmed'] is False
    assert r['estimated_strength_loss'] is None


@pytest.mark.parametrize('news', ['Neck','Pelvic'])
def test_bare_anatomical_source_labels_do_not_establish_a_diagnosis(news):
    r=extract_personnel_news(news,None)
    assert r['raw_news']==news
    assert r['clinical_state_confirmed'] is False
    assert r['actual_target_absence_confirmed'] is False
    assert r['normalized_return_date'] is None


@pytest.mark.parametrize('news,status', [(123,'a'),('Knock',42),([],None)])
def test_non_text_inputs_are_rejected(news,status):
    with pytest.raises(ValueError):
        extract_personnel_news(news,status)
