from datetime import date, timedelta

import numpy as np
import pytest

from football_model.gap_outcome import FEATURES, GAPOutcomeModel


def sample():
    rng=np.random.default_rng(184)
    x=rng.normal(size=(180,4))
    latent=x[:,0]+.4*x[:,1]+rng.logistic(size=len(x))
    y=np.where(latent<-.6,2,np.where(latent<.6,1,0))
    start=date(2023,8,1)
    return [{'match_id':f'm{i}', 'date':(start+timedelta(days=i)).isoformat(),
             'forecast_before_date':(start+timedelta(days=i-3)).isoformat(),
             'feature_before_date':(start+timedelta(days=i-3)).isoformat(),
             'gap_differences':dict(zip(FEATURES,values)), 'actual_outcome':int(label)}
            for i,(values,label) in enumerate(zip(x,y))]


def test_native_order_export_and_future_strength_direction():
    model=GAPOutcomeModel().fit(sample(),date(2024,3,1))
    base={'date':'2024-04-10','forecast_before_date':'2024-04-07','feature_before_date':'2024-04-07'}
    rows=[{**base,'gap_differences':dict(zip(FEATURES,[v,0,0,0]))} for v in (-2,2)]
    p=model.predict_many(rows)
    assert np.allclose(p.sum(axis=1),1)
    assert p[1,0]>p[0,0] and p[1,2]<p[0,2]
    native=model.result.model.predict(model.result.params,exog=model.scaler.transform(
        [[r['gap_differences'][k] for k in FEATURES] for r in rows]))
    assert np.allclose(p,native[:,::-1])
    assert model.parameters()['latent_thresholds'][0]<model.parameters()['latent_thresholds'][1]


def test_cutoff_label_and_snapshot_rejection():
    rows=sample()
    with pytest.raises(ValueError,match='precede'):
        GAPOutcomeModel().fit(rows,date(2023,9,1))
    rows[0]['feature_before_date']='2023-08-01'
    with pytest.raises(ValueError,match='snapshot'):
        GAPOutcomeModel().fit(rows,date(2024,3,1))
