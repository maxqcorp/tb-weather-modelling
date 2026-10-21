"""Synthetic numerical and leakage checks, performed before study fitting."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
import statsmodels.api as sm
from copy import deepcopy
from unittest.mock import patch
from .data import FEATURES,WEATHER,DISTRICTS,origin_features
from .models import fit_pipeline,nb_functions,tune_family
from .evaluation import block_indices,paired_uncertainty,error_metrics
from .interpretation import tree_contributions


def synthetic_frame(seed=103):
    rng=np.random.default_rng(seed);n=215
    dates=pd.date_range('2020-03-02',periods=43,freq='W-MON')
    f=pd.DataFrame({'unit':DISTRICTS*43,'week_start':np.repeat(dates,5),'partition':'train'})
    f['week_end']=f.week_start+pd.Timedelta(days=6)
    for c in FEATURES['E']:f[c]=rng.normal(size=n)
    f['target']=rng.poisson(np.exp(1+.15*f[FEATURES['B'][0]].to_numpy()+.08*np.tile(np.arange(5),43)))
    return f


def run_checks(config):
    results=[]
    f=synthetic_frame();setting={'ridge_lambda':1,'dispersion':.5}
    nb=fit_pipeline(f,'nb','E',setting,config,DISTRICTS)
    x=nb.transform(f);y=f.target.to_numpy()
    penalty=np.r_[0,np.ones(15),np.zeros(4)]
    fun,jac,hess,glm=nb_functions(x,y,.5,penalty)
    beta=np.linspace(-.05,.05,x.shape[1])
    gradient_error=float(check_grad(fun,jac,beta))
    eps=1e-5
    numerical=np.column_stack([(jac(beta+np.eye(len(beta))[i]*eps)-jac(beta-np.eye(len(beta))[i]*eps))/(2*eps) for i in range(len(beta))])
    hessian_error=float(np.max(np.abs(numerical-hess(beta))))
    assert gradient_error<1e-5 and hessian_error<1e-4
    reference=glm.fit_regularized(alpha=penalty,L1_wt=0,refit=False,opt_method='bfgs')
    objective_difference=float(fun(nb.model)-fun(reference.params))
    assert objective_difference<1e-7
    results.append({'check':'NB2_objective_gradient_hessian_and_reference_fit','passed':True,
                    'gradient_error':gradient_error,'hessian_error':hessian_error,
                    'objective_minus_statsmodels_reference':objective_difference,'solver':nb.diagnostics})

    altered=f.copy();altered[FEATURES['E']]=1e9
    means=nb.means.copy();nb.transform(altered)
    assert np.array_equal(means,nb.means)
    constant=f.copy();constant['annual_sin']=0
    constant_model=fit_pipeline(constant,'nb','B',setting,config,DISTRICTS)
    assert constant_model.scales[FEATURES['B'].index('annual_sin')]==1
    results.append({'check':'fitting_partition_scaling_and_zero_variance','passed':True})

    boost=fit_pipeline(f,'boosting','E',{'max_depth':1,'n_estimators':50,'reg_lambda':1},config,DISTRICTS)
    phi,base,meta=tree_contributions(boost,f,f.iloc[:10])
    assert np.isfinite(phi).all() and boost.predict(f).min()>=0
    results.append({'check':'Poisson_boosting_and_SHAP_raw_margin','passed':True,**meta})

    weeks=pd.date_range('2020-01-06',periods=65,freq='W-MON')
    panel=pd.DataFrame({'unit':'unit','week_start':weeks,'week_end':weeks+pd.Timedelta(days=6),
                        'target':np.arange(len(weeks))%7,'full_study_week':True})
    days=pd.date_range('2019-01-01','2021-12-31',freq='D')
    weather=pd.DataFrame({c:np.arange(len(days))*.01+i for i,c in enumerate(WEATHER)},index=days)
    policy=pd.DataFrame({'date':days,'covid_stringency_index':np.arange(len(days))%100})
    t=pd.Timestamp('2021-01-04')
    a=origin_features(panel,weather,policy,'unit',t)
    p2=panel.copy();p2.loc[p2.week_start.ge(t),'target']=99999
    w2=weather.copy();w2.loc[w2.index>=t,:]=99999
    pol2=policy.copy();pol2.loc[pol2.date.ge(t),'covid_stringency_index']=99999
    assert a==origin_features(p2,w2,pol2,'unit',t)
    expected=panel.loc[panel.week_start.eq('2020-12-28'),'target'].iloc[0]
    assert a['tb_count_lag1']==expected
    changed_past=panel.copy();changed_past.loc[changed_past.week_start.eq('2020-12-28'),'target']=888
    assert origin_features(changed_past,weather,policy,'unit',t)['tb_count_lag1']==888
    changed=panel.copy();changed.loc[changed.week_start.eq('2020-12-28'),'full_study_week']=False
    assert origin_features(changed,weather,policy,'unit',t) is None
    delayed=origin_features(panel,weather,policy,'unit',t,delay=1)
    expected=panel.loc[panel.week_start.eq('2020-12-21'),'target'].iloc[0]
    assert delayed['tb_count_lag1']==expected
    results.append({'check':'future_count_weather_policy_invariance_bridge_and_partial_history','passed':True})

    ids=block_indices(12,4,100,56)
    assert ids.shape==(100,12) and ((ids>=0)&(ids<12)).all()
    assert (np.diff(ids[:,:4],axis=1)==1).all()
    # Exact oracle: a constant one-unit reduction in absolute error yields gain 1.
    rows=[]
    for w in pd.date_range('2021-01-04',periods=12,freq='W-MON'):
        for u in ['a','b']:
            for m,p in [('P06',3),('P07',2)]:rows.append({'week_start':w,'unit':u,'model_id':m,'target':1,'prediction':p})
    records=paired_uncertainty(pd.DataFrame(rows),config,[('oracle','P06','P07')])
    assert all(r['MAE_gain']==1 and r['MAE_gain_CI95']==[1,1] for r in records)
    assert np.isclose(error_metrics([0,0],[1,3])['RMSE'],np.sqrt(5))
    results.append({'check':'paired_blocks_duplicate_positions_and_metric_oracle','passed':True})
    mocked=deepcopy(config)
    mocked['xgboost']['grid']={'max_depth':[1],'n_estimators':[50],'reg_lambda':[1,10]}
    class FakeFit:
        diagnostics={'synthetic_mock':True}
        def predict(self,frame):return np.ones(len(frame))
    def fake_fit(rows,family,s,setting,cfg,units,omit_policy):
        if setting['reg_lambda']==1 and s=='E' and rows.week_start.max()==pd.Timestamp('2020-10-26'):
            raise ArithmeticError('Intentional synthetic failure in one set/fold')
        return FakeFit()
    with patch('study.models.fit_pipeline',side_effect=fake_fit) as fake:
        chosen,log=tune_family(f,'boosting',mocked,DISTRICTS)
        assert chosen['reg_lambda']==10 and fake.call_count==18
        assert not log['configurations'][0]['valid'] and len(log['configurations'][0]['cells'])==9
        invalid=f.assign(partition='test')
        try:tune_family(invalid,'boosting',mocked,DISTRICTS)
        except AssertionError:pass
        else:raise AssertionError('Tuning accepted held-out test partition')
    results.append({'check':'failed_configuration_excluded_whole_and_test_partition_rejected','passed':True,'mock_fit_calls':18})
    return {'all_passed':True,'synthetic_only':True,'study_data_used':False,'checks':results,
            'SHAP_mode':meta['mode'],'synthetic_fits_executed':4}


if __name__=='__main__':
    config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    print(json.dumps(run_checks(config),indent=2))
