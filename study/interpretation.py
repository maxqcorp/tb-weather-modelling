"""Tree contributions with explicit background and output-scale checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata,spearmanr
from .data import CORE,WEATHER
from .evaluation import block_indices


def tree_contributions(pipeline,background,frame,force_mode=None):
    if force_mode not in (None,'interventional'):
        raise ValueError('This study requires interventional Tree SHAP')
    xtrain=pipeline.transform(background);x=pipeline.transform(frame)
    model=pipeline.model
    raw=model.predict(x,output_margin=True)
    try:
        masker=shap.maskers.Independent(xtrain,max_samples=len(xtrain))
        explainer=shap.TreeExplainer(model,data=masker,feature_perturbation='interventional',model_output='raw')
        assert len(explainer.data)==len(background),'SHAP reduced the required background rows'
        phi=explainer.shap_values(x,check_additivity=True)
        base=np.full(len(x),float(explainer.expected_value))
        if not np.allclose(base+np.sum(phi,axis=1),raw,atol=1e-5,rtol=1e-5):
            raise ArithmeticError('Interventional additivity failed')
    except (ValueError,AssertionError,ArithmeticError,TypeError,NotImplementedError) as exc:
        raise RuntimeError('Interventional Tree SHAP is required; the explanation stage stopped without using another method') from exc
    error=float(np.max(np.abs(base+np.sum(phi,axis=1)-raw)))
    return np.asarray(phi),np.asarray(base),{'mode':'interventional','output_scale':'log_expected_count_margin',
        'background_rows':len(background),'max_additivity_error':error,'fallback':None}



def explain(pipeline,training,test,config,mode):
    phi,base,meta=tree_contributions(pipeline,training,test,force_mode=mode)
    names=pipeline.names
    weather_names=[f'era5_{x}_prev4w' for x in (CORE if pipeline.feature_set=='C' else WEATHER)]
    indices=[names.index(n) for n in weather_names]
    weeks=sorted(test.week_start.unique());units=sorted(test.unit.unique())
    key=pd.MultiIndex.from_frame(test[['week_start','unit']])
    expected=pd.MultiIndex.from_product([weeks,units],names=['week_start','unit'])
    order=key.get_indexer(expected);assert (order>=0).all()
    arr=phi[order].reshape(len(weeks),len(units),len(names))
    groups=[('all','all',np.ones(len(weeks),bool),None)]
    groups += [('district',u,np.ones(len(weeks),bool),j) for j,u in enumerate(units)]
    dates=pd.DatetimeIndex(weeks)
    groups += [('quarter',f'Q{q}',dates.quarter==q,None) for q in range(1,5)]
    groups += [('half',h,(dates.month<=6) if h=='H1' else (dates.month>6),None) for h in ['H1','H2']]
    groups += [('district_quarter',f'{u} | Q{q}',dates.quarter==q,j) for j,u in enumerate(units) for q in range(1,5)]
    summaries=[];ranks={};draws=config['uncertainty']['draws']
    for scope,label,wmask,unit_index in groups:
        a=arr[wmask]
        if unit_index is not None:a=a[:,unit_index:unit_index+1]
        importance=np.abs(a).mean(axis=(0,1));wrank=rankdata(-importance[indices],method='average')
        ranks[(scope,label)]=wrank
        length=2 if scope in ['quarter','district_quarter'] else 4
        ids=block_indices(len(a),length,draws,config['uncertainty']['seed'])
        sampled=np.abs(a[ids]).mean(axis=(1,2))[:,indices]
        rr=rankdata(-sampled,axis=1,method='average')
        top2=np.mean(rr<=2,axis=0)
        for j,n in enumerate(names):
            rec={'scope':scope,'group':label,'feature':n,'mean_absolute_SHAP':float(importance[j]),
                 'mean_signed_SHAP':float(a[:,:,j].mean()),'weeks':int(len(a)),'rows':int(a.shape[0]*a.shape[1])}
            if n in weather_names:
                k=weather_names.index(n);denom=float(importance[indices].sum())
                rec.update({'weather_rank':float(wrank[k]),'weather_share':float(importance[j]/denom) if denom else None,
                    'top_two_block_frequency':float(top2[k]),'rank_CI95':np.quantile(rr[:,k],[.025,.975]).tolist()})
            summaries.append(rec)
    comparisons=[]
    for scope in ['district','quarter','half']:
        keys=[k for k in ranks if k[0]==scope]
        for i,k in enumerate(keys):
            for l in keys[i+1:]:
                a,b=ranks[k],ranks[l]
                rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
                topa=set(np.array(weather_names)[a<=2]);topb=set(np.array(weather_names)[b<=2])
                comparisons.append({'scope':scope,'a':k[1],'b':l[1],'spearman':rho,'top_two_overlap':len(topa&topb)})
    stability=[]
    for n in weather_names:
        joint=[r for r in summaries if r['scope']=='district_quarter' and r['feature']==n and r.get('weather_share') is not None]
        fraction=float(np.mean([r['weather_rank']<=2 for r in joint])) if joint else None
        overall=next(r for r in summaries if r['scope']=='all' and r['feature']==n)
        stability.append({'feature':n,'eligible_district_quarter_groups':len(joint),
            'top_two_group_fraction':fraction,'full_year_top_two_block_frequency':overall['top_two_block_frequency'],
            'descriptive_75_percent_convention_met':bool(fraction is not None and fraction>=.75 and overall['top_two_block_frequency']>=.75)})
    return {'metadata':meta,'feature_names':names,'summaries':summaries,'ranking_comparisons':comparisons,'descriptive_stability':stability},phi,base
