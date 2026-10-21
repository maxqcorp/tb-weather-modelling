"""Paired errors and time-block uncertainty; no independent row resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPARISONS=[('RQ1_boosting_vs_NB_core','P04','P07'),
             ('RQ1_boosting_vs_persistence','P01','P07'),
             ('RQ2_core_vs_baseline','P06','P07'),
             ('RQ2_expanded_vs_core','P07','P08'),
             ('RQ2_expanded_vs_baseline','P06','P08'),
             ('NB_core_vs_baseline','P03','P04'),
             ('NB_expanded_vs_core','P04','P05')]


def error_metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),
            'mean_signed_error':float(np.mean(e)),'rows':int(e.size)}


def metrics_table(predictions):
    rows=[]
    for model,g in predictions.groupby('model_id',sort=True):
        g=g.copy();g['quarter']='Q'+g.week_start.dt.quarter.astype(str)
        g['half']=np.where(g.week_start.dt.month<=6,'H1','H2')
        groups=[('pooled','all',g)]
        groups += [('unit',str(u),f) for u,f in g.groupby('unit')]
        groups += [('quarter',str(u),f) for u,f in g.groupby('quarter')]
        groups += [('half',str(u),f) for u,f in g.groupby('half')]
        p=g.loc[g.unit.isin(['WP PUTRAJAYA','Putrajaya'])]
        for label,mask in [('Putrajaya_zero',p.target.eq(0)),('Putrajaya_nonzero',p.target.ne(0))]:
            if mask.any():groups.append(('sparsity',label,p.loc[mask]))
        for scope,label,f in groups:
            rows.append({'model_id':model,'scope':scope,'group':label,'weeks':int(f.week_start.nunique()),**error_metrics(f.target,f.prediction)})
        unitstats=[error_metrics(f.target,f.prediction) for _,f in g.groupby('unit')]
        rows.append({'model_id':model,'scope':'macro','group':'equal_units','weeks':int(g.week_start.nunique()),
            'rows':len(g),**{m:float(np.mean([x[m] for x in unitstats])) for m in ['MAE','RMSE','mean_signed_error']}})
    return rows


def block_indices(n,length,draws,seed):
    if not 1<=length<=n:raise ValueError('Invalid block length')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-length+1,size=(draws,int(np.ceil(n/length))))
    return (starts[:,:,None]+np.arange(length)[None,None,:]).reshape(draws,-1)[:,:n]


def cube_from_predictions(predictions,models=None):
    models=sorted(predictions.model_id.unique()) if models is None else models
    units=sorted(predictions.unit.unique());weeks=sorted(predictions.week_start.unique())
    index=pd.MultiIndex.from_product([weeks,units],names=['week_start','unit'])
    arrays=[];truth=None
    for model in models:
        g=predictions.loc[predictions.model_id.eq(model)].set_index(['week_start','unit'])
        if g.index.has_duplicates or len(g)!=len(index):raise ValueError('Unpaired model keys')
        g=g.reindex(index)
        if g[['target','prediction']].isna().any().any():raise ValueError('Missing paired forecasts')
        y=g.target.to_numpy().reshape(len(weeks),len(units))
        if truth is None:truth=y
        else:assert np.array_equal(y,truth)
        arrays.append(g.prediction.to_numpy().reshape(len(weeks),len(units)))
    return truth,np.stack(arrays,axis=2),weeks,units,models


def paired_uncertainty(predictions,config,comparisons=None):
    if comparisons is None:comparisons=COMPARISONS
    available=set(predictions.model_id)
    comparisons=[c for c in comparisons if c[1] in available and c[2] in available]
    needed=sorted({m for _,a,b in comparisons for m in [a,b]})
    if not needed:return []
    y,p,weeks,units,models=cube_from_predictions(predictions,needed)
    ae=np.abs(p-y[:,:,None]);se=(p-y[:,:,None])**2
    results=[]
    for length in [config['uncertainty']['block_weeks'],config['uncertainty']['sensitivity_block_weeks']]:
        ids=block_indices(len(weeks),length,config['uncertainty']['draws'],config['uncertainty']['seed'])
        mae=ae[ids].mean(axis=(1,2));rmse=np.sqrt(se[ids].mean(axis=(1,2)))
        for label,comparator,new in comparisons:
            a,b=models.index(comparator),models.index(new)
            absdiff=mae[:,a]-mae[:,b]
            pct=np.divide(100*absdiff,mae[:,a],out=np.full(len(ids),np.nan),where=mae[:,a]>0)
            rgain=rmse[:,a]-rmse[:,b]
            ca,cb=float(ae[:,:,a].mean()),float(ae[:,:,b].mean())
            results.append({'contrast':label,'comparator':comparator,'new_model':new,
                'sign':'positive_gain_favours_new_model','block_weeks':length,'draws':len(ids),
                'weeks':len(weeks),'units':len(units),'MAE_comparator':ca,'MAE_new':cb,
                'MAE_gain':ca-cb,'MAE_gain_CI95':np.quantile(absdiff,[.025,.975]).tolist(),
                'percentage_MAE_gain':100*(ca-cb)/ca if ca else None,
                'percentage_gain_CI95':np.nanquantile(pct,[.025,.975]).tolist() if np.isfinite(pct).any() else [None,None],
                'undefined_percentage_draws':int(np.isnan(pct).sum()),
                'RMSE_gain':float(np.sqrt(se[:,:,a].mean())-np.sqrt(se[:,:,b].mean())),
                'RMSE_gain_CI95':np.quantile(rgain,[.025,.975]).tolist()})
    return results
