"""Shared-setting count models, with explicit numerical diagnostics."""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from xgboost import XGBRegressor
from .data import FEATURES


@dataclass
class Pipeline:
    family:str
    feature_set:str
    numeric:list
    units:list
    means:np.ndarray
    scales:np.ndarray
    model:object
    params:dict
    diagnostics:dict

    @property
    def names(self):return self.numeric+[f'unit_{u}' for u in self.units[1:]]

    def transform(self,frame):
        x=frame[self.numeric].to_numpy(dtype=float)
        if self.family=='nb':x=(x-self.means)/self.scales
        identity=np.column_stack([frame.unit.eq(u).to_numpy(dtype=float) for u in self.units[1:]])
        x=np.column_stack([x,identity])
        if self.family=='nb':x=np.column_stack([np.ones(len(x)),x])
        return x

    def predict(self,frame):
        x=self.transform(frame)
        if self.family=='nb':p=np.exp(x@self.model)
        else:p=self.model.predict(x)
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Nonfinite or negative forecast')
        return np.asarray(p,dtype=float)


def nb_functions(x,y,dispersion,penalty):
    glm=sm.GLM(y,x,family=sm.families.NegativeBinomial(alpha=dispersion,link=sm.families.links.Log()))
    n=len(y)
    def fun(beta):return -glm.loglike(beta)/n+np.sum(penalty*beta**2)/2
    def jac(beta):return -glm.score(beta)/n+penalty*beta
    def hess(beta):return -glm.hessian(beta,observed=True)/n+np.diag(penalty)
    return fun,jac,hess,glm


def fit_pipeline(frame,family,feature_set,setting,config,units,omit_policy=False):
    numeric=list(FEATURES[feature_set])
    if omit_policy:numeric.remove('covid_stringency_prevweek')
    y=frame.target.to_numpy(dtype=float)
    assert len(y)>0 and np.isfinite(y).all() and (y>=0).all() and np.equal(y,np.floor(y)).all()
    assert set(frame.unit)==set(units)
    means=frame[numeric].mean().to_numpy()
    scales=frame[numeric].std(ddof=0).to_numpy()
    scales[scales==0]=1
    pipeline=Pipeline(family,feature_set,numeric,list(units),means,scales,None,dict(setting),{})
    x=pipeline.transform(frame)
    if not np.isfinite(x).all():raise ValueError('Missing or nonfinite predictors')
    if family=='nb':
        penalty=np.r_[0,np.repeat(setting['ridge_lambda'],len(numeric)),np.zeros(len(units)-1)]
        fun,jac,hess,glm=nb_functions(x,y,setting['dispersion'],penalty)
        start=np.zeros(x.shape[1])
        global_mean=y.mean()+0.1
        start[0]=np.log(global_mean)
        for i,u in enumerate(units[1:]):start[1+len(numeric)+i]=np.log(frame.loc[frame.unit.eq(u),'target'].mean()+0.1)-start[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=minimize(fun,start,jac=jac,hess=hess,method='trust-exact',
                            options={'gtol':config['negative_binomial']['tolerance'],
                                     'maxiter':config['negative_binomial']['maxiter']})
        grad=float(np.max(np.abs(jac(result.x))))
        converged=bool(np.isfinite(result.x).all() and np.isfinite(result.fun) and grad<=config['negative_binomial']['tolerance'])
        pipeline.diagnostics={'solver':'scipy_trust_exact_statsmodels_NB2_likelihood',
            'success':bool(result.success),'gradient_infinity_norm':grad,'iterations':int(result.nit),
            'objective':float(result.fun),'converged':converged,'message':str(result.message),
            'warnings':[str(w.message) for w in caught]}
        if not converged:raise ArithmeticError(f'NB solver did not meet tolerance: grad={grad:.3g}')
        pipeline.model=result.x
    elif family=='boosting':
        fixed=config['xgboost']
        pipeline.model=XGBRegressor(objective='count:poisson',learning_rate=fixed['learning_rate'],
            min_child_weight=fixed['min_child_weight'],reg_alpha=fixed['reg_alpha'],gamma=fixed['gamma'],
            max_delta_step=fixed['max_delta_step'],subsample=fixed['subsample'],
            colsample_bytree=fixed['colsample_bytree'],device='cpu',tree_method='hist',n_jobs=1,
            random_state=fixed['seed'],verbosity=0,**setting)
        pipeline.model.fit(x,y)
        pipeline.diagnostics={'converged':True,'objective':'count:poisson','trees':setting['n_estimators']}
    else:raise ValueError('Unknown model family')
    pipeline.predict(frame)
    return pipeline


def macro_mae(frame,prediction):
    return float(np.mean([np.abs(frame.loc[frame.unit.eq(u),'target'].to_numpy()-prediction[frame.unit.eq(u).to_numpy()]).mean() for u in sorted(frame.unit.unique())]))


def tune_family(training,family,config,units,feature_sets=('B','C','E'),omit_policy=False,progress=None):
    assert (training.partition=='train').all() and training.week_end.max().year==2020
    spec=config['negative_binomial' if family=='nb' else 'xgboost']
    names=list(spec['grid'])
    candidates=[dict(zip(names,vals)) for vals in itertools.product(*spec['grid'].values())]
    logs=[]
    for candidate_id,setting in enumerate(candidates):
        cells=[];valid=True
        for s in feature_sets:
            for i,fold in enumerate(config['folds'],1):
                fit=training.loc[training.week_start.between(fold['fit_start'],fold['fit_end'])]
                validation=training.loc[training.week_start.between(fold['validation_start'],fold['validation_end'])]
                assert fit.week_end.max()<validation.week_start.min()
                try:
                    model=fit_pipeline(fit,family,s,setting,config,units,omit_policy)
                    score=macro_mae(validation,model.predict(validation))
                    cells.append({'set':s,'fold':i,'macro_MAE':score,'fit_rows':len(fit),'validation_rows':len(validation),'diagnostics':model.diagnostics})
                except (ArithmeticError,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    valid=False
                    cells.append({'set':s,'fold':i,'error':str(exc)})
        log={'candidate':candidate_id,'setting':setting,'valid':valid,'cells':cells,
             'mean_macro_MAE':float(np.mean([c['macro_MAE'] for c in cells])) if valid else None}
        logs.append(log)
        if progress:progress(f'{family}: configuration {candidate_id+1}/{len(candidates)}; valid={valid}')
    good=[c for c in logs if c['valid']]
    if not good:raise ArithmeticError('No common valid configuration; stop before test scoring')
    best=min(c['mean_macro_MAE'] for c in good)
    near=[c for c in good if c['mean_macro_MAE']<=best+spec['tie_margin_MAE']]
    if family=='boosting':key=lambda c:(c['setting']['max_depth'],c['setting']['n_estimators'],-c['setting']['reg_lambda'])
    else:key=lambda c:(-c['setting']['ridge_lambda'],abs(c['setting']['dispersion']-.5))
    chosen=min(near,key=key)
    return chosen['setting'],{'family':family,'selection':'equal_set_equal_fold_macro_MAE_then_prespecified_tie_rule',
                              'minimum_score':best,'chosen_candidate':chosen['candidate'],'chosen_score':chosen['mean_macro_MAE'],
                              'chosen_setting':chosen['setting'],'configurations':logs}
