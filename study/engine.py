"""Primary fitting, frozen-origin prediction and registered sensitivity runs."""
from __future__ import annotations

import copy
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .common import write_json,frame_records
from .data import build_dataset,FEATURES
from .models import fit_pipeline,tune_family
from .evaluation import metrics_table,paired_uncertainty

MODEL_IDS={('nb','B'):'P03',('nb','C'):'P04',('nb','E'):'P05',
           ('boosting','B'):'P06',('boosting','C'):'P07',('boosting','E'):'P08'}


def progress(message):print(message,flush=True)


def frozen_forecasts(models,test,include_benchmarks=True):
    records=[]
    for origin,week in test.groupby('week_start',sort=True):
        week=week.sort_values('unit')
        for key,model in models.items():
            values=model.predict(week)
            for row,value in zip(week.itertuples(),values):
                records.append({'model_id':MODEL_IDS[key],'unit':row.unit,'week_start':origin,
                                'target':row.target,'prediction':float(value)})
        if include_benchmarks:
            for model,column in [('P01','tb_count_lag1'),('P02','tb_count_prev4w_mean')]:
                for row in week.itertuples():
                    records.append({'model_id':model,'unit':row.unit,'week_start':origin,
                                    'target':row.target,'prediction':float(getattr(row,column))})
    out=pd.DataFrame(records)
    assert not out.duplicated(['model_id','unit','week_start']).any()
    assert out.groupby('model_id').size().eq(len(test)).all()
    assert np.isfinite(out.prediction).all() and out.prediction.ge(0).all()
    return out


def fit_final(data,units,settings,config,families=('nb','boosting'),sets=('B','C','E'),omit_policy=False):
    train=data.loc[data.partition.eq('train')]
    models={}
    for family in families:
        for s in sets:
            model=fit_pipeline(train,family,s,settings[family],config,units,omit_policy)
            models[(family,s)]=model
            progress(f'Final 2020 fit: {family} {s}, rows={len(train)}')
    return models


def save_evaluation(directory,predictions,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'predictions.json',frame_records(predictions))
    metrics=metrics_table(predictions)
    intervals=paired_uncertainty(predictions,config)
    write_json(directory/'metrics.json',metrics)
    write_json(directory/'paired_uncertainty.json',intervals)
    return metrics,intervals


def run_sensitivities(store,config,primary_settings,primary_models,out):
    out=Path(out);registry=[]
    specifications=[
        ('S01',{'geography':'city'},('nb','boosting'),('B','C','E'),True),
        ('S02',{'source':'mixed'},('boosting',),('B','C'),False),
        ('S03',{'source':'nasa'},('boosting',),('B','C'),False),
        ('S04',{'window':14},('boosting',),('B','C'),False),
        ('S05',{'window':56},('boosting',),('B','C'),False),
        ('S06',{'revised':True},('nb','boosting'),('B','C','E'),True),
        ('S07',{'delay':1},('boosting',),('B','C'),False),
        ('S08',{'omit_policy':True},('boosting',),('B','C'),False),
        ('S10',{'source':'footprint'},('boosting',),('B','C','E'),False)]
    for run_id,changes,families,sets,retune in specifications:
        progress(f'{run_id}: preparing registered sensitivity {changes}')
        directory=out/run_id
        data,units=build_dataset(store,**changes)
        test=data.loc[data.partition.eq('test')]
        if changes.get('geography')=='city':assert len(test)==102
        else:assert len(test)==255
        settings=copy.deepcopy(primary_settings)
        if retune:
            for family in families:
                settings[family],log=tune_family(data.loc[data.partition.eq('train')],family,config,units,progress=progress)
                write_json(directory/f'tuning_{family}.json',log)
        models=fit_final(data,units,settings,config,families,sets,changes.get('omit_policy',False))
        predictions=frozen_forecasts(models,test,include_benchmarks=retune)
        save_evaluation(directory,predictions,config)
        write_json(directory/'fit_diagnostics.json',{MODEL_IDS[k]:v.diagnostics for k,v in models.items()})
        write_json(directory/'settings.json',settings)
        rec={'id':run_id,'status':'completed','changes':changes,'retuned_in_2020':retune,
             'training_rows':int(data.partition.eq('train').sum()),'test_rows':len(test),
             'forecast_series':len(predictions.model_id.unique()),'final_fits':len(models),
             'hyperparameter_validation_fits':216 if retune else 0}
        registry.append(rec)
        write_json(out/'registry.json',registry)

    progress('S09: fixed-setting quarterly expanding refits')
    data,units=build_dataset(store)
    models={key:copy.deepcopy(value) for key,value in primary_models.items() if key[0]=='boosting'}
    frames=[];logs=[]
    origins=pd.to_datetime(config['refit_sensitivity_origins'])
    for t,week in data.loc[data.partition.eq('test')].groupby('week_start',sort=True):
        if t in origins:
            past=data.loc[data.partition.ne('bridge') & data.week_end.lt(t)]
            for key in models:
                models[key]=fit_pipeline(past,'boosting',key[1],primary_settings['boosting'],config,units)
                logs.append({'origin':t.strftime('%Y-%m-%d'),'model':MODEL_IDS[key],'fitting_rows':len(past),'last_fitting_week':past.week_start.max().strftime('%Y-%m-%d')})
        frames.append(frozen_forecasts(models,week,include_benchmarks=False))
    save_evaluation(out/'S09',pd.concat(frames,ignore_index=True),config)
    write_json(out/'S09/refits.json',logs)
    registry.append({'id':'S09','status':'completed','changes':{'quarterly_refit':True},'retuned_in_2021':False,
                     'test_rows':255,'forecast_series':3,'additional_fits':9})
    registry.sort(key=lambda x:x['id'])
    write_json(out/'registry.json',registry)
    return registry


def location_quality_audit(data_root,out):
    root=Path(data_root)
    old=pd.read_csv(root/'climate/district_grid_2019_2021/patient_location_weather_linkage.csv',
                    usecols=['record_index','district','notification_date','era5_land_grid_id','era5_grid_id'])
    old['notification_date']=pd.to_datetime(old.notification_date)
    assert len(old)==3699 and old.record_index.is_unique
    v2=pd.read_csv(root/'tb_cleaned_google_rechecked_v2.csv',usecols=[
        'source_record_index','latitude','longitude','AG_AH_verification_status','coordinate_precision'])
    v2['record_index']=pd.to_numeric(v2.source_record_index,errors='coerce')
    v2=v2.loc[v2.record_index.notna()].drop(columns='source_record_index')
    assert v2.record_index.is_unique
    joined=old.merge(v2,on='record_index',how='left',validate='one_to_one')
    accepted=joined.latitude.notna() & joined.longitude.notna()
    assert len(joined)==3699 and accepted.sum()==124
    stats={'full_in_period_cohort':3699,'accepted_in_period':124,
           'accepted_fraction':124/3699,'unlinked_or_without_accepted_coordinates':int((~accepted).sum()),
           'coverage_by_district':[], 'coordinate_values_exported':False,
           'interpretation':'outcome-selected property-location subset; no population location validation'}
    for u,g in joined.assign(accepted=accepted).groupby('district'):
        n=int(g.accepted.sum())
        stats['coverage_by_district'].append({'district':u,'denominator':len(g),'accepted':n if n>=5 else 'suppressed_below_5'})
    a=joined.loc[accepted].copy()
    concordance={}
    for source,resolution,oldcol in [('era5_land',.1,'era5_land_grid_id'),('era5',.25,'era5_grid_id')]:
        lat=np.round(np.round(a.latitude.to_numpy()/resolution)*resolution,2)
        lon=np.round(np.round(a.longitude.to_numpy()/resolution)*resolution,2)
        new=np.array([f'{source}_{la:.2f}_{lo:.2f}' for la,lo in zip(lat,lon)])
        present=a[oldcol].notna().to_numpy()
        concordance[source]={'accepted_points_with_old_cell':int(present.sum()),
                             'same_rounded_cell':int(np.sum(new[present]==a.loc[present,oldcol].to_numpy())),
                             'note':'rounding concordance only; not a validation of residence or full-cohort representativeness'}
    stats['grid_concordance']=concordance
    write_json(out,stats)
    return stats


def audit_fixed_footprints(data_root):
    root=Path(data_root)/'climate/district_grid_2019_2021'
    weights=pd.read_csv(root/'district_proxy_grid_weights.csv')
    old=pd.read_csv(root/'patient_location_weather_linkage.csv',
                    usecols=['district','notification_date','era5_land_grid_id','era5_grid_id'])
    old['notification_date']=pd.to_datetime(old.notification_date)
    training=old.loc[old.notification_date.dt.year.eq(2020)]
    assert weights.reference_year.eq(2020).all()
    for (unit,source),g in weights.groupby(['district','source']):
        col='era5_land_grid_id' if source=='era5_land' else 'era5_grid_id'
        expected=set(training.loc[training.district.eq(unit),col].dropna())
        assert set(g.grid_id)==expected and np.allclose(g.weight,1/len(expected))
    return {'reference_year':2020,'unique_2020_cell_membership_reconstructed':True,
            'no_2021_locations_in_weight_definition':True,
            'primary_tuning_uses_these_weights':False,
            'limitation':'old unverified points; downloaded grid domain informed by available cohort; exploratory only'}
