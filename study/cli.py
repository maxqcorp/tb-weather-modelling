"""Separate development, locked evaluation and sensitivity execution stages."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .common import write_json,sha256,frame_records
from .checks import run_checks
from .data import DataStore,build_dataset,assert_primary_matches_prepared,FEATURES
from .models import tune_family
from .engine import fit_final,frozen_forecasts,save_evaluation,progress,MODEL_IDS,run_sensitivities,location_quality_audit,audit_fixed_footprints
from .interpretation import tree_contributions,explain
from .reporting import report


def code_fingerprints():
    here=Path(__file__).resolve().parent
    files=list(here.glob('*.py'))+[here.parent/'run_study.py']
    return {str(p.relative_to(here.parent)):sha256(p) for p in sorted(files)}


def verify_lock(out):
    frozen=json.loads((out/'implementation_manifest.json').read_text())
    if frozen['code_sha256']!=code_fingerprints():raise RuntimeError('Implementation changed after development freeze')
    for p,h in frozen['input_sha256'].items():
        if sha256(p)!=h:raise RuntimeError('Input changed after development freeze')
    lock=json.loads((out/'primary_lock.json').read_text())
    for name,h in lock['artefact_sha256'].items():
        if sha256(out/name)!=h:raise RuntimeError('Frozen primary artefact changed')
    return frozen,lock


def status(out,state,**kwargs):
    write_json(out/'run_status.json',{'state':state,**kwargs})


def main():
    parser=argparse.ArgumentParser(description='Run the prespecified retrospective study using local aggregate data.')
    parser.add_argument('stage',choices=['check','train','evaluate','sensitivities','report','all'])
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config.json')
    parser.add_argument('--source-audit',type=Path)
    parser.add_argument('--prepared-primary',type=Path)
    parser.add_argument('--date',default=datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).strftime('%Y-%m-%d'))
    parser.add_argument('--execute',action='store_true',help='Explicitly authorise study fitting and/or scoring.')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage not in ['check','report'] and not args.execute:parser.error('--execute is required for study execution')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    store=DataStore(args.data_root)
    data,units=build_dataset(store)
    if args.prepared_primary:assert_primary_matches_prepared(data,args.prepared_primary)
    assert len(data)==475 and data.loc[data.partition.eq('test'),'target'].sum()==1725
    if args.source_audit:
        audit=json.loads(args.source_audit.read_text())
        for name,h in audit['source_sha256'].items():
            if sha256(args.data_root.parent/name)!=h:raise RuntimeError(f'Source version changed: {name}')

    if args.stage in ['check','train','all']:
        if (out/'primary_lock.json').exists():raise RuntimeError('Use a new run directory; existing primary models are locked')
        checks=run_checks(config)
        write_json(out/'startup_checks.json',checks)
        progress('All synthetic startup checks passed.')
        config.update({'execution_date_local':args.date,'model_execution_enabled':bool(args.execute),
                       'preparation_only':not args.execute,'status':'execution_authorised' if args.execute else 'checks_only',
                       'solver_implementation':'explicit_scipy_trust_exact_on_prespecified_statsmodels_NB2_objective'})
        write_json(out/'config.executed.json',config)
        input_paths=list(store.paths.values())
        for extra in ['climate/district_grid_2019_2021/patient_location_weather_linkage.csv','tb_cleaned_google_rechecked_v2.csv']:
            path=args.data_root/extra
            if path.exists():input_paths.append(path)
        env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','xgboost','shap','matplotlib']}
        frozen={'date_local':args.date,'code_sha256':code_fingerprints(),
                'input_sha256':{str(p.resolve()):sha256(p) for p in input_paths},
                'original_config_sha256':sha256(args.config),'environment':env,
                'python':platform.python_version(),'platform':platform.platform(),
                'scope':'retrospective finalised in-extract notifications; 2021 errors not used in development',
                'numerical_clarification':'Same NB2/ridge objective; explicit trust-exact controls replace uncontrolled convenience ridge solver',
                'startup_SHAP_mode':checks['SHAP_mode']}
        write_json(out/'implementation_manifest.json',frozen)
        write_json(out/'cohort_summary.json',{'matched_rows':len(data),'train_rows':215,'test_rows':255,'bridge_rows':5,
            'matched_notifications':int(data.target.sum()),'training_notifications':int(data.loc[data.partition.eq('train'),'target'].sum()),
            'test_notifications':int(data.loc[data.partition.eq('test'),'target'].sum()),
            '2021_test_errors_inspected_at_design_freeze':False})
        status(out,'synthetic_checks_passed_no_study_fit')
        if args.stage=='check':return

    if args.stage in ['train','all']:
        train=data.loc[data.partition.eq('train')]
        settings={}
        for family in ['nb','boosting']:
            settings[family],log=tune_family(train,family,config,units,progress=progress)
            write_json(out/f'tuning_{family}.json',log)
            progress(f'{family} common selected setting: {settings[family]}')
        models=fit_final(data,units,settings,config)
        write_json(out/'selected_settings.json',settings)
        modes={};compat={}
        for s in ['C','E']:
            _,_,meta=tree_contributions(models[('boosting',s)],train,train.iloc[:10])
            modes[s]=meta['mode'];compat[s]=meta
        write_json(out/'training_SHAP_compatibility.json',compat)
        write_json(out/'fit_diagnostics.json',{MODEL_IDS[k]:m.diagnostics for k,m in models.items()})
        with (out/'fitted_models.private.pkl').open('wb') as f:pickle.dump(models,f)
        names=['selected_settings.json','training_SHAP_compatibility.json','fit_diagnostics.json','fitted_models.private.pkl','config.executed.json']
        write_json(out/'primary_lock.json',{'stage':'locked_before_test_scoring',
            'artefact_sha256':{n:sha256(out/n) for n in names},'SHAP_modes':modes,
            'development_fits':216,'final_primary_fits':6})
        status(out,'2020_development_completed_primary_models_locked')
        progress('Primary models and settings locked before 2021 error computation.')
        if args.stage=='train':return

    if args.stage in ['evaluate','all']:
        _,lock=verify_lock(out)
        config=json.loads((out/'config.executed.json').read_text())
        with (out/'fitted_models.private.pkl').open('rb') as f:models=pickle.load(f)
        test=data.loc[data.partition.eq('test')].sort_values(['week_start','unit'])
        train=data.loc[data.partition.eq('train')]
        predictions=frozen_forecasts(models,test)
        assert len(predictions)==2040
        save_evaluation(out/'primary',predictions,config)
        zero=predictions.loc[predictions.model_id.eq('P07') & predictions.unit.eq('WP PUTRAJAYA')].copy()
        zero['model_id']='D01';zero['prediction']=0
        save_evaluation(out/'diagnostics/always_zero_Putrajaya',zero,config)
        corr=train[FEATURES['E']].corr()
        write_json(out/'interpretation/training_correlations.json',corr.to_dict())
        for s in ['C','E']:
            m=models[('boosting',s)]
            summary,phi,base=explain(m,train,test,config,lock['SHAP_modes'][s])
            write_json(out/f'interpretation/{MODEL_IDS[('boosting',s)]}_SHAP.json',summary)
            np.savez_compressed(out/f'interpretation/{MODEL_IDS[('boosting',s)]}_contributions.private.npz',phi=phi,base=base)
        write_json(out/'primary_outputs_lock.json',{'primary_output_sha256':{str(p.relative_to(out)):sha256(p) for p in sorted((out/'primary').glob('*.json'))},
                    'no_test_based_model_revision':True})
        status(out,'primary_evaluation_and_interpretation_completed')
        progress('Primary 2021 evaluation and interpretation completed.')
        if args.stage=='evaluate':
            report(out);return

    if args.stage in ['sensitivities','all']:
        verify_lock(out)
        config=json.loads((out/'config.executed.json').read_text())
        settings=json.loads((out/'selected_settings.json').read_text())
        with (out/'fitted_models.private.pkl').open('rb') as f:models=pickle.load(f)
        primary_lock=json.loads((out/'primary_outputs_lock.json').read_text())
        for name,h in primary_lock['primary_output_sha256'].items():assert sha256(out/name)==h
        write_json(out/'audits/footprint_definition.json',audit_fixed_footprints(args.data_root))
        registry=run_sensitivities(store,config,settings,models,out/'sensitivities')
        try:location_quality_audit(args.data_root,out/'audits/location_quality.json')
        except FileNotFoundError:
            write_json(out/'audits/location_quality.json',{'status':'not_available','reason':'private optional patient-linkage files not provided'})
        for name,h in primary_lock['primary_output_sha256'].items():assert sha256(out/name)==h
        status(out,'all_prespecified_experiments_completed',primary_forecast_records=2040,
               primary_test_rows=255,sensitivity_families=len(registry),
               planned_hyperparameter_validation_fits=648,final_and_adaptive_study_fits=42)

    if args.stage in ['report','all','sensitivities']:
        report(out)
        progress(f'Aggregate results report saved in {out}')


if __name__=='__main__':main()
