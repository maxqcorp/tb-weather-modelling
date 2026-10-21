"""Aggregate scientific report and static figures from saved local results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write_json

LABELS={'P01':'Previous week','P02':'Four-week mean','P03':'NB baseline','P04':'NB core',
        'P05':'NB expanded','P06':'Boosting baseline','P07':'Boosting core','P08':'Boosting expanded','D01':'Always zero'}
SHORT={'temp_mean_c':'Temperature','rh_mean_pct':'Relative humidity','rain_total_mm':'Rainfall',
       'wind_mean_ms':'Wind speed','solar_total_mjm2':'Solar radiation','temp_range_c':'Temperature range'}


def weather_label(name):
    for key,label in SHORT.items():
        if key in name:return label
    return name


def load(path):return json.loads(Path(path).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def plot_primary(out):
    out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    metrics=load(out/'primary/metrics.json')
    pooled=[r for r in metrics if r['scope']=='pooled']
    pooled=sorted(pooled,key=lambda r:r['model_id'])
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric in zip(axes,['MAE','RMSE']):
        ax.barh([LABELS[r['model_id']] for r in pooled],[r[metric] for r in pooled],color=['#657785' if r['model_id']<'P06' else '#215B80' for r in pooled])
        ax.invert_yaxis();ax.set_xlabel(f'{metric} (notifications per district-week)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Held-out 2021 prediction errors: fixed models fitted on 2020')
    fig.savefig(figdir/'primary_errors.png',dpi=180);plt.close(fig)

    pred=pd.DataFrame(load(out/'primary/predictions.json'));pred.week_start=pd.to_datetime(pred.week_start)
    units=sorted(pred.unit.unique())
    fig,axes=plt.subplots(len(units),1,figsize=(12,11),sharex=True,layout='constrained')
    for ax,u in zip(axes,units):
        g=pred.loc[pred.unit.eq(u)]
        truth=g.loc[g.model_id.eq('P07')].sort_values('week_start')
        ax.plot(truth.week_start,truth.target,color='black',label='Observed notifications',linewidth=1.3)
        for model,col in [('P01','#A8A8A8'),('P04','#C37632'),('P07','#21628B')]:
            f=g.loc[g.model_id.eq(model)].sort_values('week_start')
            ax.plot(f.week_start,f.prediction,label=LABELS[model],color=col,linewidth=1.1,alpha=.9)
        ax.set_title(u,loc='left',fontsize=10);ax.set_ylabel('Count');ax.grid(alpha=.18)
    axes[0].legend(ncol=4,fontsize=8,loc='upper right')
    axes[-1].set_xlabel('Target week start in 2021')
    fig.suptitle('Sequential one-week-ahead forecasts using completed observed history')
    fig.savefig(figdir/'forecast_by_district.png',dpi=180);plt.close(fig)

    contrasts=[r for r in load(out/'primary/paired_uncertainty.json') if r['block_weeks']==4 and r['contrast'].startswith('RQ2')]
    if contrasts:
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        labels={'RQ2_core_vs_baseline':'Core versus baseline','RQ2_expanded_vs_core':'Expanded versus core','RQ2_expanded_vs_baseline':'Expanded versus baseline'}
        for i,r in enumerate(contrasts):
            lo,hi=r['MAE_gain_CI95'];point=r['MAE_gain']
            ax.plot([lo,hi],[i,i],color='#215B80',linewidth=2);ax.scatter([point],[i],color='#215B80')
        ax.set_yticks(range(len(contrasts)),[labels[r['contrast']] for r in contrasts]);ax.axvline(0,color='black',linestyle='--',linewidth=1)
        ax.set_xlabel('MAE gain (comparator minus new model); positive indicates improvement')
        ax.set_title('Boosting weather ablations: paired 95% intervals, four-week blocks')
        ax.grid(axis='x',alpha=.2);fig.savefig(figdir/'weather_gain_intervals.png',dpi=180);plt.close(fig)
    for model in ['P07','P08']:
        path=out/f'interpretation/{model}_SHAP.json'
        if not path.exists():continue
        summary=load(path)
        records=[r for r in summary['summaries'] if r['scope']=='all' and 'weather_rank' in r]
        records.sort(key=lambda r:r['weather_rank'])
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        ax.barh([weather_label(r['feature']) for r in records],[r['mean_absolute_SHAP'] for r in records],color='#215B80')
        ax.invert_yaxis();ax.set_xlabel('Mean absolute SHAP contribution (log expected count)')
        ax.set_title(LABELS[model]+': held-out weather contributions')
        fig.savefig(figdir/f'{model}_weather_SHAP.png',dpi=180);plt.close(fig)
        district=[r for r in summary['summaries'] if r['scope']=='district' and 'weather_rank' in r]
        heat=pd.DataFrame(district).pivot(index='feature',columns='group',values='weather_rank').reindex([r['feature'] for r in records])
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        im=ax.imshow(heat.to_numpy(),cmap='Blues_r',vmin=1,vmax=len(records),aspect='auto')
        ax.set_yticks(range(len(heat)),[weather_label(x) for x in heat.index]);ax.set_xticks(range(len(heat.columns)),heat.columns,rotation=20,ha='right',fontsize=8)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):ax.text(j,i,f'{heat.iloc[i,j]:g}',ha='center',va='center',color='black')
        fig.colorbar(im,ax=ax,label='Rank (1 = largest mean absolute contribution)')
        ax.set_title(LABELS[model]+': weather ranks by recorded district')
        fig.savefig(figdir/f'{model}_rank_by_district.png',dpi=180);plt.close(fig)


def report(out):
    out=Path(out)
    config=load(out/'config.executed.json')
    selected=load(out/'selected_settings.json')
    checks=load(out/'startup_checks.json')
    metrics=load(out/'primary/metrics.json')
    intervals=load(out/'primary/paired_uncertainty.json')
    pooled={r['model_id']:r for r in metrics if r['scope']=='pooled'}
    main=next(r for r in intervals if r['contrast']=='RQ2_core_vs_baseline' and r['block_weeks']==4)
    expanded=next(r for r in intervals if r['contrast']=='RQ2_expanded_vs_core' and r['block_weeks']==4)
    def interval_text(r):return f"{r['MAE_gain']:.3f} notifications per district-week (95% interval {r['MAE_gain_CI95'][0]:.3f} to {r['MAE_gain_CI95'][1]:.3f})"
    status=load(out/'run_status.json')
    text=[
        '# TB weather experiment results',
        '',f"Run date: {config['execution_date_local']}. Retrospective one-week-ahead notification forecasting.",
        '',f"The core-weather boosting model had a held-out MAE of {pooled['P07']['MAE']:.3f}, compared with {pooled['P06']['MAE']:.3f} for baseline boosting. The paired MAE gain from core weather was {interval_text(main)}. Positive gains favour the new model.",
        '',f"The expanded-versus-core boosting gain was {interval_text(expanded)}. These comparisons concern this extract and evaluation period; they do not establish causal meteorological or climate-change effects.",
        '', '## Design and execution',
        '', 'The primary cohort contained 215 training rows from 43 target weeks wholly in 2020 and 255 test rows from 51 target weeks wholly in 2021. The cross-year week starting 28 December 2020 was excluded as a target but retained as completed history. All five recorded districts used the same temporal masks. Four KL districts shared the fixed KL ERA5 series; Putrajaya used the second city series.',
        '', 'A model was fitted once on the eligible 2020 targets. Completed observed past weeks refreshed predictors through 2021; parameters and preprocessing remained fixed in the primary evaluation. Three chronological 2020 folds selected one common setting per family across baseline, core and expanded sets.',
        '', 'The NB2 implementation used the prespecified likelihood and ridge penalty with an explicit trust-exact optimiser, maximum 2,000 iterations and gradient tolerance 0.00000001. This implementation clarification was checked on synthetic data before study fitting because the installed convenience routine did not pass the required solver controls to its ridge branch.',
        '', f"Selected boosting settings: `{json.dumps(selected['boosting'],sort_keys=True)}`. Selected NB2 settings: `{json.dumps(selected['nb'],sort_keys=True)}`.",
        '', '## Primary performance',
        '',table(['Model','MAE','RMSE','Mean signed error'],[[LABELS[m],f"{pooled[m]['MAE']:.3f}",f"{pooled[m]['RMSE']:.3f}",f"{pooled[m]['mean_signed_error']:.3f}"] for m in sorted(pooled)]),
        '', 'Errors are notifications per district-week. Positive signed error indicates overprediction. Equal-district macro MAE equals pooled MAE in this balanced panel; macro RMSE is reported separately in the metric records.',
        '', '![Primary error comparison](figures/primary_errors.png)',
        '', '## District performance',
        '',table(['District','Persistence MAE','NB core MAE','Boosting core MAE','Boosting expanded MAE'],[
            [u]+[f"{next(r['MAE'] for r in metrics if r['scope']=='unit' and r['group']==u and r['model_id']==m):.3f}" for m in ['P01','P04','P07','P08']]
            for u in sorted({r['group'] for r in metrics if r['scope']=='unit'})]),
        '', '![Forecasts by district](figures/forecast_by_district.png)',
        '', '## Paired comparisons and uncertainty',
        '',table(['Contrast','Block weeks','MAE gain','95% interval','MAE gain percent'],[[r['contrast'],r['block_weeks'],f"{r['MAE_gain']:.3f}",f"{r['MAE_gain_CI95'][0]:.3f} to {r['MAE_gain_CI95'][1]:.3f}",f"{r['percentage_MAE_gain']:.2f}" if r['percentage_MAE_gain'] is not None else 'Undefined'] for r in intervals]),
        '', 'Every draw sampled time blocks with all districts and models retained together. Intervals are pointwise and conditional on the fitted models. A positive estimate with an interval spanning zero is uncertain evidence of improvement. Secondary contrasts and subgroup patterns do not constitute independent confirmatory findings.',
        '', '![Paired weather gains](figures/weather_gain_intervals.png)',
        '', '## Interpretation and stability'
    ]
    for model in ['P07','P08']:
        s=load(out/f'interpretation/{model}_SHAP.json')
        overall=sorted([r for r in s['summaries'] if r['scope']=='all' and 'weather_rank' in r],key=lambda r:r['weather_rank'])
        text += ['',f"{LABELS[model]} used {s['metadata']['mode']} contributions on the log expected count margin. Its maximum additivity error was {s['metadata']['max_additivity_error']:.3g}.",'',
            table(['Weather variable','Rank','Mean absolute SHAP','Top-two block frequency'],[[weather_label(r['feature']),r['weather_rank'],f"{r['mean_absolute_SHAP']:.4f}",f"{r['top_two_block_frequency']:.3f}"] for r in overall]),
            '',f'![Weather contributions for {model}](figures/{model}_weather_SHAP.png)',
            '',table(['Variable','Top-two district-quarter fraction','Descriptive stability convention met'],[[weather_label(r['feature']),f"{r['top_two_group_fraction']:.3f}" if r['top_two_group_fraction'] is not None else 'Undefined',r['descriptive_75_percent_convention_met']] for r in s['descriptive_stability']])]
    text += ['', 'Correlated weather/calendar predictors can redistribute SHAP credit. District differences under shared city exposures describe the model and notification histories, not distinct measured district climates. Ranking resamples vary explained weeks rather than refitting models.', '', '## Sensitivity analyses']
    registry_path=out/'sensitivities/registry.json'
    if registry_path.exists():
        registry=load(registry_path);rows=[]
        for rec in registry:
            ints=load(out/f"sensitivities/{rec['id']}/paired_uncertainty.json")
            r=next((x for x in ints if x['contrast']=='RQ2_core_vs_baseline' and x['block_weeks']==4),None)
            rows.append([rec['id'],json.dumps(rec['changes'],sort_keys=True),rec.get('training_rows','Expanding'),rec['test_rows'],f"{r['MAE_gain']:.3f}" if r else 'Not applicable',f"{r['MAE_gain_CI95'][0]:.3f} to {r['MAE_gain_CI95'][1]:.3f}" if r else 'Not applicable'])
        text += ['',table(['Run','Change','Training rows','Test rows','Core MAE gain','95% interval'],rows),
                 '', 'The city sensitivity changes outcome scale and is compared within that scale. The NASA sensitivity changes source and day convention and supplies identical city meteorological histories. The old footprint sensitivity remains exploratory and combines spatial/source differences. All prespecified sensitivity outputs are retained; none replaces the primary analysis.']
    else:text += ['', 'Sensitivity execution is not yet complete; consult run_status.json.']
    text += ['', '## Quality control and reproducibility',
        '', f"Synthetic startup checks passed: {checks['all_passed']}. Study execution status: {status['state']}. Source hashes, code hashes, exact package versions, tuning cell scores, fit diagnostics and frozen models are retained locally. Study fitting and test scoring are separated by a pre-evaluation model/settings lock.",
        '', 'The publication package contains code, configuration, requirements and synthetic checks only. Patient-level records, actual input tables, fitted models, private predictions and API credentials are excluded. Exact numerical reproduction of this study requires authorised access to the same aggregate inputs; synthetic checks alone do not reproduce the reported study estimates.',
        '', '## Limitations',
        '', 'The extract does not independently certify reporting completeness, zero-notification weeks, unique TB episodes or the exact Daerah reporting jurisdiction. Finalised counts and reanalysis do not establish historical real-time availability. Only two annual cycles and two nearby weather histories are represented. The patient-location quality subset is outcome-selected and does not validate full-cohort residence. Conclusions apply to retrospective predictive performance in this extract, not infection risk or climate-change causation.',
        '', 'Detailed numerical records are stored in primary/, interpretation/ and sensitivities/. Results are derived from actual model execution; no missing or failed run is replaced with an invented estimate.'
    ]
    (out/'results_report.md').write_text('\n'.join(text)+'\n')
    plot_primary(out)
