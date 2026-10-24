"""Draw review figures from frozen aggregate outputs without fitting study models."""
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
from collections import defaultdict
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True, help='Private completed study run with saved aggregate outputs')
parser.add_argument('--output-dir', type=Path, required=True, help='New private directory for all six manuscript figures')
args = parser.parse_args()
RUN = args.run_dir.resolve()
OUT = args.output_dir.resolve()
if not RUN.is_dir():
    parser.error('--run-dir must be an existing completed study run')
if OUT == RUN or RUN in OUT.parents:
    parser.error('--output-dir must be outside the archived study run')
PACKAGE_ROOT = Path(__file__).resolve().parent
if OUT == PACKAGE_ROOT or PACKAGE_ROOT in OUT.parents:
    parser.error('--output-dir must be outside the public code package')
if OUT.exists():
    parser.error('--output-dir must be a new directory')
OUT.mkdir(parents=True)

BLUE, GREEN, RED = '#0000ff', '#008000', '#ff0000'
read_files = {}
checks = {}
records = []
layout_warnings = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    path = RUN / name
    read_files[str(path.relative_to(RUN))] = sha(path)
    return json.loads(path.read_text())

metrics = read('primary/metrics.json')
predictions = read('primary/predictions.json')
uncertainty = read('primary/paired_uncertainty.json')
cohort = read('cohort_summary.json')
registry = read('sensitivities/registry.json')
explanations = {m: read(f'interpretation/{m}_SHAP.json') for m in ['P07', 'P08']}
units = ['CHERAS', 'KEPONG', 'LEMBAH PANTAI', 'TITIWANGSA', 'WP PUTRAJAYA']
model_ids = [f'P{i:02d}' for i in range(1, 9)]
model_names = ['Previous week', 'Four-week mean', 'NB baseline', 'NB core', 'NB expanded',
               'Boosting baseline', 'Boosting core', 'Boosting expanded']
short_names = ['P01\nPrevious\nweek', 'P02\nFour-week\nmean', 'P03\nNB\nbaseline',
               'P04\nNB\ncore', 'P05\nNB\nexpanded', 'P06\nBoosting\nbaseline',
               'P07\nBoosting\ncore', 'P08\nBoosting\nexpanded']

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 13, 'axes.titlesize': 15,
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.linewidth': 1.0, 'axes.edgecolor': 'black', 'axes.facecolor': 'white',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'text.color': 'black',
    'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True,
    'ytick.right': True, 'grid.color': '#999999', 'grid.linestyle': ':',
    'grid.linewidth': .65, 'axes.axisbelow': True, 'legend.frameon': True,
    'legend.fancybox': False, 'legend.edgecolor': 'black', 'legend.framealpha': 1,
    'legend.fontsize': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.unicode_minus': True,
})

def classic(ax, grid='both'):
    ax.tick_params(which='both', direction='in', top=True, right=True, length=5)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color('black')
        side.set_linewidth(1)
    if grid:
        ax.grid(True, axis=grid)
    return ax

def panel(ax, label, title):
    ax.set_title(f'{label}  {title}', loc='left', pad=13, fontweight='normal')

def save(fig, number, slug, caption):
    name = f'figure_{number}_{slug}'
    fig.canvas.draw()
    # Labels retain at least about 7.4 pt at the manuscript's 6.65 inch width.
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_fontsize() < 14:
            label.set_fontsize(14)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or (text.get_clip_on() and text not in fig.texts):
            continue
        bbox = text.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fw + 1 or bbox.y1 > fh + 1:
            layout_warnings.append({'figure': number, 'text': text.get_text(), 'bbox': list(bbox.bounds)})
    for ext in ['pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}')
    fig.savefig(OUT / f'{name}.png', dpi=450)
    preview = OUT / 'previews' / f'{name}.png'
    preview.parent.mkdir(exist_ok=True)
    fig.savefig(preview, dpi=160)
    records.append({'figure': number, 'name': name, 'caption': caption,
                    'preview': str(preview.relative_to(OUT)),
                    'formats': ['png', 'pdf', 'svg'],
                    'files_sha256': {ext: sha(OUT / f'{name}.{ext}') for ext in ['png', 'pdf', 'svg']}})
    plt.close(fig)

def metric(model, key, unit=None):
    row = next(r for r in metrics if r['model_id'] == model and
               r['scope'] == ('unit' if unit else 'pooled') and
               r['group'] == (unit if unit else 'all'))
    return row[key]

def contrast(name, block=4):
    return next(r for r in uncertainty if r['contrast'] == name and r['block_weeks'] == block)

# Verify plotted primary errors directly against the saved aggregate forecasts.
for model in model_ids:
    rows = [r for r in predictions if r['model_id'] == model]
    checks[f'{model}_255_predictions'] = len(rows) == 255
    errors = np.array([r['prediction'] - r['target'] for r in rows])
    checks[f'{model}_pooled_MAE'] = bool(np.isclose(np.mean(abs(errors)), metric(model, 'MAE'), atol=1e-12))
    checks[f'{model}_pooled_RMSE'] = bool(np.isclose(np.sqrt(np.mean(errors**2)), metric(model, 'RMSE'), atol=1e-12))
    for unit in units:
        e = np.array([r['prediction'] - r['target'] for r in rows if r['unit'] == unit])
        checks[f'{model}_{unit}_MAE'] = len(e) == 51 and bool(np.isclose(np.mean(abs(e)), metric(model, 'MAE', unit), atol=1e-12))
targets = defaultdict(set)
for row in predictions:
    targets[(row['unit'], row['week_start'])].add(row['target'])
checks['observations_identical_across_models'] = all(len(v) == 1 for v in targets.values())
checks['cohort_accounting'] = cohort['train_rows'] + cohort['bridge_rows'] + cohort['test_rows'] == cohort['matched_rows']
checks['notification_accounting'] = cohort['training_notifications'] + 34 + cohort['test_notifications'] == cohort['matched_notifications']

# Figure 1 is a vector schematic with a quantitative fold timeline.
fig = plt.figure(figsize=(12.5, 10.1))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.2], left=.115, right=.97, top=.94, bottom=.12, hspace=.35)
ax = fig.add_subplot(gs[0]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
panel(ax, 'A', 'Study workflow')
box_w = .30
def box(x, y, text, w=box_w, h=.22):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor='black', linewidth=1.05))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=15, linespacing=1.25)
def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1, color='black'))
box(.29, .78, 'Eligible matched cohort\n475 district-weeks\n3,334 notification records', w=.42, h=.22)
box(.00, .43, '2020 development\n215 district-weeks\n1,575 records')
box(.35, .43, 'Bridge target excluded\n5 district-weeks\n34 records')
box(.70, .43, '2021 held-out test\n255 district-weeks\n1,725 records')
arrow((.40, .78), (.15, .65)); arrow((.50, .78), (.50, .65)); arrow((.60, .78), (.85, .65))
box(.00, .02, '2020 model selection\nThree chronological folds')
box(.35, .02, 'Final 2020 fitting\nOne setting per family')
box(.70, .02, '2021 evaluation\nParameters held fixed')
arrow((.15, .43), (.15, .24)); arrow((.30, .13), (.35, .13))
arrow((.65, .13), (.70, .13)); arrow((.85, .43), (.85, .24))
ax2 = fig.add_subplot(gs[1]); classic(ax2, 'x'); panel(ax2, 'B', 'Chronological development and evaluation')
date = lambda s: mdates.date2num(datetime.fromisoformat(s))
start = date('2020-03-02'); dev_end = date('2020-12-28'); bridge_end = date('2021-01-04'); end = date('2021-12-27')
ax2.broken_barh([(start, dev_end-start)], (3.55, .52), facecolors=BLUE, edgecolors='black', linewidths=.65)
ax2.broken_barh([(dev_end, bridge_end-dev_end)], (3.55, .52), facecolors='white', edgecolors='black', hatch='////', linewidths=.65)
ax2.broken_barh([(bridge_end, end-bridge_end)], (3.55, .52), facecolors=RED, edgecolors='black', linewidths=.65)
ax2.text((start+dev_end)/2, 3.81, '43 development weeks', color='white', ha='center', va='center', fontsize=11.5)
ax2.text((bridge_end+end)/2, 3.81, '51 test weeks', color='white', ha='center', va='center', fontsize=11.5)
folds = [(19, '2020-07-13', '2020-09-07'), (27, '2020-09-07', '2020-11-02'), (35, '2020-11-02', '2020-12-28')]
for y, (weeks, split, stop) in zip([2.6, 1.7, .8], folds):
    split, stop = date(split), date(stop)
    ax2.broken_barh([(start, split-start)], (y-.24, .48), facecolors=BLUE, edgecolors='black', linewidths=.65)
    ax2.broken_barh([(split, stop-split)], (y-.24, .48), facecolors=GREEN, edgecolors='black', linewidths=.65)
    ax2.text((start+split)/2, y, f'{weeks} weeks', color='white', ha='center', va='center', fontsize=11.5)
    ax2.text((split+stop)/2, y, '8 weeks', color='white', ha='center', va='center', fontsize=11.5)
ax2.set_yticks([3.81, 2.6, 1.7, .8], ['Primary', 'Fold 1', 'Fold 2', 'Fold 3'])
ax2.set_ylim(.2, 4.5); ax2.set_xlim(date('2020-02-15'), date('2022-01-10'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 7, 11])); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.set_xlabel('Target week start')
ax2.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, ec='black', label='Fitting'),
                    Rectangle((0, 0), 1, 1, fc=GREEN, ec='black', label='Validation'),
                    Rectangle((0, 0), 1, 1, fc=RED, ec='black', label='Test'),
                    Rectangle((0, 0), 1, 1, fc='white', ec='black', hatch='////', label='Bridge target excluded')], loc='lower right', fontsize=10.5)
fig.text(.54, .027, 'Model selection used 2020 only. Completed observed history updated the 2021 inputs.\nHistorical availability of finalised counts and weather was assumed retrospectively.', ha='center', fontsize=12.5, linespacing=1.4)
save(fig, 1, 'workflow_and_temporal_design', 'The matched cohort was divided into 2020 development targets, an excluded bridge target and 2021 test targets before model selection. Tuning used three chronological folds entirely within 2020. The bridge week beginning 28 December 2020 remained usable as completed history. Primary parameters were fixed during 2021 while completed observed history updated the inputs. Historical availability of finalised counts and weather was assumed retrospectively. Four Kuala Lumpur districts shared one city weather history. Putrajaya had a separate history.')

# Figure 2 retains the observed weekly chronology and the saved fractional forecasts.
fig = plt.figure(figsize=(12.5, 10.5))
gs = fig.add_gridspec(3, 2, left=.08, right=.98, top=.83, bottom=.13, hspace=.43, wspace=.17)
axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[2,:])]
line_specs = [('P02', RED, '--', 'Four-week mean'), ('P04', GREEN, '-.', 'NB core'), ('P06', BLUE, '--', 'Boosting baseline'), ('P07', BLUE, '-', 'Boosting core')]
for ax, unit in zip(axes, units):
    classic(ax)
    rows = sorted([r for r in predictions if r['model_id']=='P07' and r['unit']==unit], key=lambda r:r['week_start'])
    dates = [datetime.fromisoformat(r['week_start']) for r in rows]
    ax.plot(dates, [r['target'] for r in rows], color='black', lw=1.25, marker='.', markersize=4, label='Observed')
    for m, color, ls, label in line_specs:
        rr = sorted([r for r in predictions if r['model_id']==m and r['unit']==unit], key=lambda r:r['week_start'])
        ax.plot(dates, [r['prediction'] for r in rr], color=color, ls=ls, lw=1.45, label=label)
    ax.set_title(unit, loc='left', fontsize=15, pad=8)
    ax.set_ylabel('Notifications', fontsize=13)
    ax.set_ylim(bottom=0); ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.set_xlim(datetime(2020, 12, 28), datetime(2021, 12, 27))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(labelsize=13)
fig.suptitle('Weekly observed and predicted TB notifications', y=.98, fontsize=17)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .94), ncol=3, fontsize=13)
fig.text(.54, .078, 'Target week start in 2021', ha='center', fontsize=14)
fig.text(.54, .024, 'District panels have different vertical scales. Curves show point forecasts.\nThe baseline and core boosting curves overlap closely in several panels.', ha='center', fontsize=12.5, linespacing=1.3)
save(fig, 2, 'weekly_forecasts_by_district', 'Observed weekly notification records are compared with the four-week mean, core NB2 regression and baseline and core boosting. The four-week mean was the stronger of the two prespecified simple benchmarks by pooled MAE. Baseline boosting had the lowest pooled MAE. All eight primary models remain reported in Table 5 and Figure 3. Parameters were fitted using 2020 targets and remained fixed during 2021. District panels have different vertical scales. The curves show point forecasts without calibrated prediction intervals.')

# Figure 3 combines overall error with district-specific errors without inventing CIs.
fig = plt.figure(figsize=(12.5, 10.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=.20, right=.93, top=.94, bottom=.13, hspace=.38)
ax = fig.add_subplot(gs[0]); classic(ax, 'x'); panel(ax, 'A', 'Overall forecast errors')
y = np.arange(8)
mae = np.array([metric(m, 'MAE') for m in model_ids]); rmse = np.array([metric(m, 'RMSE') for m in model_ids])
ax.plot(mae, y-.12, 'o', color=BLUE, ms=7, label='MAE')
ax.plot(rmse, y+.12, 's', color=GREEN, ms=6.5, mfc='white', mew=1.3, label='RMSE')
ax.set_yticks(y, [f'{m}  {name}' for m, name in zip(model_ids, model_names)])
ax.invert_yaxis(); ax.set_ylim(7.65, -.65); ax.set_xlim(1.95, 4.08)
ax.set_xlabel('Error in notifications per district-week')
ax.legend(loc='lower right', ncol=2)
for i, value in enumerate(mae):
    ax.annotate(f'{value:.3f}', (value, i-.12), xytext=(-6, -2), textcoords='offset points', ha='right', va='center', fontsize=10.5, color=BLUE)
ax.text(.98, .93, 'Lower values indicate smaller errors', transform=ax.transAxes, ha='right', fontsize=11)
ax = fig.add_subplot(gs[1]); panel(ax, 'B', 'District forecast MAE')
matrix = np.array([[metric(m, 'MAE', u) for m in model_ids] for u in units])
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=4, aspect='auto', interpolation='nearest')
ax.set_xticks(np.arange(8), short_names, fontsize=10.5); ax.set_yticks(np.arange(5), units, fontsize=11)
ax.tick_params(direction='in', top=True, right=True, length=4)
for i in range(5):
    for j in range(8):
        ax.text(j, i, f'{matrix[i,j]:.4f}', ha='center', va='center', fontsize=10.5, color='white' if matrix[i,j]>2.35 else 'black')
ax.set_xticks(np.arange(-.5, 8, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=.8); ax.tick_params(which='minor', length=0)
cax=fig.add_axes([.947, .13, .014, .351]);cb=fig.colorbar(im, cax=cax); cb.set_ticks([0,1,2,3,4]);cb.ax.tick_params(labelsize=10)
fig.text(.56, .027, 'Numbers are MAE. Differences are descriptive and do not establish model superiority.\nDistrict counts have different scales, so model comparisons should be made within each row.', ha='center', fontsize=11.5, linespacing=1.4)
save(fig, 3, 'model_errors_and_district_heatmap', 'Panel A shows pooled MAE and RMSE for all eight primary models across the same 255 held-out district-week observations. Blue point labels give MAE. Panel B shows district-specific MAE. These are observed errors without uncertainty intervals. Small numerical differences do not establish model superiority. Table 6 reports paired uncertainty for the prespecified comparisons. District outcomes have different count scales, so the heatmap supports comparisons between models within each district.')

# Figure 4 exposes the small negative upper bound explicitly rather than rounding to zero.
fig, ax = plt.subplots(figsize=(12.5, 6.9));fig.subplots_adjust(left=.32, right=.97, top=.9, bottom=.23)
classic(ax, 'x')
ax.set_title('Incremental weather value', fontsize=17, pad=20)
names = ['RQ2_core_vs_baseline', 'RQ2_expanded_vs_core', 'RQ2_expanded_vs_baseline', 'NB_core_vs_baseline', 'NB_expanded_vs_core']
labs = ['Boosting core versus baseline', 'Boosting expanded versus core', 'Boosting expanded versus baseline', 'NB core versus baseline', 'NB expanded versus core']
for i, name in enumerate(names):
    r = contrast(name); value = r['MAE_gain']; lo, hi = r['MAE_gain_CI95']; color = BLUE if i<3 else GREEN
    ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o' if i<3 else 's', ms=8, color=color, ecolor=color, capsize=5, lw=1.4, mfc=color if i==0 else 'white', mew=1.2)
    ax.text(.012, i-.21, f'{value:.6f}  [{lo:.6f}, {hi:.6f}]', fontsize=10.5, transform=ax.get_yaxis_transform(), va='center')
ax.axvline(0, color='black', ls='--', lw=1); ax.axhline(2.5, color='#777777', ls=':', lw=.75)
ax.set_yticks(np.arange(5), labs); ax.set_ylim(4.6, -.75); ax.set_xlim(-.05, .016)
ax.xaxis.set_major_locator(MultipleLocator(.01))
ax.set_xlabel('MAE gain in notifications per district-week')
fig.text(.65, .115, 'Negative gain favours comparator     Positive gain favours new model', ha='center', fontsize=14)
fig.text(.5, .041, 'Points and conditional 95% paired block-bootstrap intervals using four-week blocks and 2,000 draws.\nThe primary core-versus-baseline boosting upper bound is −0.000115, which lies narrowly below zero.', ha='center', fontsize=11.5, linespacing=1.4)
save(fig, 4, 'incremental_weather_gain', 'Gain equals comparator MAE minus new-model MAE. Positive values favour the new model. Points and lines show estimates and conditional pointwise 95% intervals from 2,000 paired moving-block draws using four-week blocks. The primary boosting comparison is the filled blue point. Six decimal places retain the sign of its small negative upper bound. The 0.41% MAE deterioration does not establish an operationally important effect. These intervals exclude model-fitting and selection uncertainty and are not prediction intervals for future counts.')

# Figure 5 uses a separate weather scale because the count-history contribution dominates.
all_shap = {m: {a['feature']: a['mean_absolute_SHAP'] for a in explanations[m]['summaries'] if a['scope']=='all' and a['group']=='all'} for m in explanations}
weather_features = ['era5_rh_mean_pct_prev4w','era5_rain_total_mm_prev4w','era5_temp_mean_c_prev4w','era5_wind_mean_ms_prev4w','era5_solar_total_mjm2_prev4w','era5_temp_range_c_prev4w']
pretty = {'tb_count_prev4w_mean':'Previous four-week\ncount mean', 'covid_stringency_prevweek':'Previous-week policy', 'tb_count_lag1':'Count lag 1', 'tb_count_lag2':'Count lag 2', 'unit_KEPONG':'Kepong indicator', 'era5_rh_mean_pct_prev4w':'Relative humidity', 'era5_rain_total_mm_prev4w':'Rainfall', 'era5_temp_mean_c_prev4w':'Temperature', 'era5_wind_mean_ms_prev4w':'Wind speed', 'era5_solar_total_mjm2_prev4w':'Solar radiation', 'era5_temp_range_c_prev4w':'Temperature range'}
nonzero = sorted({f for d in all_shap.values() for f, v in d.items() if v>0}, key=lambda f: max(d.get(f,0) for d in all_shap.values()), reverse=True)
shares = {m: sum(d.get(f,0) for f in weather_features) / sum(d.values())*100 for m,d in all_shap.items()}
checks['core_weather_attribution_share'] = round(shares['P07'], 2)==1.48
checks['expanded_weather_attribution_share'] = round(shares['P08'], 2)==2.63
fig = plt.figure(figsize=(12.5, 8.7))
gs=fig.add_gridspec(1,2,left=.18,right=.97,bottom=.24,top=.85,wspace=.67,width_ratios=[1,1])
for j, (features, title, xmax) in enumerate([(nonzero, 'Overall predictor contributions', .72), (weather_features, 'Weather contributions', .0168)]):
    ax=fig.add_subplot(gs[j]);classic(ax,'x');panel(ax, 'A' if j==0 else 'B', title)
    for i, f in enumerate(features):
        for m, color, offset, hatch in [('P07', BLUE, -.19, None), ('P08', GREEN, .19, '///')]:
            y=i+offset
            if f not in all_shap[m]:
                ax.text(xmax*.015,y,'Not included',va='center',fontsize=10.5)
            else:
                v=all_shap[m][f]
                ax.barh(y,v,height=.30,color=color,edgecolor=color,hatch=hatch,alpha=.88)
                ax.text(v+xmax*.013,y,f'{v:.4f}' if j==0 else f'{v:.6f}',va='center',fontsize=10)
    ax.set_yticks(np.arange(len(features)),[pretty[f]for f in features],fontsize=12)
    ax.set_ylim(len(features)-.4,-.7);ax.set_xlim(0,xmax);ax.set_xlabel('Mean absolute SHAP value\nLog expected count scale',fontsize=13)
    if j==1:ax.xaxis.set_major_locator(MultipleLocator(.004))
fig.legend(handles=[Rectangle((0,0),1,1,fc=BLUE,ec=BLUE,label='Core boosting'),Rectangle((0,0),1,1,fc=GREEN,ec=GREEN,hatch='///',label='Expanded boosting')],loc='upper center',bbox_to_anchor=(.57,.98),ncol=2,fontsize=13)
fig.text(.57,.905,f'Weather attribution share     Core {shares["P07"]:.2f}%     Expanded {shares["P08"]:.2f}%',ha='center',fontsize=13)
fig.text(.57,.04,'Panel A includes features with a nonzero contribution in either model. Panel B uses a separate scale.\nAttribution shares describe model contributions and do not measure accuracy or causal effects.',ha='center',fontsize=11.5,linespacing=1.4)
save(fig,5,'shap_predictor_and_weather_contributions','Mean absolute interventional Tree SHAP contributions describe the frozen core and expanded boosting models across 255 held-out observations. All 215 training observations formed the background. Values are on the log expected count scale. Panel A includes predictors with a nonzero contribution in either model. Panel B uses a different horizontal scale. Not included identifies variables absent from the core model. Zero values refer to the evaluated observations. Weather shares of 1.48% and 2.63% describe separate models and do not measure accuracy, explained variance or causal effects.')

# Figure 6 separates city and district units rather than pooling unlike error scales.
labels = {'S01':'City aggregation', 'S02':'ERA5-Land and ERA5 mixture', 'S03':'NASA core weather', 'S04':'14-day weather window', 'S05':'56-day weather window', 'S06':'Exclude revised diagnoses', 'S07':'One extra week of delay', 'S08':'Omit policy stringency', 'S09':'Quarterly refitting', 'S10':'Old district footprints'}
sensitivity = {}
for r in registry:
    sensitivity[r['id']] = next(a for a in read(f'sensitivities/{r["id"]}/paired_uncertainty.json') if a['contrast']=='RQ2_core_vs_baseline' and a['block_weeks']==4)
fig = plt.figure(figsize=(12.5, 10.1))
gs=fig.add_gridspec(2,1,left=.31,right=.96,top=.92,bottom=.18,height_ratios=[5.3,1.15],hspace=.52)
ax=fig.add_subplot(gs[0]);classic(ax,'x');panel(ax,'A','District-week analyses')
rows=[('Primary',contrast('RQ2_core_vs_baseline'))]+[(k,sensitivity[k])for k in labels if k!='S01']
for i,(k,r)in enumerate(rows):
    v=r['MAE_gain'];lo,hi=r['MAE_gain_CI95'];color=BLUE if k=='Primary' else RED if k=='S10' else GREEN
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='D'if k=='Primary'else '^'if k=='S10'else 'o',ms=7,color=color,ecolor=color,capsize=4,lw=1.3,mfc=color if k=='Primary'else'white',mew=1.2)
    if k=='S07':ax.annotate('Identical saved forecasts',(v,i),xytext=(10,-1),textcoords='offset points',va='center',fontsize=10.5)
ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(range(len(rows)),['Primary  28-day ERA5']+[f'{k}  {labels[k]}'for k,_ in rows[1:]])
ax.set_ylim(len(rows)-.5,-.65);ax.set_xlim(-.072,.035);ax.xaxis.set_major_locator(MultipleLocator(.02))
ax.set_xlabel('MAE gain in notifications per district-week')
ax=fig.add_subplot(gs[1]);classic(ax,'x');panel(ax,'B','City aggregation assessed on a separate outcome scale')
r=sensitivity['S01'];v=r['MAE_gain'];lo,hi=r['MAE_gain_CI95']
ax.errorbar(v,0,xerr=[[v-lo],[hi-v]],fmt='o',color=GREEN,mfc='white',ms=7,mew=1.2,capsize=4)
ax.axvline(0,color='black',ls='--',lw=1);ax.set_xlim(-.072,.035);ax.xaxis.set_major_locator(MultipleLocator(.02))
ax.set_yticks([0],['S01  City aggregation']);ax.set_ylim(-.75,.75);ax.set_xlabel('MAE gain in notifications per city-week')
ax.annotate('Identical saved forecasts',(0,0),xytext=(10,0),textcoords='offset points',va='center',fontsize=10.5)
fig.text(.55,.985,'Core weather versus baseline across sensitivity analyses',ha='center',va='top',fontsize=17)
fig.text(.57,.075,'Positive gain favours core weather. Conditional 95% intervals use four-week paired blocks and 2,000 draws.\nZero-width intervals reflect identical saved forecasts. S10 is exploratory. Compare effects within each analysis.',ha='center',fontsize=11.5,linespacing=1.4)
save(fig,6,'sensitivity_analysis_forest','Gain equals baseline MAE minus core-weather MAE. Positive values favour core weather. Conditional pointwise 95% intervals use 2,000 paired moving-block draws with four-week blocks. Panel A uses district-week units. Panel B separates the city-level analysis because its outcome scale differs. S01 and S06 retuned the models within 2020. The remaining analyses retained primary boosting settings. S01 and S07 produced identical saved forecasts and zero-width intervals, which do not establish future equivalence. S10 is exploratory. The analyses change different conditions and are not independent replications. Interpret each contrast within its own analysis.')

checks['source_files_unchanged'] = all(sha(RUN/k)==v for k,v in read_files.items())
checks['six_figures_created'] = len(records)==6
checks['no_text_outside_canvas'] = not layout_warnings
manifest={'created_on':datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).strftime('%d %B %Y'),'renderer':'Matplotlib reproducing classic MATLAB plotting style','manuscript_modified':False,'scope':'Saved aggregate predictions, metrics, uncertainty and SHAP summaries only. No patient-source reading or model fitting.','source_files_sha256':read_files,'checks':checks,'all_passed':all(checks.values()),'layout_warnings':layout_warnings,'figures':records}
(OUT/'figure_verification.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
(OUT/'captions.txt').write_text('\n\n'.join(f'Figure {r["figure"]}\n{r["caption"]}'for r in records)+'\n')
print(json.dumps({'figures':len(records),'checks':len(checks),'all_passed':all(checks.values()),'failed':[k for k,v in checks.items()if not v],'layout_warnings':layout_warnings},indent=2))
assert all(checks.values()), 'Review verification or layout warnings before delivery.'
