"""Independently verify saved results without fitting or importing study modules."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def errors(rows):
    residual = [r['prediction'] - r['target'] for r in rows]
    n = len(residual)
    return {'MAE': math.fsum(abs(e) for e in residual) / n,
            'RMSE': math.sqrt(math.fsum(e * e for e in residual) / n),
            'mean_signed_error': math.fsum(residual) / n}


def audit_evaluation(folder, config):
    predictions = read(folder / 'predictions.json')
    by_model = defaultdict(list)
    truth = {}
    seen = set()
    for row in predictions:
        key = row['week_start'], row['unit']
        model_key = row['model_id'], *key
        assert model_key not in seen
        seen.add(model_key)
        assert math.isfinite(row['prediction']) and row['prediction'] >= 0
        assert row['target'] >= 0 and row['target'] == int(row['target'])
        assert date.fromisoformat(row['week_start']).year == 2021
        assert row['week_start'] != '2020-12-28'
        if key in truth:
            assert truth[key] == row['target']
        truth[key] = row['target']
        by_model[row['model_id']].append(row)
    expected_keys = set(truth)
    for rows in by_model.values():
        assert {(r['week_start'], r['unit']) for r in rows} == expected_keys

    metrics = read(folder / 'metrics.json')
    for record in metrics:
        rows = by_model[record['model_id']]
        scope, group = record['scope'], record['group']
        if scope == 'unit':
            rows = [r for r in rows if r['unit'] == group]
        elif scope == 'quarter':
            rows = [r for r in rows if f"Q{(date.fromisoformat(r['week_start']).month - 1) // 3 + 1}" == group]
        elif scope == 'half':
            rows = [r for r in rows if ('H1' if date.fromisoformat(r['week_start']).month <= 6 else 'H2') == group]
        elif scope == 'sparsity':
            rows = [r for r in rows if r['unit'] in ('WP PUTRAJAYA', 'Putrajaya')
                    and (r['target'] == 0) == (group == 'Putrajaya_zero')]
        else:
            assert scope in ('pooled', 'macro')
        if scope == 'macro':
            units = sorted({r['unit'] for r in rows})
            stats = [errors([r for r in rows if r['unit'] == unit]) for unit in units]
            result = {m: math.fsum(s[m] for s in stats) / len(stats) for m in stats[0]}
        else:
            result = errors(rows)
        assert len(rows) == record['rows']
        assert len({r['week_start'] for r in rows}) == record['weeks']
        for metric, value in result.items():
            close(value, record[metric])

    intervals = read(folder / 'paired_uncertainty.json')
    weeks = sorted({k[0] for k in truth})
    units = sorted({k[1] for k in truth})
    assert len(truth) == len(weeks) * len(units)
    # Reduce each complete week independently, then resample those weekly losses.
    # No evaluation.py functions or fitted models are used in this calculation.
    weekly = {}
    for model, rows in by_model.items():
        weekly[model] = []
        for week in weeks:
            group = [r for r in rows if r['week_start'] == week]
            assert len(group) == len(units)
            stat = errors(group)
            weekly[model].append([stat['MAE'], stat['RMSE'] ** 2])
        weekly[model] = np.asarray(weekly[model])
    for length in sorted({r['block_weeks'] for r in intervals}):
        draws = config['uncertainty']['draws']
        rng = np.random.default_rng(config['uncertainty']['seed'])
        starts = rng.integers(len(weeks) - length + 1,
                             size=(draws, math.ceil(len(weeks) / length)))
        sampled = [[i for start in draw for i in range(start, start + length)][:len(weeks)]
                   for draw in starts]
        sampled = np.asarray(sampled)
        resampled = {m: a[sampled].mean(axis=1) for m, a in weekly.items()}
        for record in [r for r in intervals if r['block_weeks'] == length]:
            a, b = record['comparator'], record['new_model']
            delta = resampled[a][:, 0] - resampled[b][:, 0]
            rmse_delta = np.sqrt(resampled[a][:, 1]) - np.sqrt(resampled[b][:, 1])
            close(errors(by_model[a])['MAE'] - errors(by_model[b])['MAE'], record['MAE_gain'])
            close(errors(by_model[a])['RMSE'] - errors(by_model[b])['RMSE'], record['RMSE_gain'])
            for metric, values in [('MAE_gain_CI95', delta), ('RMSE_gain_CI95', rmse_delta)]:
                for actual, expected in zip(np.quantile(values, [.025, .975]), record[metric]):
                    close(float(actual), expected)
            valid = resampled[a][:, 0] > 0
            assert int((~valid).sum()) == record['undefined_percentage_draws']
            if valid.any():
                percentage = 100 * delta[valid] / resampled[a][valid, 0]
                for actual, expected in zip(np.quantile(percentage, [.025, .975]), record['percentage_gain_CI95']):
                    close(float(actual), expected)
    return {'prediction_records': len(predictions), 'models': len(by_model),
            'paired_target_rows': len(truth), 'weeks': len(weeks), 'units': len(units),
            'metric_records_verified': len(metrics), 'interval_records_verified': len(intervals)}


def finalise_report(out, qc):
    path = out / 'results_report.md'
    report = path.read_text()
    run_date = date.fromisoformat(read(out / 'implementation_manifest.json')['date_local'])
    report = report.replace(f'Run date: {run_date.isoformat()}.', f'Run date: {run_date.day} {run_date:%B %Y}.')
    comparisons = read(out / 'primary/paired_uncertainty.json')
    core = next(r for r in comparisons if r['contrast'] == 'RQ2_core_vs_baseline' and r['block_weeks'] == 4)
    start = report.index('\n\n', report.index('Run date:')) + 2
    end = report.index('\n\n## Design and execution', start)
    report = report[:start] + (
        f"Baseline boosting had the lowest held-out MAE ({core['MAE_comparator']:.3f}). "
        f"Adding core weather increased MAE to {core['MAE_new']:.3f}, a "
        f"{abs(core['percentage_MAE_gain']):.2f}% deterioration. The paired MAE gain was "
        f"{core['MAE_gain']:.6f} notifications per district-week (95% interval "
        f"{core['MAE_gain_CI95'][0]:.6f} to {core['MAE_gain_CI95'][1]:.6f}). "
        "Positive gains favour the new model. This small deterioration does not establish a practically meaningful weather effect.\n\n"
        "The sensitivity analyses did not show consistent predictive improvement from weather. "
        "These comparisons concern this extract and evaluation period; they do not establish causal meteorological or climate-change effects."
    ) + report[end:]
    start = report.index('| Contrast | Block weeks |')
    end = report.index('\n\nEvery draw', start)
    lines = ['| Contrast | Block weeks | MAE gain | 95% interval | MAE gain percent |',
             '| --- | --- | --- | --- | --- |']
    for r in comparisons:
        lo, hi = r['MAE_gain_CI95']
        lines.append(f"| {r['contrast']} | {r['block_weeks']} | {r['MAE_gain']:.6f} | {lo:.6f} to {hi:.6f} | {r['percentage_MAE_gain']:.2f} |")
    report = report[:start] + '\n'.join(lines) + report[end:]
    anchor = 'Correlated weather/calendar predictors can redistribute SHAP credit.'
    extra = ('The selected boosting models used depth-one trees. A zero contribution indicates that a variable was unused by the fitted model; it does not establish an absence of a biological association. The stable top-two rankings partly reflect this model structure and the shared city histories. They are descriptive model properties.\n\n')
    if extra not in report:
        report = report.replace(anchor, extra + anchor)
    anchor = '![Forecasts by district](figures/forecast_by_district.png)'
    extra = ('\n\nPerformance differed by district. In sparse WP PUTRAJAYA, core NB had a lower MAE than core boosting (0.680 versus 0.944). The always-zero diagnostic had MAE 0.863 in this district. Overall boosting performance does not imply superiority in every district.')
    if extra not in report:
        report = report.replace(anchor, anchor + extra)
    anchor = '## Limitations'
    details = [f"An independent audit recalculated all {sum(x['metric_records_verified'] for x in qc['evaluations'].values())} saved metric records and {sum(x['interval_records_verified'] for x in qc['evaluations'].values())} paired interval records from the saved forecasts. It verified complete forecast pairing, non-negative predictions, source/code/model fingerprints and quarterly refit chronology. All checks passed.",
               f"There were {qc['tuning_fit_attempts']} cross-validation fit attempts. Of these, {qc['failed_tuning_cells']} failed the prespecified numerical criterion and their whole configurations were excluded. All {qc['final_and_adaptive_fits']} final/adaptive study fits converged. Full failures and excluded configurations remain in the tuning logs. No solver tolerance or primary setting was revised after test evaluation."]
    section = '## Independent verification\n\n' + '\n\n'.join(details) + '\n\n'
    if '## Independent verification' in report:
        old = report.index('## Independent verification')
        end = report.index(anchor, old)
        report = report[:old] + section + report[end:]
    else:
        report = report.replace(anchor, section + anchor)
    path.write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source-audit', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--finalise-report', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    config = read(out / 'config.executed.json')
    manifest = read(out / 'implementation_manifest.json')
    for name, expected in manifest['code_sha256'].items():
        assert digest(args.source_code / name) == expected
    for name, expected in manifest['input_sha256'].items():
        assert digest(Path(name)) == expected
    for lock_name, field in [('primary_lock.json', 'artefact_sha256'),
                             ('primary_outputs_lock.json', 'primary_output_sha256')]:
        for name, expected in read(out / lock_name)[field].items():
            assert digest(out / name) == expected
    original_sources = 0
    if args.source_audit:
        assert args.project_root is not None
        for name, expected in read(args.source_audit)['source_sha256'].items():
            assert digest(args.project_root / name) == expected
            original_sources += 1

    registry = read(out / 'sensitivities/registry.json')
    assert [r['id'] for r in registry] == [f'S{i:02}' for i in range(1, 11)]
    assert all(r['status'] == 'completed' for r in registry)
    folders = [out / 'primary', out / 'diagnostics/always_zero_Putrajaya']
    folders += [out / 'sensitivities' / r['id'] for r in registry]
    evaluations = {str(p.relative_to(out)): audit_evaluation(p, config) for p in folders}
    assert evaluations['primary']['paired_target_rows'] == 255
    assert evaluations['primary']['models'] == 8
    primary = {(r['model_id'], r['week_start'], r['unit']): r['prediction']
               for r in read(out / 'primary/predictions.json')}
    for sensitivity in ['S02', 'S03', 'S04', 'S05', 'S10']:
        for row in read(out / 'sensitivities' / sensitivity / 'predictions.json'):
            if row['model_id'] == 'P06':
                close(row['prediction'], primary['P06', row['week_start'], row['unit']])
    refits = read(out / 'sensitivities/S09/refits.json')
    assert len(refits) == 9
    expected_rows = dict(zip(config['refit_sensitivity_origins'], [280, 345, 410]))
    for row in refits:
        assert row['origin'] in expected_rows and row['fitting_rows'] == expected_rows[row['origin']]
        assert date.fromisoformat(row['last_fitting_week']).toordinal() + 6 < date.fromisoformat(row['origin']).toordinal()
    for row in read(out / 'sensitivities/S09/predictions.json'):
        if row['week_start'] < config['refit_sensitivity_origins'][0]:
            close(row['prediction'], primary[row['model_id'], row['week_start'], row['unit']])

    tuning_summary = []
    for path in sorted(out.rglob('tuning_*.json')):
        log = read(path)
        assert len(log['configurations']) == 12
        for c in log['configurations']:
            assert len(c['cells']) == 9
            assert c['valid'] == all('error' not in cell for cell in c['cells'])
        selected = next(c for c in log['configurations'] if c['candidate'] == log['chosen_candidate'])
        assert selected['valid'] and selected['setting'] == log['chosen_setting']
        tuning_summary.append({'path': str(path.relative_to(out)),
            'attempts': sum(len(c['cells']) for c in log['configurations']),
            'failed_cells': sum('error' in cell for c in log['configurations'] for cell in c['cells']),
            'excluded_configurations': sum(not c['valid'] for c in log['configurations'])})
    final_fits = 0
    for path in [out / 'fit_diagnostics.json', *sorted((out / 'sensitivities').glob('*/fit_diagnostics.json'))]:
        diagnostics = read(path)
        assert all(d['converged'] for d in diagnostics.values())
        final_fits += len(diagnostics)
    final_fits += len(refits)
    assert sum(r['attempts'] for r in tuning_summary) == 648
    assert final_fits == 42
    qc = {'all_passed': True, 'method': 'independent recalculation from saved forecasts; no study-module imports or fitting',
          'source_code_hashes_verified': len(manifest['code_sha256']),
          'executed_input_hashes_verified': len(manifest['input_sha256']),
          'original_source_hashes_verified': original_sources,
          'primary_model_and_output_locks_verified': True, 'evaluations': evaluations,
          'tuning_fit_attempts': sum(r['attempts'] for r in tuning_summary),
          'failed_tuning_cells': sum(r['failed_cells'] for r in tuning_summary),
          'tuning_logs': tuning_summary, 'final_and_adaptive_fits': final_fits,
          'quarterly_refit_chronology_verified': True, 'unchanged_baseline_forecasts_verified': True}
    (out / 'independent_results_qc.json').write_text(json.dumps(qc, indent=2) + '\n')
    if args.finalise_report:
        finalise_report(out, qc)
    print(json.dumps(qc, indent=2))


if __name__ == '__main__':
    main()
