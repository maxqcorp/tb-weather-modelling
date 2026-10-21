# TB weather notification forecasting

This code implements a retrospective one-week-ahead ecological forecasting study of recorded TB notifications in Kuala Lumpur and Putrajaya. It compares previous-week and four-week-mean benchmarks, ridge-regularised NB2 regression and shallow Poisson gradient boosting. Baseline, core-weather and expanded-weather sets share a chronological 2020 development period and 2021 evaluation period.

## What is included

The package contains Python source, the statistical configuration, an exact dependency lock and synthetic numerical/leakage checks. It contains no real health records, addresses, coordinates, actual input tables, fitted models, private predictions, study results or credentials. This repository distributes the experiment code and supporting reproducibility definitions.

## Environment and synthetic checks

Python 3.12 is required. The lock records the study's macOS ARM64 environment. Other operating systems may require additional platform-specific binary dependencies; recheck package imports and the numerical checks rather than assuming bitwise cross-platform equivalence.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python run_synthetic_checks.py
```

The launcher supplies a process-local OpenMP search path on macOS using the library vendored in the pinned scikit-learn wheel. It does not modify system libraries. Synthetic checks fit only generated data and can be run without access to the study dataset.

## Data availability

Actual inputs are deliberately excluded. Exact reproduction of the study estimates requires authorised access to the same aggregate inputs and source versions. Synthetic checks verify implementation properties but cannot reproduce empirical estimates from withheld data. Patient-level records must not be committed to this repository.

The data root must contain:

```text
climate/district_grid_2019_2021/district_weekly_tb_weather.csv
climate/district_grid_2019_2021/district_weather_proxies_daily.csv
climate/district_grid_2019_2021/district_proxy_grid_weights.csv
climate/kl_putrajaya_2019_2021/weekly_tb_weather_2020_2021.csv
climate/kl_putrajaya_2019_2021/daily_weather_all_sources.csv
climate/kl_putrajaya_2019_2021/malaysia_covid_stringency_2020_2021.csv
```

`DATA_DEFINITION.md` describes their roles and required columns. The primary outcome is the full notification extract count, not the smaller treatment-outcome-selected cleaned cohort. Primary weather uses fixed city ERA5 histories; four KL district outcomes share the same KL history. Existing patient-derived district footprints are exploratory only.

The primary train/evaluate stages use the six aggregate tables listed above. The full sensitivities stage additionally requires the confidential `climate/district_grid_2019_2021/patient_location_weather_linkage.csv` for its mandatory footprint-definition audit. Its required audit fields are `district`, `notification_date`, `era5_land_grid_id` and `era5_grid_id`. The separate accepted-location audit also uses `record_index` and the confidential `tb_cleaned_google_rechecked_v2.csv`; this latter audit is reported as unavailable if its file is absent. These private files are not distributed and must not be placed in the public repository. Access must be controlled by the data custodian. Without the linkage file, primary reproduction remains possible with the aggregate tables, while the full sensitivity stage cannot complete. No geocoding or paid API calls are needed for the experiment.

## Run stages

Use a new private output directory. The following paths are placeholders to be replaced locally.

```bash
.venv/bin/python run_study.py train --data-root /path/to/authorised/data --output /path/to/private/run --execute
.venv/bin/python run_study.py evaluate --data-root /path/to/authorised/data --output /path/to/private/run --execute
.venv/bin/python run_study.py sensitivities --data-root /path/to/authorised/data --output /path/to/private/run --execute
.venv/bin/python audit_run.py --output /path/to/private/run --finalise-report
```

The `all` stage combines development, evaluation and sensitivities. `--execute` is required for study fitting or scoring. The development stage runs synthetic checks, tunes only within 2020, fits the final models and creates a primary model/settings lock before any 2021 errors are computed. Evaluation verifies code/input/model fingerprints. It updates observed past-week predictors but never refits primary model parameters in 2021. The quarterly-refit sensitivity is explicitly separate.

The final audit command requires all ten sensitivity outputs. It independently recalculates saved errors and bootstrap intervals without fitting models or importing the modelling modules. It checks source/code/model fingerprints against the local run manifests, records excluded tuning configurations and increases report interval precision. Its local report contains actual results and must remain outside this code-only repository. Run manifests contain absolute local input paths; audit each run in its original data environment or preserve the documented path mapping. Regenerating the base report with the `report` stage requires rerunning the final audit to restore its verification section and presentation refinements.

Do not load model pickle files from untrusted sources. The code reloads only its own locally created private artefacts during a controlled run.

## Methodological safeguards

- Use Monday-to-Sunday targets. Remove the 28 December 2020 cross-year week as a target, while retaining it as completed history.
- Keep every district together in each temporal mask and each time-block bootstrap sample.
- Fit scaling and model parameters only within the fitting partition. Never tune on 2021.
- Share one selected setting per model family across the three feature sets.
- Use the specified NB2 likelihood and ridge penalty with explicit numerical controls, and verify gradient/Hessian consistency before fitting.
- Explain Poisson boosting on the raw log expected count margin. Keep the background strictly within training data and check additivity.
- Keep all registered sensitivity results and failures; do not replace the primary analysis with the most favourable sensitivity.

The study concerns predictive value in a short finalised extract. It does not establish infection risk, validated residential exposure, real-time operational accuracy or causal climate-change effects. Reporting completeness and the exact registry district definition require separate documentation.

## Module map

`data.py` constructs origin-restricted features; `models.py` fits and tunes models; `evaluation.py` calculates paired errors and block intervals; `interpretation.py` checks and summarises contributions; `engine.py` manages frozen forecasts and sensitivities; `cli.py` records staged execution and locks; `checks.py` provides data-free synthetic checks; `reporting.py` creates aggregate tables and figures.

The actual input data and model artefacts remain outside the public package. Review repository contents before publication and retain the data exclusions in `.gitignore`.
