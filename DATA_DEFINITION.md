# Dataset definition for code reproduction

The analysis unit is a recorded reporting district and a Monday-to-Sunday week. The target is the number of extract records notified during that week. The five units are CHERAS, KEPONG, LEMBAH PANTAI, TITIWANGSA and WP PUTRAJAYA. Daerah is treated as a recorded reporting label, not a verified residential boundary.

Source preparation precedes the shared model code. Notification dates are parsed, records are counted and continuous weekly calendars are created before the input tables below are read. The raw-source preparation requirements are described in `SOURCE_PREPARATION.md`.

The original full extract is not required by the primary model code and is not included. The local aggregate weekly panel supplies `district`, `week_start`, `week_end`, `target_notifications`, `target_notifications_excl_revised` and `full_study_week`. Complete continuous weekly slots, including the cross-year history, must be preserved. Partial study-boundary weeks cannot supply complete historical counts.

The daily fixed-city table supplies `city`, `date` and provider-prefixed temperature, humidity, rainfall, wind, radiation and daily temperature-range columns. The policy table supplies `date` and `covid_stringency_index`. The old footprint daily table supplies `district`, `date` and corresponding unprefixed weather columns; it is used only in its exploratory sensitivity. The footprint weights table records `district`, `source`, `grid_id`, `weight` and `reference_year`.

`INPUT_SCHEMA.json` lists exact required aggregate column names and uniqueness keys. The city weekly table is retained as an input-version reference; city sensitivity outcomes are reconstructed by summing the district counts. Dates are parsed as ISO calendar dates in files. Fixed-city ERA5 and ERA5-Land histories use the study's local-day aggregation; the NASA alternative retains its supplied day convention. This difference is part of the source sensitivity. Missing numeric values are not imputed. The outcome counts are extract records and are not certified unique TB episodes or population incidence rates.

Primary baseline features are count lags 1/2/4/8, a previous four-week count mean, annual sine/cosine, week index and preceding-week policy stringency. The annual terms are sin(2π × day of year / 365.25) and cos(2π × day of year / 365.25), using the forecast Monday's day of year. Core weather adds prior 28-day mean temperature (degrees Celsius), mean relative humidity (percent) and total rainfall (millimetres). Expanded weather adds mean wind (metres per second), total solar radiation (megajoules per square metre) and mean daily temperature range (degrees Celsius). Means and totals are reconstructed from complete prior daily windows.

The common lag-complete primary cohort has 475 rows across 95 target weeks. The strict split has 215 rows in 43 training weeks wholly in 2020, 255 rows in 51 test weeks wholly in 2021 and five cross-year rows excluded as targets. These are reference cohort specifications rather than a released dataset. Input files must match the original authorised cohort; results from substitute or synthetic data must be labelled accordingly.

Input paths and feature names are definitions, not evidence of public data access rights. The package contains no real patient or aggregate study input values.

## Additional confidential audit inputs

The complete sensitivity stage requires `climate/district_grid_2019_2021/patient_location_weather_linkage.csv` for the mandatory footprint-definition audit. Its required fields are `district`, `notification_date`, `era5_land_grid_id` and `era5_grid_id`. These are record-level audit inputs and are not part of the released aggregate schema.

The separate location-quality audit additionally uses `record_index` in that linkage file and `source_record_index`, `latitude`, `longitude`, `AG_AH_verification_status` and `coordinate_precision` from the confidential `tb_cleaned_google_rechecked_v2.csv`. This latter audit is recorded as unavailable if the optional cleaned file is absent. These private files are not distributed. Primary fitting and evaluation use the aggregate inputs, while the complete sensitivity and audit workflow requires the additional controlled inputs. The full original-source audit also requires the retained source-audit manifest and its corresponding source files, as described in `README.md`.
