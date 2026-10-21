# Dataset definition for code reproduction

The analysis unit is a recorded reporting district and a Monday-to-Sunday week. The target is the number of extract records notified during that week. The five units are CHERAS, KEPONG, LEMBAH PANTAI, TITIWANGSA and WP PUTRAJAYA. Daerah is treated as a recorded reporting label, not a verified residential boundary.

The original full extract is not required by the primary model code and is not included. The local aggregate weekly panel supplies `district`, `week_start`, `week_end`, `target_notifications`, `target_notifications_excl_revised` and `full_study_week`. Complete continuous weekly slots, including the cross-year history, must be preserved. Partial study-boundary weeks cannot supply complete historical counts.

The daily fixed-city table supplies `city`, `date` and provider-prefixed temperature, humidity, rainfall, wind, radiation and daily temperature-range columns. The policy table supplies `date` and `covid_stringency_index`. The old footprint daily table supplies `district`, `date` and corresponding unprefixed weather columns; it is used only in its exploratory sensitivity. The footprint weights table records `district`, `source`, `grid_id`, `weight` and `reference_year`.

`INPUT_SCHEMA.json` lists exact required aggregate column names and uniqueness keys. The city weekly table is retained as an input-version reference; city sensitivity outcomes are reconstructed by summing the district counts. Dates are parsed as ISO calendar dates in files. Fixed-city ERA5 and ERA5-Land histories use the study's local-day aggregation; the NASA alternative retains its supplied day convention. This difference is part of the source sensitivity. Missing numeric values are not imputed. The outcome counts are extract records and are not certified unique TB episodes or population incidence rates.

Primary baseline features are count lags 1/2/4/8, a previous four-week count mean, annual sine/cosine, week index and preceding-week policy stringency. Core weather adds prior 28-day mean temperature (degrees Celsius), mean relative humidity (percent) and total rainfall (millimetres). Expanded weather adds mean wind (metres per second), total solar radiation (megajoules per square metre) and mean daily temperature range (degrees Celsius). Means and totals are reconstructed from complete prior daily windows.

The common lag-complete primary cohort has 475 rows across 95 target weeks. The strict split has 215 rows in 43 training weeks wholly in 2020, 255 rows in 51 test weeks wholly in 2021 and five cross-year rows excluded as targets. These are reference cohort specifications rather than a released dataset. Input files must match the original authorised cohort; results from substitute or synthetic data must be labelled accordingly.

Input paths and feature names are definitions, not evidence of public data access rights. The package contains no real patient or aggregate study input values.
