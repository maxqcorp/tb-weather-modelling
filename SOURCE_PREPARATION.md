# Source preparation for the TB weather study

Algorithm 1 describes the full study workflow. The shared Python modelling pipeline begins after source preparation. It reads the prepared aggregate tables defined in `DATA_DEFINITION.md` and `INPUT_SCHEMA.json`. Raw-source date parsing, notification counting and calendar reindexing are not implemented by the released model modules.

## Notification counts

Use the authorised full notification extract and preserve its source version. Parse `Tarikh Notifikasi` as day-first notification dates and retain records within the study interval. Use recorded `Daerah` labels rather than reconstructing districts from residential coordinates. Include all recorded treatment outcomes in the primary count. Do not replace the full notification extract with the smaller outcome-selected cleaned cohort.

Count extract records by recorded district and Monday-to-Sunday notification week. Reindex every district to the same continuous weekly calendar. An absent group becomes zero records under this extract construction. This does not certify reporting completeness or unique TB episodes. Preserve partial boundary weeks in the source panel and exclude them as complete targets or complete historical weeks when constructing model features.

Retain the cross-year week beginning 28 December 2020 as completed history while excluding it as a fitting or scoring target. `full_study_week`, `week_start` and `week_end` describe calendar coverage. They do not establish operational data availability at historical forecast origins.

The prepared `target_notifications_excl_revised` column contains the separately constructed revised-diagnosis exclusion used for S06. The shared model code selects this prepared outcome column and rebuilds its count histories. It does not infer or recreate the source diagnosis classification. Reproduction requires the same authorised exclusion definition and source version.

## Weather and policy inputs

Prepare complete cached daily city weather and national policy tables with the column names and units in `INPUT_SCHEMA.json`. Preserve source identifiers, returned grid metadata and the original day conventions. Primary weather uses the fixed city ERA5 histories. NASA source sensitivity retains its supplied local-solar-day convention. The model code reads these caches without making weather API or geocoding calls.

The modelling code constructs all historical count, weather and policy predictors from completed periods before each forecast origin. It then applies the chronological development and test partitions. The common cohort depends on the complete prescribed history across the prepared tables.

## Sensitivity and source audits

The exploratory district-footprint sensitivity reads prepared daily district weather averages. The mandatory footprint-definition audit also reads the confidential notification-to-grid linkage to check membership and equal weights using 2020 records. The optional location-quality audit reads an additional confidential cleaned linkage file. These audit inputs are distinct from the aggregate tables used for primary model fitting and evaluation.

The independent audit can verify original source fingerprints only when the retained source-audit manifest and corresponding files are supplied. The full command and historical source-version requirements are documented in `README.md`.
