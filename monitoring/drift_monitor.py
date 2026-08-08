import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASELINE_PATH = Path("data/baseline.csv")
NEW_DATA_PATH = Path("data/new_batch.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

HTML_REPORT_PATH = REPORT_DIR / "drift_report.html"
STATUS_PATH = REPORT_DIR / "drift_status.json"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

baseline = pd.read_csv(BASELINE_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

print("Baseline shape:", baseline.shape)
print("New batch shape:", new_data.shape)


# --------------------------------------------------
# 3. Generate Evidently drift report
# --------------------------------------------------

report = Report(
    metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ]
)

result = report.run(
    reference_data=baseline,
    current_data=new_data,
)

result.save_html(HTML_REPORT_PATH)

print("Evidently drift report generated successfully.")


# --------------------------------------------------
# 4. Calculate a simple drift decision
# --------------------------------------------------

# Only compare columns that exist in both datasets.
# SalePrice is excluded because new_batch.csv does not contain the target.
common_columns = [
    column
    for column in baseline.columns
    if column in new_data.columns and column != "SalePrice"
]

drifted_columns = []

for column in common_columns:
    # Numeric columns
    if pd.api.types.is_numeric_dtype(baseline[column]):
        baseline_mean = baseline[column].mean()
        new_mean = new_data[column].mean()

        baseline_std = baseline[column].std()

        # Avoid division by zero
        if pd.notna(baseline_std) and baseline_std > 0:
            standardized_shift = abs(new_mean - baseline_mean) / baseline_std

            # Consider the column drifted if the mean changed
            # by more than half a baseline standard deviation.
            if standardized_shift > 0.5:
                drifted_columns.append(column)

    # Categorical columns
    else:
        baseline_distribution = (
            baseline[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(normalize=True)
        )

        new_distribution = (
            new_data[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(normalize=True)
        )

        all_categories = baseline_distribution.index.union(
            new_distribution.index
        )

        baseline_distribution = baseline_distribution.reindex(
            all_categories,
            fill_value=0,
        )

        new_distribution = new_distribution.reindex(
            all_categories,
            fill_value=0,
        )

        total_difference = (
            baseline_distribution - new_distribution
        ).abs().sum() / 2

        # Simple categorical drift threshold
        if total_difference > 0.2:
            drifted_columns.append(column)


# --------------------------------------------------
# 5. Decide whether overall drift occurred
# --------------------------------------------------

total_columns = len(common_columns)
drifted_count = len(drifted_columns)

if total_columns > 0:
    drift_share = drifted_count / total_columns
else:
    drift_share = 0.0

# Trigger retraining if more than 30% of columns drift.
drift_detected = drift_share > 0.30


# --------------------------------------------------
# 6. Save drift status
# --------------------------------------------------

status = {
    "drift_detected": drift_detected,
    "drift_share": drift_share,
    "drifted_columns_count": drifted_count,
    "total_columns_checked": total_columns,
    "drifted_columns": drifted_columns,
}

with open(STATUS_PATH, "w") as file:
    json.dump(status, file, indent=4)


# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\nDrift monitoring complete.")
print("Columns checked:", total_columns)
print("Drifted columns:", drifted_count)
print("Drift share:", round(drift_share, 4))
print("Drift detected:", drift_detected)

print("\nHTML report saved to:")
print(HTML_REPORT_PATH)

print("\nDrift status saved to:")
print(STATUS_PATH)