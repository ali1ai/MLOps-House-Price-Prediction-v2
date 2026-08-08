import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from pathlib import Path

# -----------------------------
# 1. Load baseline + new data
# -----------------------------
BASELINE_PATH = Path("data/baseline.csv")
NEW_DATA_PATH = Path("data/new_batch.csv")

baseline = pd.read_csv(BASELINE_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

# -----------------------------
# 2. Build Evidently report
# -----------------------------
report = Report(
    metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),
    ]
)

report.run(reference_data=baseline, current_data=new_data)

# -----------------------------
# 3. Save HTML + JSON reports
# -----------------------------
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

report.save_html(REPORT_DIR / "drift_report.html")
report.save_json(REPORT_DIR / "drift_report.json")

print("Drift report generated successfully.")
