# Monitoring a Financial Fraud Detection Model

A post-deployment monitoring workflow for a binary fraud model. It reports
precision, recall, average precision (PR-AUC), Brier score and calibration;
checks false-positive/false-negative concentration across channel, region and
customer segment; compares recent feature distributions with a reference
baseline; and can compare exported feature-importance and per-row attribution
snapshots.

The default run is a **synthetic demonstration**, not bank data or a trained
model:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python monitoring_fraud_model.py
```

The demo writes CSV summaries and PNG plots to `monitoring_output/`. Generated
additive score components in this demo are not SHAP values.

The run also plots RBI's published annual reported-fraud totals for FY 2023–24
to FY 2025–26 from [`data/india_rbi_reported_frauds.csv`](./data/india_rbi_reported_frauds.csv).
The latest reported total is **10,114 cases involving ₹48,021 crore** in FY
2025–26. This is separate national-level context, not production-model
transaction data.

### RBI aggregate-data scope

The source is [RBI Annual Report 2025–26, Chapter VI, Table VI.2](https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?Id=1466).
It counts frauds of ₹1 lakh and above reported during each year for the bank
groups listed in that table. It does **not** provide an all-India total that
includes co-operative banks or regional rural banks, so this series must not
be described as covering every Indian bank. Amount involved is not the loss
incurred, and cases reported in a year may have occurred earlier. RBI notes
that FY 2025–26 includes 314 cases involving ₹30,199 crore that were classified
afresh from earlier financial years. The source's dash for nil/negligible
amounts is represented as zero in the CSV.

## Production and baseline inputs

Pass both CSV files to disable the demo:

```bash
python monitoring_fraud_model.py \
  --production-csv production_predictions.csv \
  --baseline-csv baseline_features.csv \
  --output-dir monitoring_output
```

The production CSV must contain:

| Column | Meaning |
| --- | --- |
| `timestamp` | Prediction/event timestamp |
| `y_true` | Matured binary outcome (`0` = non-fraud, `1` = fraud) |
| `score` | Model fraud probability, in `[0, 1]` |
| `y_pred` | Optional binary decision; if omitted, `score >= --threshold` is used |

Optional `channel`, `region`, and `customer_segment` columns enable subgroup
tables. Other shared columns are treated as features. The baseline CSV should
contain a historical/reference sample with the same feature columns; it does
not need labels. Only overlapping features are tested for data drift.

For importance comparison, pass both CSV exports with `feature,importance`
columns using `--importance-baseline` and `--importance-current`. For
explainability comparison, pass both per-row attribution CSV exports using
`--attributions-baseline` and `--attributions-current`; numeric columns should
be feature attributions (for example, precomputed SHAP values). The script does
not load serialized model objects or calculate SHAP values.

Useful options include `--threshold`, `--drift-days`, `--min-volume`,
`--psi-threshold`, and `--output-dir`. PSI and KS statistics are diagnostic
signals, not proof of model harm. Small subgroup counts and immature labels
can make performance estimates unstable. Do not use this workflow alone to
change production thresholds or retrain a model.

## Outputs

- `weekly_metrics.csv`, `performance_trends.png`, `calibration.png`
- `slice_metrics.csv`
- `feature_drift.csv`, `feature_drift.png`
- `india_rbi_fraud_trend.png` (published annual context; separate from model metrics)
- Optional `importance_shifts.csv` / `importance_shifts.png`
- Optional `attribution_shifts.csv` / `attribution_shifts.png`
