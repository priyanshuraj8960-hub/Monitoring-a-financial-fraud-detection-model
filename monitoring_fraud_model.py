"""Post-deployment monitoring for a binary financial-fraud model.

Run ``python monitoring_fraud_model.py`` for a synthetic, clearly labeled demo,
or pass production and baseline CSVs to analyze real monitoring exports.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss


RANDOM_SEED = 2104
RBI_FRAUD_DATA = Path(__file__).with_name("data") / "india_rbi_reported_frauds.csv"
REQUIRED_PRODUCTION_COLUMNS = {"timestamp", "y_true", "score"}
SLICE_COLUMNS = ("channel", "region", "customer_segment")
EXCLUDED_FEATURE_COLUMNS = {
    "timestamp",
    "y_true",
    "y_pred",
    "score",
    "fraud_score",
    *SLICE_COLUMNS,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def make_demo_data(
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic synthetic inputs and synthetic audit-export examples."""
    rng = np.random.default_rng(seed)
    feature_names = (
        "amount_gbp",
        "transactions_last_hour",
        "account_age_days",
        "days_since_password_change",
    )
    channels = np.array(["card_present", "online", "mobile"])
    regions = np.array(["London", "South East", "Midlands", "North"])
    segments = np.array(["retail", "small_business", "premium"])

    def make_features(size: int, current: bool) -> pd.DataFrame:
        amount = rng.lognormal(mean=4.0 if current else 3.45, sigma=0.8, size=size)
        velocity = rng.poisson(lam=2.2 if current else 1.2, size=size)
        account_age = rng.gamma(shape=2.2, scale=520, size=size).clip(1, 6000)
        password_age = rng.gamma(shape=2.0, scale=70, size=size).clip(0, 900)
        channel_probs = [0.28, 0.48, 0.24] if current else [0.42, 0.36, 0.22]
        frame = pd.DataFrame(
            {
                "amount_gbp": amount,
                "transactions_last_hour": velocity,
                "account_age_days": account_age,
                "days_since_password_change": password_age,
                "channel": rng.choice(channels, size=size, p=channel_probs),
                "region": rng.choice(regions, size=size),
                "customer_segment": rng.choice(segments, size=size, p=[0.7, 0.2, 0.1]),
            }
        )
        return frame

    baseline_features = make_features(6000, current=False)
    current_features = make_features(12000, current=True)
    end = pd.Timestamp.now(tz="UTC").floor("D")
    timestamps = end - pd.to_timedelta(rng.integers(0, 84 * 24 * 60, size=len(current_features)), unit="m")

    def logit_for(frame: pd.DataFrame, current: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        amount_term = 0.58 * (np.log1p(frame["amount_gbp"].to_numpy()) - 4.2)
        velocity_term = 0.58 * frame["transactions_last_hour"].to_numpy()
        account_term = -0.00022 * frame["account_age_days"].to_numpy()
        password_term = 0.0014 * frame["days_since_password_change"].to_numpy()
        online_term = (frame["channel"].to_numpy() == "online").astype(float) * 0.34
        intercept = -5.15
        if current:
            # The synthetic live process has both a population and a label-relationship shift.
            velocity_term = 0.78 * frame["transactions_last_hour"].to_numpy()
            intercept = -4.55
        parts = {
            "amount_gbp": amount_term,
            "transactions_last_hour": velocity_term,
            "account_age_days": account_term,
            "days_since_password_change": password_term,
            "channel_online": online_term,
        }
        return intercept + sum(parts.values()), parts

    logits, current_parts = logit_for(current_features, current=True)
    probabilities = _sigmoid(logits)
    labels = rng.binomial(1, probabilities)
    # Scores imitate a model whose thresholded outputs are imperfect and a little stale.
    scores = _sigmoid(logits * 0.82 + rng.normal(0, 0.38, size=len(logits)))
    production = current_features.copy()
    production.insert(0, "timestamp", timestamps)
    production["y_true"] = labels
    production["score"] = scores
    production["y_pred"] = (scores >= 0.5).astype(int)

    baseline_logits, baseline_parts = logit_for(baseline_features, current=False)
    baseline_importance = pd.DataFrame(
        {
            "feature": list(baseline_parts),
            "importance": [float(np.mean(np.abs(values))) for values in baseline_parts.values()],
        }
    )
    current_importance = pd.DataFrame(
        {
            "feature": list(current_parts),
            "importance": [float(np.mean(np.abs(values))) for values in current_parts.values()],
        }
    )
    # These are toy additive contributions for demonstrating the comparison workflow;
    # they are not generated by SHAP or a trained model.
    baseline_explanations = pd.DataFrame(baseline_parts)
    current_explanations = pd.DataFrame(current_parts)
    return (
        production,
        baseline_features,
        baseline_importance,
        current_importance,
        baseline_explanations,
        current_explanations,
    )


def load_production(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_PRODUCTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path} contains invalid or missing timestamps.")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(None)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame[["y_true", "score"]].isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric y_true/score values.")
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if not frame["score"].between(0, 1).all():
        raise ValueError("score must contain probabilities in the range [0, 1].")
    if "y_pred" in frame:
        frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
        if frame["y_pred"].isna().any() or not frame["y_pred"].isin([0, 1]).all():
            raise ValueError("y_pred must contain only binary values 0 and 1.")
    else:
        frame["y_pred"] = (frame["score"] >= threshold).astype(int)
    return frame


def metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["y_true"].astype(int).to_numpy()
    predicted = frame["y_pred"].astype(int).to_numpy()
    scores = frame["score"].astype(float).to_numpy()
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    positives = tp + fn
    negatives = fp + tn
    return {
        "n": len(frame),
        "fraud_rate": float(np.mean(truth)) if len(truth) else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / positives if positives else np.nan,
        "pr_auc": average_precision_score(truth, scores) if positives else np.nan,
        "brier_score": brier_score_loss(truth, scores) if len(truth) else np.nan,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / negatives if negatives else np.nan,
        "false_negative_rate": fn / positives if positives else np.nan,
    }


def time_metrics(production: pd.DataFrame) -> pd.DataFrame:
    frame = production.copy()
    frame["window"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None).dt.to_period("W").astype(str)
    return pd.DataFrame(
        [{"window": window, **metric_row(group)} for window, group in frame.groupby("window", sort=True)]
    )


def plot_metric_trends(metrics: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    metric_names = ("precision", "recall", "pr_auc", "brier_score")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, metric_names):
        axis.plot(metrics["window"], metrics[name], marker="o", linewidth=1.6)
        if name in {"precision", "recall", "pr_auc"}:
            axis.set_ylim(0, 1)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Week")
        axis.tick_params(axis="x", labelrotation=60)
        axis.grid(alpha=0.25)
    figure.suptitle(f"Weekly model performance (classification threshold = {threshold:.2f})")
    figure.savefig(output_dir / "performance_trends.png", dpi=150)
    plt.close(figure)


def plot_calibration(production: pd.DataFrame, output_dir: Path) -> None:
    truth = production["y_true"].astype(int).to_numpy()
    scores = production["score"].astype(float).to_numpy()
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    if len(np.unique(truth)) > 1:
        observed, predicted = calibration_curve(truth, scores, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label="Current scores")
    else:
        axis.text(0.5, 0.5, "Calibration curve unavailable:\nonly one outcome class present", ha="center")
    axis.set(xlabel="Mean predicted fraud probability", ylabel="Observed fraud rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title(f"Calibration (Brier score: {brier_score_loss(truth, scores):.4f})")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(figure)


def plot_rbi_annual_context(data_path: Path, output_dir: Path) -> pd.DataFrame:
    required = {"financial_year", "bank_group", "reported_cases", "amount_inr_crore"}
    frame = pd.read_csv(data_path)
    if not required.issubset(frame.columns):
        raise ValueError(f"{data_path} must contain: {', '.join(sorted(required))}")
    totals = frame.loc[frame["bank_group"] == "Total"].copy()
    if totals.empty or totals["financial_year"].duplicated().any():
        raise ValueError(f"{data_path} must have exactly one Total row per financial year.")
    for column in ("reported_cases", "amount_inr_crore"):
        totals[column] = pd.to_numeric(totals[column], errors="coerce")
        if totals[column].isna().any() or (totals[column] < 0).any():
            raise ValueError(f"{data_path} contains invalid {column} totals.")
    totals = totals.sort_values("financial_year")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(totals["financial_year"], totals["reported_cases"], marker="o", color="#4472C4")
    axes[0].set(title="Reported fraud cases", ylabel="Cases")
    axes[1].plot(totals["financial_year"], totals["amount_inr_crore"], marker="o", color="#ED7D31")
    axes[1].set(title="Amount involved", ylabel="₹ crore")
    for axis in axes:
        axis.set_xlabel("Financial year")
        axis.grid(alpha=0.25)
    figure.suptitle("India: RBI-reported frauds (Table VI.2 coverage)")
    figure.text(
        0.5,
        -0.035,
        "₹1 lakh+ cases reported in each year; not necessarily occurred in that year. "
        "Amount involved is not the loss incurred.",
        ha="center",
        fontsize=8,
    )
    figure.savefig(output_dir / "india_rbi_fraud_trend.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return totals


def slice_metrics(production: pd.DataFrame, min_volume: int) -> pd.DataFrame:
    rows = []
    for column in SLICE_COLUMNS:
        if column not in production:
            continue
        for value, group in production.groupby(column, dropna=False, sort=True):
            row = {"slice": column, "value": str(value), **metric_row(group)}
            row["below_minimum_volume"] = len(group) < min_volume
            rows.append(row)
    return pd.DataFrame(rows)


def _psi(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    epsilon = 1e-6
    reference = (reference_counts.astype(float) + epsilon)
    current = (current_counts.astype(float) + epsilon)
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def drift_table(
    production: pd.DataFrame,
    baseline: pd.DataFrame,
    drift_days: int,
) -> pd.DataFrame:
    cutoff = production["timestamp"].max() - pd.Timedelta(days=drift_days)
    current = production.loc[production["timestamp"] >= cutoff]
    feature_names = [
        column
        for column in baseline.columns
        if column in current.columns and column not in EXCLUDED_FEATURE_COLUMNS
    ]
    if not feature_names:
        raise ValueError("No feature columns overlap between baseline and recent production data.")
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        baseline_values = baseline[feature]
        current_values = current[feature]
        numeric = pd.api.types.is_numeric_dtype(baseline_values) and pd.api.types.is_numeric_dtype(current_values)
        if numeric:
            reference = pd.to_numeric(baseline_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            observed = pd.to_numeric(current_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            reference_valid = reference.dropna().to_numpy(dtype=float)
            current_valid = observed.dropna().to_numpy(dtype=float)
            if not len(reference_valid) or not len(current_valid):
                psi = ks_stat = ks_pvalue = np.nan
            else:
                quantiles = np.unique(np.quantile(reference_valid, np.linspace(0, 1, 11)))
                if len(quantiles) <= 2:
                    edges = np.array([-np.inf, np.inf])
                else:
                    edges = np.r_[-np.inf, quantiles[1:-1], np.inf]
                reference_hist = np.histogram(reference_valid, bins=edges)[0]
                current_hist = np.histogram(current_valid, bins=edges)[0]
                reference_counts = np.r_[reference_hist, reference.isna().sum()]
                current_counts = np.r_[current_hist, observed.isna().sum()]
                psi = _psi(reference_counts, current_counts)
                ks = ks_2samp(reference_valid, current_valid, alternative="two-sided", method="auto")
                ks_stat, ks_pvalue = float(ks.statistic), float(ks.pvalue)
            rows.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "baseline_missing_rate": float(reference.isna().mean()),
                    "current_missing_rate": float(observed.isna().mean()),
                }
            )
        else:
            reference = baseline_values.astype("string").fillna("<MISSING>")
            observed = current_values.astype("string").fillna("<MISSING>")
            categories = reference.unique().tolist() + [
                value for value in observed.unique().tolist() if value not in set(reference.unique())
            ]
            reference_counts = reference.value_counts().reindex(categories, fill_value=0).to_numpy()
            current_counts = observed.value_counts().reindex(categories, fill_value=0).to_numpy()
            rows.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "psi": _psi(reference_counts, current_counts),
                    "ks_statistic": np.nan,
                    "ks_pvalue": np.nan,
                    "baseline_missing_rate": float(baseline_values.isna().mean()),
                    "current_missing_rate": float(current_values.isna().mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def plot_drift(drift: pd.DataFrame, output_dir: Path, psi_threshold: float) -> None:
    top = drift.head(15).sort_values("psi")
    figure, axis = plt.subplots(figsize=(9, max(4, len(top) * 0.38)), constrained_layout=True)
    axis.barh(top["feature"], top["psi"], color="#4472C4")
    axis.axvline(psi_threshold, color="#C00000", linestyle="--", label=f"Review trigger ({psi_threshold:g})")
    axis.set(xlabel="Population Stability Index (PSI)", title="Largest feature distribution changes")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_dir / "feature_drift.png", dpi=150)
    plt.close(figure)


def load_importance(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if not {"feature", "importance"}.issubset(frame.columns):
        raise ValueError(f"{label} importance CSV must have 'feature' and 'importance' columns.")
    frame = frame[["feature", "importance"]].copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    if frame["importance"].isna().any():
        raise ValueError(f"{label} importance values must be numeric.")
    return frame


def compare_importance(
    baseline: pd.DataFrame | None,
    current: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    if baseline is None or current is None:
        print("Feature-importance comparison skipped; provide both importance CSV exports to enable it.")
        return
    merged = baseline.merge(current, on="feature", how="outer", suffixes=("_baseline", "_current")).fillna(0)
    merged["importance_shift"] = merged["importance_current"] - merged["importance_baseline"]
    merged["absolute_shift"] = merged["importance_shift"].abs()
    merged = merged.sort_values("absolute_shift", ascending=False)
    merged.to_csv(output_dir / "importance_shifts.csv", index=False)
    top = merged.head(15).sort_values("importance_shift")
    figure, axis = plt.subplots(figsize=(9, max(4, len(top) * 0.38)), constrained_layout=True)
    axis.barh(top["feature"], top["importance_shift"], color=np.where(top["importance_shift"] >= 0, "#70AD47", "#ED7D31"))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Current minus baseline importance", title="Feature-importance shifts")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_dir / "importance_shifts.png", dpi=150)
    plt.close(figure)
    print("\nLargest importance shifts:")
    print(merged.head(10)[["feature", "importance_baseline", "importance_current", "importance_shift"]].to_string(index=False))


def load_attributions(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.empty or frame.select_dtypes(include="number").empty:
        raise ValueError(f"{label} attribution CSV must contain per-row numeric feature-attribution columns.")
    return frame.select_dtypes(include="number")


def compare_attributions(
    baseline: pd.DataFrame | None,
    current: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    if baseline is None or current is None:
        print("Attribution comparison skipped; provide both per-row attribution CSVs to enable it.")
        return
    baseline_mean = baseline.abs().mean()
    current_mean = current.abs().mean()
    merged = pd.concat(
        [baseline_mean.rename("baseline_mean_abs"), current_mean.rename("current_mean_abs")], axis=1
    ).fillna(0)
    merged["absolute_shift"] = merged["current_mean_abs"] - merged["baseline_mean_abs"]
    merged["magnitude_of_shift"] = merged["absolute_shift"].abs()
    merged = merged.sort_values("magnitude_of_shift", ascending=False)
    merged.to_csv(output_dir / "attribution_shifts.csv", index_label="feature")
    top = merged.head(15).sort_values("absolute_shift")
    figure, axis = plt.subplots(figsize=(9, max(4, len(top) * 0.38)), constrained_layout=True)
    axis.barh(top.index, top["absolute_shift"], color=np.where(top["absolute_shift"] >= 0, "#70AD47", "#ED7D31"))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Change in mean absolute attribution", title="Per-feature attribution shifts")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_dir / "attribution_shifts.png", dpi=150)
    plt.close(figure)
    print("\nLargest attribution shifts:")
    print(merged.head(10)[["baseline_mean_abs", "current_mean_abs", "absolute_shift"]].to_string())


def summarize_concept_drift(metrics: pd.DataFrame, production: pd.DataFrame, min_volume: int) -> None:
    latest = metrics.tail(4)
    print("\nConcept/performance drift (outcomes and model quality, not just feature distributions):")
    print(
        latest[["window", "n", "fraud_rate", "precision", "recall", "pr_auc", "brier_score"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    if len(production) < min_volume:
        warnings.warn("Overall production sample is below the configured minimum-volume review threshold.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-csv", type=Path, help="Predictions/outcomes CSV; omit for synthetic demo.")
    parser.add_argument("--baseline-csv", type=Path, help="Historical feature-reference CSV.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold used when y_pred is absent.")
    parser.add_argument("--drift-days", type=int, default=28, help="Recent window length for feature-drift checks.")
    parser.add_argument("--min-volume", type=int, default=100, help="Flag slices smaller than this sample count.")
    parser.add_argument("--psi-threshold", type=float, default=0.2, help="PSI review trigger, not a significance test.")
    parser.add_argument("--importance-baseline", type=Path, help="Baseline feature importance CSV.")
    parser.add_argument("--importance-current", type=Path, help="Current feature importance CSV.")
    parser.add_argument("--attributions-baseline", type=Path, help="Baseline per-row SHAP attribution CSV.")
    parser.add_argument("--attributions-current", type=Path, help="Current per-row SHAP attribution CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("monitoring_output"), help="Folder for reports and plots.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.drift_days <= 0 or args.min_volume <= 0:
        parser.error("--drift-days and --min-volume must be positive.")
    if args.psi_threshold < 0:
        parser.error("--psi-threshold must be non-negative.")
    if bool(args.production_csv) != bool(args.baseline_csv):
        parser.error("--production-csv and --baseline-csv must be provided together.")
    return args


def main() -> None:
    args = parse_args()
    demo_mode = args.production_csv is None
    if demo_mode:
        (
            production,
            baseline_features,
            demo_importance_baseline,
            demo_importance_current,
            demo_attributions_baseline,
            demo_attributions_current,
        ) = make_demo_data()
        importance_baseline = demo_importance_baseline
        importance_current = demo_importance_current
        attributions_baseline = demo_attributions_baseline
        attributions_current = demo_attributions_current
        print("DEMO ONLY: all records, drift, model outputs, importance and additive attributions are synthetic.")
    else:
        production = load_production(args.production_csv, args.threshold)
        baseline_features = pd.read_csv(args.baseline_csv)
        importance_baseline = load_importance(args.importance_baseline, "Baseline")
        importance_current = load_importance(args.importance_current, "Current")
        attributions_baseline = load_attributions(args.attributions_baseline, "Baseline")
        attributions_current = load_attributions(args.attributions_current, "Current")
    if production.empty or baseline_features.empty:
        raise ValueError("Production and baseline inputs must each contain at least one row.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rbi_totals = plot_rbi_annual_context(RBI_FRAUD_DATA, args.output_dir)
    latest_rbi = rbi_totals.iloc[-1]
    print(
        "\nIndia RBI-reported annual context "
        f"({latest_rbi['financial_year']}): {int(latest_rbi['reported_cases']):,} cases; "
        f"₹{latest_rbi['amount_inr_crore']:,.0f} crore involved."
    )
    print(
        "RBI Table VI.2 coverage does not include co-operative banks or regional rural banks; "
        "see README for scope and reporting caveats."
    )
    metrics = time_metrics(production)
    metrics.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    plot_metric_trends(metrics, args.output_dir, args.threshold)
    plot_calibration(production, args.output_dir)

    slices = slice_metrics(production, args.min_volume)
    slices.to_csv(args.output_dir / "slice_metrics.csv", index=False)
    print("\nSlice performance (small groups are flagged; treat rates cautiously):")
    if slices.empty:
        print("No channel, region, or customer_segment columns were present.")
    else:
        display_columns = [
            "slice",
            "value",
            "n",
            "precision",
            "recall",
            "false_positives",
            "false_negatives",
            "false_positive_rate",
            "false_negative_rate",
            "below_minimum_volume",
        ]
        print(slices[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    drift = drift_table(production, baseline_features, args.drift_days)
    drift.to_csv(args.output_dir / "feature_drift.csv", index=False)
    plot_drift(drift, args.output_dir, args.psi_threshold)
    flagged = drift.loc[drift["psi"] >= args.psi_threshold]
    print(
        f"\nFeature drift: {len(flagged)} of {len(drift)} features meet PSI review trigger "
        f"{args.psi_threshold:g}. A trigger is a prompt to investigate, not proof of harmful drift."
    )
    print(drift.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    summarize_concept_drift(metrics, production, args.min_volume)

    compare_importance(importance_baseline, importance_current, args.output_dir)
    if demo_mode:
        print("\nDemo attribution note: these are synthetic additive score components, not SHAP values.")
    compare_attributions(attributions_baseline, attributions_current, args.output_dir)
    print(f"\nOutputs written to: {args.output_dir.resolve()}")
    print(
        "\nReview checklist: validate label maturity and leakage; investigate sustained quality changes; "
        "check subgroup sample sizes and error costs; confirm data-pipeline changes; and escalate before "
        "changing thresholds or retraining."
    )


if __name__ == "__main__":
    main()
