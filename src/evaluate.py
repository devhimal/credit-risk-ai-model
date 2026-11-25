"""Evaluation metrics, business-cost thresholding, plots, and error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.features import add_engineered_features, feature_names_from_pipeline

PALETTE = {"negative": "#4C78A8", "positive": "#E45756", "accent": "#54A24B"}
sns.set_theme(style="whitegrid", context="notebook")


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray | None = None,
) -> dict:
    """Compute discrimination, calibration, and confusion-matrix metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flag_rate": float(y_pred.mean()),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_true, proba)),
                "pr_auc": float(average_precision_score(y_true, proba)),
                "brier_score": float(brier_score_loss(y_true, proba)),
            }
        )
    return metrics


def expected_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
) -> dict:
    """Cost of a binary decision using relative FN/FP units."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = cost_fn * fn + cost_fp * fp
    return {
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "total_cost_units": float(total),
        "cost_per_account": float(total / len(y_true)),
        "cost_per_1000_accounts": float(1000 * total / len(y_true)),
    }


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
    capacity_fraction: float = config.REVIEW_CAPACITY_FRACTION,
    thresholds: list[float] | None = None,
) -> dict:
    """Scan probability thresholds for cost, recall, and operational load.

    The selected threshold minimises expected cost *subject to* the share of
    flagged accounts not exceeding the manual-review capacity. This makes the
    business assumption explicit and easy to sensitivity-test.
    """
    thresholds = thresholds or config.THRESHOLD_GRID
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = cost_fn * fn + cost_fp * fp
        rows.append(
            {
                "threshold": float(threshold),
                "flag_rate": float(y_pred.mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "cost_per_account": float(total_cost / len(y_true)),
                "cost_per_1000_accounts": float(1000 * total_cost / len(y_true)),
                "within_capacity": bool(y_pred.mean() <= capacity_fraction),
            }
        )

    unconstrained = min(
        rows, key=lambda row: (row["cost_per_account"], -row["recall"])
    )
    eligible = [row for row in rows if row["within_capacity"]]
    constrained = min(
        eligible, key=lambda row: (row["cost_per_account"], -row["recall"])
    )
    return {
        "assumptions": {
            "cost_false_negative": cost_fn,
            "cost_false_positive": cost_fp,
            "review_capacity_fraction": capacity_fraction,
        },
        "best_unconstrained": unconstrained,
        "best_within_capacity": constrained,
        "selected": constrained,
        "table": rows,
    }


def evaluate_pipeline(
    pipeline, X: pd.DataFrame, y: pd.Series, threshold: float
) -> dict:
    """Evaluate a fitted pipeline at a fixed threshold."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = compute_metrics(y, pred, proba)
    metrics["threshold"] = float(threshold)
    metrics["cost"] = expected_cost(y, pred)
    return {"metrics": metrics, "probabilities": proba, "predictions": pred}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_eda_figures(df: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    """Create the EDA figures shown in the notebook and dashboard."""
    output_dir = output_dir or config.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_engineered_features(df)
    target = config.TARGET
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = frame[target].value_counts().sort_index()
    bars = ax.bar(
        ["No default (0)", "Default (1)"],
        counts.values,
        color=[PALETTE["negative"], PALETTE["positive"]],
    )
    ax.bar_label(bars, fmt="{:,.0f}")
    ax.set_title("Class distribution (imbalanced target)")
    ax.set_ylabel("Accounts")
    paths.append(_save(fig, output_dir / "eda_class_distribution.png"))

    corr = frame[[*config.BASE_NUMERIC_FEATURES, target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=False,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation between raw numeric features and target")
    paths.append(_save(fig, output_dir / "eda_correlation_heatmap.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=frame,
        x="age",
        hue=target,
        fill=True,
        common_norm=False,
        palette={0: PALETTE["negative"], 1: PALETTE["positive"]},
        ax=ax,
    )
    ax.set_title("Age distribution by repayment outcome")
    ax.set_xlabel("Age (years)")
    paths.append(_save(fig, output_dir / "eda_age_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_frame = frame.assign(log_limit=np.log1p(frame["limit_bal"].clip(lower=0)))
    sns.boxplot(
        data=plot_frame,
        x=target,
        y="log_limit",
        hue=target,
        palette={0: PALETTE["negative"], 1: PALETTE["positive"]},
        legend=False,
        ax=ax,
    )
    ax.set_xticks([0, 1], ["No default", "Default"])
    ax.set_ylabel("log(1 + credit limit)")
    ax.set_xlabel("")
    ax.set_title("Credit limit by repayment outcome")
    paths.append(_save(fig, output_dir / "eda_limit_balance.png"))

    delinquency = (
        frame.groupby("num_delinquent_months")[target]
        .agg(["mean", "size"])
        .reset_index()
        .query("size >= 50")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        delinquency["num_delinquent_months"],
        delinquency["mean"],
        marker="o",
        color=PALETTE["positive"],
    )
    ax.set_xlabel("Months with payment delay > 0 (of 6)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Default rate rises sharply with recent delinquency")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    paths.append(_save(fig, output_dir / "eda_delinquency_default_rate.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=frame.assign(
            utilization=frame["utilization_ratio"].clip(-1, 3)
        ),
        x="utilization",
        hue=target,
        fill=True,
        common_norm=False,
        palette={0: PALETTE["negative"], 1: PALETTE["positive"]},
        ax=ax,
    )
    ax.set_title("Average utilisation ratio by outcome")
    ax.set_xlabel("Average bill / (credit limit + 1)")
    paths.append(_save(fig, output_dir / "eda_utilization.png"))

    return paths


def feature_importance(pipeline) -> pd.Series:
    """Extract coefficients or tree importances from a fitted pipeline."""
    names = feature_names_from_pipeline(pipeline)
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_).ravel()
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_).ravel()
    else:  # pragma: no cover - other model families
        return pd.Series(dtype=float)
    if len(names) != len(values):
        names = [f"feature_{i}" for i in range(len(values))]
    series = pd.Series(values, index=names, name="importance")
    return series.reindex(series.abs().sort_values(ascending=False).index)


def plot_feature_importance(pipeline, model_name: str, path: Path) -> Path:
    series = feature_importance(pipeline).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [
        PALETTE["positive"] if value > 0 else PALETTE["negative"]
        for value in series.values
    ]
    ax.barh(series.index, series.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top 20 features - {model_name}")
    ax.set_xlabel("Coefficient" if "logistic" in model_name.lower() else "Importance")
    return _save(fig, path)


def plot_model_diagnostics(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """ROC, PR, confusion matrix, and threshold-cost figures for final model."""
    output_dir = output_dir or config.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    proba_val = pipeline.predict_proba(X_val)[:, 1]
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    paths: dict[str, Path] = {}

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        fpr,
        tpr,
        color=PALETTE["negative"],
        lw=2,
        label=f"ROC-AUC = {roc_auc_score(y_test, proba_test):.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve - {model_name} (test)")
    ax.legend(loc="lower right")
    paths["roc"] = _save(fig, output_dir / "model_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        recall,
        precision,
        color=PALETTE["positive"],
        lw=2,
        label=f"PR-AUC = {average_precision_score(y_test, proba_test):.3f}",
    )
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve - {model_name} (test)")
    ax.legend(loc="upper right")
    paths["pr"] = _save(fig, output_dir / "model_pr_curve.png")

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No flag", "Flag"],
        yticklabels=["No default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f} (test)")
    paths["confusion_matrix"] = _save(fig, output_dir / "model_confusion_matrix.png")

    analysis = threshold_analysis(y_val, proba_val)
    table = pd.DataFrame(analysis["table"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        table["threshold"],
        table["cost_per_1000_accounts"],
        color=PALETTE["negative"],
        lw=2,
    )
    ax.axvline(
        threshold,
        color=PALETTE["accent"],
        linestyle="--",
        label=f"Selected threshold = {threshold:.2f}",
    )
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Expected cost per 1,000 accounts (units)")
    ax.set_title("Validation cost curve (5x FN cost, 1x FP cost)")
    ax.legend()
    paths["threshold_cost"] = _save(
        fig, output_dir / "model_threshold_cost_curve.png"
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax2 = ax.twinx()
    ax.plot(table["threshold"], table["recall"], label="Recall", color=PALETTE["positive"])
    ax.plot(
        table["threshold"],
        table["precision"],
        label="Precision",
        color=PALETTE["negative"],
    )
    ax2.plot(
        table["threshold"],
        table["flag_rate"],
        label="Flag rate",
        color=PALETTE["accent"],
        linestyle=":",
    )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Recall / precision")
    ax2.set_ylabel("Flag rate")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax.set_title("Validation trade-off: recall, precision, and review load")
    paths["threshold_tradeoff"] = _save(
        fig, output_dir / "model_threshold_tradeoff.png"
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_test, proba_test, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", color=PALETTE["negative"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration - {model_name} (test)")
    ax.legend()
    paths["calibration"] = _save(fig, output_dir / "model_calibration.png")

    paths["feature_importance"] = plot_feature_importance(
        pipeline, model_name, output_dir / "model_feature_importance.png"
    )
    return paths


def error_analysis(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    output_dir: Path | None = None,
) -> dict:
    """Break down errors by age, credit limit, and delinquency segments."""
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    frame = add_engineered_features(X_test.copy())
    frame["y_true"] = np.asarray(y_test)
    frame["y_pred"] = pred
    frame["y_prob"] = proba

    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[0, 29, 39, 49, 200],
        labels=["<30", "30-39", "40-49", "50+"],
        right=True,
    )
    frame["limit_group"] = pd.qcut(
        frame["limit_bal"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    frame["delinquency_group"] = pd.cut(
        frame["max_delinquency"],
        bins=[-10, 0, 1, 100],
        labels=["None", "1 month", "2+ months"],
    )

    segment_keys = ["age_group", "limit_group", "delinquency_group"]
    summaries: dict[str, list[dict]] = {}
    for key in segment_keys:
        rows = []
        for value, group in frame.groupby(key, observed=True):
            tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
            fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
            fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
            tn = int(((group["y_true"] == 0) & (group["y_pred"] == 0)).sum())
            rows.append(
                {
                    "segment": str(value),
                    "n": int(len(group)),
                    "default_rate": float(group["y_true"].mean()),
                    "recall": float(tp / (tp + fn)) if (tp + fn) else None,
                    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
                    "precision": float(tp / (tp + fp)) if (tp + fp) else None,
                    "flag_rate": float(group["y_pred"].mean()),
                    "mean_predicted_probability": float(group["y_prob"].mean()),
                }
            )
        summaries[key] = rows

    flat_rows = [
        {"segment_type": key, **row} for key, rows in summaries.items() for row in rows
    ]
    pd.DataFrame(flat_rows).to_csv(
        output_dir / "error_analysis_by_segment.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key in zip(axes, segment_keys):
        table = pd.DataFrame(summaries[key])
        positions = np.arange(len(table))
        ax.bar(
            positions - 0.2,
            table["recall"].fillna(0),
            width=0.4,
            label="Recall",
            color=PALETTE["positive"],
        )
        ax.bar(
            positions + 0.2,
            table["false_positive_rate"].fillna(0),
            width=0.4,
            label="False positive rate",
            color=PALETTE["negative"],
        )
        ax.set_xticks(positions, table["segment"], rotation=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(key.replace("_", " ").title())
        ax.legend(fontsize=8)
    fig.suptitle("Error analysis by segment (test)")
    _save(fig, output_dir / "figures" / "model_error_analysis.png")

    objectives = {
        "overall": {
            "recall": float(
                recall_score(y_test, pred, zero_division=0)
            ),
            "false_positive_rate": float(
                ((pred == 1) & (np.asarray(y_test) == 0)).sum()
                / max(1, (np.asarray(y_test) == 0).sum())
            ),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "flag_rate": float(pred.mean()),
        },
        "segments": summaries,
        "notes": [
            "Recall falls and FPR rises in segments with little delinquency "
            "history because the model has fewer strong signals.",
            "Segment analysis is descriptive; small groups should be "
            "interpreted cautiously.",
            "SEX is not used as a model feature, avoiding direct gender "
            "discrimination; AGE is used and is monitored for fairness.",
        ],
    }
    (output_dir / "error_analysis.json").write_text(json.dumps(objectives, indent=2))
    return objectives


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or config.METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=float))
    return path
