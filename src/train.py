"""Train, tune, compare, and track credit-default models.

Usage
-----
    python -m src.train                # full experiment
    python -m src.train --quick        # fast smoke run (for CI)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score

from src import config, data, evaluate, features

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def experiment_scoring() -> dict:
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }


def setup_mlflow() -> str:
    config.ensure_dirs()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.EXPERIMENT_NAME,
            artifact_location=str(config.MLFLOW_ARTIFACTS_DIR / config.EXPERIMENT_NAME),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _mean_std(folds: list[dict]) -> dict:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "flag_rate",
    ]
    result: dict[str, float] = {}
    for key in keys:
        values = [fold[key] for fold in folds if key in fold and fold[key] is not None]
        if values:
            result[f"cv_mean_{key}"] = float(np.mean(values))
            result[f"cv_std_{key}"] = float(np.std(values))
    result["cv_folds"] = len(folds)
    return result


def _log_confusion(prefix: str, metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    return {
        f"{prefix}_tn": cm["tn"],
        f"{prefix}_fp": cm["fp"],
        f"{prefix}_fn": cm["fn"],
        f"{prefix}_tp": cm["tp"],
    }


def _log_metric_dict(metrics: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if key in {"confusion_matrix", "cost"}:
            continue
        if isinstance(value, (int, float)) and value is not None:
            flat[f"{prefix}{key}"] = float(value)
    flat.update(_log_confusion(prefix.rstrip("_") or "val", metrics))
    return flat


def _log_common(
    dataset_info: dict,
    split_stats: dict,
    model_name: str,
    model_family: str,
    stage: str,
) -> None:
    mlflow.log_params(
        {
            "model_name": model_name,
            "model_family": model_family,
            "stage": stage,
            "random_seed": config.RANDOM_SEED,
            "cv_folds": split_stats["cv_folds"],
            "train_rows": split_stats["train_rows"],
            "validation_rows": split_stats["validation_rows"],
            "test_rows": split_stats["test_rows"],
            "feature_version": config.MODEL_VERSION,
            "dataset_name": config.DATASET_NAME,
        }
    )
    mlflow.set_tags(
        {
            "dataset_urn": config.DATASET_URN,
            "dataset_license": config.DATASET_LICENSE,
            "dataset_zip_md5": dataset_info["raw_archive"]["md5"],
            "dataset_raw_md5": dataset_info["raw_csv"]["md5"],
            "dataset_processed_md5": dataset_info["processed_csv"]["md5"],
            "git_commit": config.get_git_commit() or "not-a-git-repository",
            "python_version": config.library_versions()["python"],
        }
    )


def _save_and_log_importance(pipeline, name: str, run_active: bool = True) -> pd.Series:
    series = evaluate.feature_importance(pipeline)
    if series.empty:
        return series
    csv_path = config.REPORTS_DIR / f"importance_{name}.csv"
    series.rename("importance").to_csv(csv_path, header=True)
    img_path = config.FIGURES_DIR / f"model_feature_importance_{name}.png"
    evaluate.plot_feature_importance(pipeline, name, img_path)
    if run_active:
        mlflow.log_artifact(str(csv_path), artifact_path="diagnostics")
        mlflow.log_artifact(str(img_path), artifact_path="diagnostics")
    return series


def _save_and_log_threshold(analysis: dict, name: str, run_active: bool = True) -> None:
    table = pd.DataFrame(analysis["table"])
    csv_path = config.REPORTS_DIR / f"threshold_{name}.csv"
    table.to_csv(csv_path, index=False)
    summary_path = config.REPORTS_DIR / f"threshold_{name}.json"
    summary = {
        "assumptions": analysis["assumptions"],
        "best_unconstrained": analysis["best_unconstrained"],
        "best_within_capacity": analysis["best_within_capacity"],
        "selected": analysis["selected"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    if run_active:
        mlflow.log_artifact(str(csv_path), artifact_path="diagnostics")
        mlflow.log_artifact(str(summary_path), artifact_path="diagnostics")


def run_constant_baseline(
    X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
) -> dict:
    """Majority-class baseline: never flag any account."""
    with mlflow.start_run(run_name="baseline-constant") as run:
        _log_common(
            dataset_info, split_stats, "constant_majority", "baseline", "baseline"
        )
        prior = float(y_train.mean())
        mlflow.log_param("decision_rule", "always predict 0 (no default)")
        mlflow.log_metric("training_prevalence", prior)

        folds = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
        for _, val_idx in skf.split(X_train, y_train):
            y_fold = y_train.iloc[val_idx]
            pred = np.zeros(len(y_fold), dtype=int)
            proba = np.full(len(y_fold), prior)
            folds.append(evaluate.compute_metrics(y_fold, pred, proba))
        cv_metrics = _mean_std(folds)

        val_proba = np.full(len(y_val), prior)
        val_pred = np.zeros(len(y_val), dtype=int)
        val_metrics = evaluate.compute_metrics(y_val, val_pred, val_proba)

        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(_log_metric_dict(val_metrics, "val_"))
        mlflow.log_metric("val_threshold", 0.5)
        mlflow.set_tags({"decision": "never flag", "threshold_policy": "fixed"})
        result = {
            "model_name": "constant_majority",
            "model_family": "baseline",
            "run_id": run.info.run_id,
            "cv": cv_metrics,
            "validation": val_metrics,
            "threshold": 0.5,
            "best_params": {"rule": "always 0"},
            "feature_importance": {},
        }
        mlflow.log_dict(
            json.loads(json.dumps(result, default=float)),
            "diagnostics/summary.json",
        )
    print(f"[baseline] constant completed: run_id={result['run_id']}")
    return result


def run_business_rule_baseline(
    X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
) -> dict:
    """Operational rule: flag any account with at least one delayed payment."""
    with mlflow.start_run(run_name="baseline-business-rule") as run:
        _log_common(
            dataset_info, split_stats, "delinquency_rule", "baseline", "baseline"
        )
        mlflow.log_param(
            "decision_rule", "flag if max(PAY_0, PAY_2..PAY_6) >= 1"
        )
        fold_metrics = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
        for _, val_idx in skf.split(X_train, y_train):
            X_fold = X_train.iloc[val_idx]
            y_fold = y_train.iloc[val_idx]
            score = (X_fold[config.PAY_COLUMNS].max(axis=1) >= 1).astype(int).to_numpy()
            fold_metrics.append(evaluate.compute_metrics(y_fold, score, score))
        cv_metrics = _mean_std(fold_metrics)

        score_val = (X_val[config.PAY_COLUMNS].max(axis=1) >= 1).astype(int).to_numpy()
        val_metrics = evaluate.compute_metrics(y_val, score_val, score_val)

        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(_log_metric_dict(val_metrics, "val_"))
        mlflow.log_metric("val_threshold", 0.5)
        mlflow.set_tags(
            {"decision": "flag any delinquency", "threshold_policy": "business rule"}
        )
        result = {
            "model_name": "delinquency_rule",
            "model_family": "baseline",
            "run_id": run.info.run_id,
            "cv": cv_metrics,
            "validation": val_metrics,
            "threshold": 0.5,
            "best_params": {"rule": "max payment delay >= 1"},
            "feature_importance": {},
        }
        mlflow.log_dict(
            json.loads(json.dumps(result, default=float)),
            "diagnostics/summary.json",
        )
    print(f"[baseline] business rule completed: run_id={result['run_id']}")
    return result


def run_logistic_baseline(
    X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
) -> dict:
    """Untuned logistic regression with the same preprocessing pipeline."""
    set_seed()
    with mlflow.start_run(run_name="baseline-logistic-untuned") as run:
        _log_common(
            dataset_info,
            split_stats,
            "logistic_untuned",
            "logistic_regression",
            "baseline",
        )
        pipeline = features.build_pipeline(
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)
        )
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED
            ),
            scoring=experiment_scoring(),
            error_score="raise",
        )
        cv_metrics: dict[str, float] = {"cv_folds": cv}
        for metric in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            cv_metrics[f"cv_mean_{metric}"] = float(np.mean(cv_result[f"test_{metric}"]))
            cv_metrics[f"cv_std_{metric}"] = float(np.std(cv_result[f"test_{metric}"]))

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_val)[:, 1]
        analysis = evaluate.threshold_analysis(y_val, proba)
        threshold = analysis["selected"]["threshold"]
        val_metrics = evaluate.compute_metrics(
            y_val, (proba >= threshold).astype(int), proba
        )
        val_metrics["threshold"] = threshold
        val_metrics["cost"] = evaluate.expected_cost(
            y_val, (proba >= threshold).astype(int)
        )

        mlflow.log_params(
            {
                "classifier": "LogisticRegression",
                "C": 1.0,
                "class_weight": "none",
                "max_iter": 2000,
            }
        )
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(_log_metric_dict(val_metrics, "val_"))
        mlflow.log_metrics(
            {
                "val_cost_per_1000_accounts": val_metrics["cost"][
                    "cost_per_1000_accounts"
                ]
            }
        )
        mlflow.log_metric("val_threshold", threshold)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="pickle",
            input_example=X_train.head(3).astype(float),
        )
        _save_and_log_importance(pipeline, "logistic_untuned")
        _save_and_log_threshold(analysis, "logistic_untuned")
        mlflow.set_tags({"decision": "probability threshold", "threshold_policy": "tuned on validation"})

        result = {
            "model_name": "logistic_untuned",
            "model_family": "logistic_regression",
            "run_id": run.info.run_id,
            "cv": cv_metrics,
            "validation": val_metrics,
            "threshold": threshold,
            "best_params": {"C": 1.0, "class_weight": None},
            "feature_importance": evaluate.feature_importance(pipeline)
            .head(15)
            .round(6)
            .to_dict(),
            "threshold_analysis": {
                "best_unconstrained": analysis["best_unconstrained"],
                "best_within_capacity": analysis["best_within_capacity"],
            },
        }
    print(f"[baseline] logistic regression completed: run_id={result['run_id']}")
    return result


def run_tuned_model(
    model_name: str,
    model_family: str,
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_info,
    split_stats,
    cv: int,
) -> dict:
    """Tune a model with GridSearchCV, evaluate on validation, and log to MLflow."""
    set_seed()
    with mlflow.start_run(run_name=f"tuned-{model_name}") as run:
        _log_common(dataset_info, split_stats, model_name, model_family, "tuned")
        pipeline = features.build_pipeline(estimator)
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED
            ),
            scoring=experiment_scoring(),
            refit="pr_auc",
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        mlflow.log_params({k: str(v) for k, v in grid.best_params_.items()})
        cv_metrics: dict[str, float] = {"cv_folds": cv}
        cv_results = grid.cv_results_
        for metric in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            mean_key = f"mean_test_{metric}"
            std_key = f"std_test_{metric}"
            if mean_key in cv_results:
                cv_metrics[f"cv_mean_{metric}"] = float(cv_results[mean_key][grid.best_index_])
                cv_metrics[f"cv_std_{metric}"] = float(cv_results[std_key][grid.best_index_])
        cv_metrics["cv_best_rank"] = int(grid.best_index_ + 1)

        proba = best.predict_proba(X_val)[:, 1]
        analysis = evaluate.threshold_analysis(y_val, proba)
        threshold = analysis["selected"]["threshold"]
        val_metrics_05 = evaluate.compute_metrics(
            y_val, (proba >= 0.5).astype(int), proba
        )
        val_metrics = evaluate.compute_metrics(
            y_val, (proba >= threshold).astype(int), proba
        )
        val_metrics["threshold"] = threshold
        val_metrics["cost"] = evaluate.expected_cost(
            y_val, (proba >= threshold).astype(int)
        )

        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(_log_metric_dict(val_metrics_05, "val_at_05_"))
        mlflow.log_metrics(_log_metric_dict(val_metrics, "val_"))
        mlflow.log_metrics(
            {
                "val_cost_per_1000_accounts": val_metrics["cost"][
                    "cost_per_1000_accounts"
                ],
                "val_cost_per_1000_accounts_at_05": evaluate.expected_cost(
                    y_val, (proba >= 0.5).astype(int)
                )["cost_per_1000_accounts"],
            }
        )
        mlflow.log_metric("val_threshold", threshold)
        mlflow.log_param("search_space_size", int(np.prod([len(v) for v in param_grid.values()])))

        importance = _save_and_log_importance(best, model_name)
        _save_and_log_threshold(analysis, model_name)
        mlflow.sklearn.log_model(
            best,
            name="model",
            serialization_format="pickle",
            input_example=X_train.head(3).astype(float),
        )
        mlflow.set_tags(
            {
                "decision": "probability threshold",
                "threshold_policy": "cost-optimal within review capacity",
                "tuning_method": "GridSearchCV",
            }
        )

        result = {
            "model_name": model_name,
            "model_family": model_family,
            "run_id": run.info.run_id,
            "cv": cv_metrics,
            "validation": val_metrics,
            "validation_at_0.5": val_metrics_05,
            "threshold": threshold,
            "best_params": grid.best_params_,
            "feature_importance": importance.head(15).round(6).to_dict(),
            "threshold_analysis": {
                "best_unconstrained": analysis["best_unconstrained"],
                "best_within_capacity": analysis["best_within_capacity"],
            },
        }
        mlflow.log_dict(
            json.loads(
                json.dumps(
                    {
                        "best_params": {
                            k: str(v) for k, v in grid.best_params_.items()
                        },
                        "validation": val_metrics,
                        "threshold_summary": result["threshold_analysis"],
                    },
                    default=float,
                )
            ),
            "diagnostics/summary.json",
        )
    print(
        f"[tuned] {model_name} completed: PR-AUC={val_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={val_metrics['roc_auc']:.4f}, threshold={threshold:.2f}, "
        f"run_id={result['run_id']}"
    )
    return result


def bootstrap_ci(
    y_true: pd.Series,
    y_proba: np.ndarray,
    metric,
    n_boot: int = 500,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Percentile bootstrap confidence interval for a probability metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        if sample_y.min() == sample_y.max():
            continue
        values.append(metric(sample_y, y_proba[indices]))
    if not values:
        return {"estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    return {
        "estimate": float(metric(y_true, y_proba)),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "n_boot": len(values),
    }


def _plot_model_comparison(results: dict, path: Path) -> None:
    rows = []
    for name, result in results.items():
        cv = result["cv"]
        if "cv_mean_pr_auc" in cv:
            rows.append(
                {
                    "model": name,
                    "pr_auc": cv["cv_mean_pr_auc"],
                    "pr_std": cv.get("cv_std_pr_auc", 0.0),
                    "roc_auc": cv.get("cv_mean_roc_auc", np.nan),
                    "roc_std": cv.get("cv_std_roc_auc", 0.0),
                }
            )
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("pr_auc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(frame) + 2.5))
    for ax, metric, std, title in (
        (axes[0], "pr_auc", "pr_std", "Cross-validated PR-AUC"),
        (axes[1], "roc_auc", "roc_std", "Cross-validated ROC-AUC"),
    ):
        ax.barh(
            frame["model"],
            frame[metric],
            xerr=frame[std],
            color=evaluate.PALETTE["negative"],
            capsize=3,
        )
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("mean +/- std (5-fold)")
    fig.suptitle("Model comparison on the training folds")
    evaluate._save(fig, path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train credit-default models")
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--skip-mlflow", action="store_true", help="train without experiment tracking"
    )
    args = parser.parse_args(argv)

    set_seed()
    config.ensure_dirs()
    if not args.skip_mlflow:
        setup_mlflow()

    print("[data] building processed dataset ...")
    frame = data.build_processed_dataset()
    dataset_info = data.load_dataset_info()
    split = features.split_data(frame)
    X_train, X_val, X_test, y_train, y_val, y_test = split
    split_stats = {
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "cv_folds": 2 if args.quick else args.cv_folds,
        "train_default_rate": float(y_train.mean()),
        "validation_default_rate": float(y_val.mean()),
        "test_default_rate": float(y_test.mean()),
    }
    print(f"[data] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    cv = split_stats["cv_folds"]
    results: dict[str, dict] = {}
    results["constant"] = run_constant_baseline(
        X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
    )
    results["delinquency_rule"] = run_business_rule_baseline(
        X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
    )
    results["logistic_untuned"] = run_logistic_baseline(
        X_train, y_train, X_val, y_val, dataset_info, split_stats, cv
    )

    if args.quick:
        lr_grid = {"classifier__C": [1.0], "classifier__class_weight": [None]}
        rf_grid = {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [8],
            "classifier__min_samples_leaf": [5],
        }
        gb_grid = {
            "classifier__n_estimators": [60],
            "classifier__learning_rate": [0.1],
            "classifier__max_depth": [2],
        }
    else:
        lr_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }
        rf_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 5],
        }
        gb_grid = {
            "classifier__n_estimators": [150],
            "classifier__learning_rate": [0.05, 0.10],
            "classifier__max_depth": [2, 3],
            "classifier__subsample": [0.9],
        }

    results["logistic_regression"] = run_tuned_model(
        "logistic_regression",
        "logistic_regression",
        LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED),
        lr_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        dataset_info,
        split_stats,
        cv,
    )
    results["random_forest"] = run_tuned_model(
        "random_forest",
        "random_forest",
        RandomForestClassifier(
            random_state=config.RANDOM_SEED, n_jobs=1, class_weight="balanced_subsample"
        ),
        rf_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        dataset_info,
        split_stats,
        cv,
    )
    results["gradient_boosting"] = run_tuned_model(
        "gradient_boosting",
        "gradient_boosting",
        GradientBoostingClassifier(random_state=config.RANDOM_SEED),
        gb_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        dataset_info,
        split_stats,
        cv,
    )

    comparison_path = config.FIGURES_DIR / "model_comparison.png"
    _plot_model_comparison(results, comparison_path)

    # Model selection uses validation PR-AUC only; the test set is untouched.
    eligible = {
        name: result
        for name, result in results.items()
        if result["model_family"] != "baseline"
    }
    final_name, final_result = max(
        eligible.items(), key=lambda item: item[1]["validation"].get("pr_auc", -1)
    )
    final_run_id = final_result["run_id"]
    final_pipeline = mlflow.sklearn.load_model(f"runs:/{final_run_id}/model")
    threshold = float(final_result["threshold"])
    print(
        f"[select] {final_name} chosen on validation PR-AUC="
        f"{final_result['validation']['pr_auc']:.4f}"
    )

    # Single evaluation of the selected model on the held-out test set.
    proba_test = final_pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    test_metrics = evaluate.compute_metrics(y_test, pred_test, proba_test)
    test_metrics["threshold"] = threshold
    test_metrics["cost"] = evaluate.expected_cost(y_test, pred_test)
    test_metrics["roc_auc_ci"] = bootstrap_ci(y_test, proba_test, roc_auc_score)
    test_metrics["pr_auc_ci"] = bootstrap_ci(y_test, proba_test, average_precision_score)
    test_metrics_at_05 = evaluate.compute_metrics(
        y_test, (proba_test >= 0.5).astype(int), proba_test
    )
    test_metrics_at_05["cost"] = evaluate.expected_cost(
        y_test, (proba_test >= 0.5).astype(int)
    )
    print(
        f"[test] PR-AUC={test_metrics['pr_auc']:.4f} "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"recall={test_metrics['recall']:.3f} FPR={test_metrics['flag_rate']:.3f}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        diagnostics = evaluate.plot_model_diagnostics(
            final_pipeline,
            X_val,
            y_val,
            X_test,
            y_test,
            threshold,
            final_name,
            config.FIGURES_DIR,
        )
        error_summary = evaluate.error_analysis(
            final_pipeline, X_test, y_test, threshold, config.REPORTS_DIR
        )
        importance = evaluate.feature_importance(final_pipeline)
        importance.rename("importance").to_csv(
            config.REPORTS_DIR / "feature_importance.csv", header=True
        )
        joblib.dump(final_pipeline, config.MODEL_PATH, compress=3)

        metrics = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_info,
            "split": split_stats,
            "baselines": {
                "constant": results["constant"],
                "delinquency_rule": results["delinquency_rule"],
                "logistic_untuned": results["logistic_untuned"],
            },
            "models": {
                name: result
                for name, result in results.items()
                if name not in {"constant", "delinquency_rule", "logistic_untuned"}
            },
            "final_model": {
                "name": final_name,
                "run_id": final_run_id,
                "threshold": threshold,
                "selection_rule": "highest validation PR-AUC among tuned models",
                "test": test_metrics,
                "test_at_0.5": test_metrics_at_05,
                "test_cost_analysis": {
                    "selected_threshold": test_metrics["cost"],
                    "threshold_0.5": test_metrics_at_05["cost"],
                },
                "threshold_policy": final_result["threshold_analysis"],
            },
            "error_analysis": error_summary,
            "top_features": importance.head(20).round(6).to_dict(),
        }
        evaluate.save_metrics(metrics, config.METRICS_PATH)

        artifact_uri = mlflow.get_run(final_run_id).info.artifact_uri
        model_meta = {
            "model_name": f"credit-default-{final_name}",
            "model_version": config.MODEL_VERSION,
            "model_family": final_result["model_family"],
            "training_date_utc": datetime.now(timezone.utc).isoformat(),
            "training_dataset_size": int(len(X_train)),
            "validation_dataset_size": int(len(X_val)),
            "test_dataset_size": int(len(X_test)),
            "dataset": {
                "name": config.DATASET_NAME,
                "urn": config.DATASET_URN,
                "doi": config.DATASET_DOI,
                "license": config.DATASET_LICENSE,
                "source_url": config.DATASET_URL,
                "raw_archive_md5": dataset_info["raw_archive"]["md5"],
                "raw_csv_md5": dataset_info["raw_csv"]["md5"],
                "processed_csv_md5": dataset_info["processed_csv"]["md5"],
            },
            "target": config.TARGET,
            "raw_feature_names": list(config.RAW_FEATURES),
            "engineered_feature_names": list(config.ENGINEERED_FEATURES),
            "model_output_feature_count": len(
                features.feature_names_from_pipeline(final_pipeline)
            ),
            "prediction_threshold": threshold,
            "prediction_description": (
                "Estimated probability that the credit-card account will "
                "default on the next month's payment. Accounts with probability "
                f">= {threshold:.2f} are flagged for manual review."
            ),
            "recommended_action": (
                "If flagged: route to a human credit-risk reviewer before "
                "issuing new credit or increasing the limit. If not flagged: "
                "continue standard automated processing. The score must not be "
                "used as the sole reason to deny credit."
            ),
            "cost_assumptions": {
                "false_negative_cost_units": config.COST_FALSE_NEGATIVE,
                "false_positive_cost_units": config.COST_FALSE_POSITIVE,
                "review_capacity_fraction": config.REVIEW_CAPACITY_FRACTION,
            },
            "best_params": {k: str(v) for k, v in final_result["best_params"].items()},
            "cross_validation": final_result["cv"],
            "validation_metrics": final_result["validation"],
            "test_metrics": test_metrics,
            "library_versions": config.library_versions(),
            "git_commit": config.get_git_commit(),
            "mlflow_run_id": final_run_id,
            "mlflow_tracking_uri": config.MLFLOW_TRACKING_URI,
            "mlflow_artifact_uri": artifact_uri,
            "mlflow_model_uri": f"runs:/{final_run_id}/model",
            "limitations": [
                "Trained on Taiwan credit-card data from 2005; performance may "
                "drift on other markets, products, or time periods.",
                "The 5:1 FN:FP cost ratio and 30% review capacity are business "
                "assumptions and must be validated with stakeholders.",
                "Probabilities are not guaranteed to be perfectly calibrated; "
                "recalibrate after major population changes.",
                "SEX is excluded from the model to avoid direct gender "
                "discrimination; other protected attributes were unavailable.",
                "The model is a decision-support tool, not an automated "
                "credit-denial system.",
            ],
        }
        config.MODEL_META_PATH.write_text(json.dumps(model_meta, indent=2))

        with mlflow.start_run(run_id=final_run_id):
            mlflow.set_tag(config.FINAL_MODEL_TAG, "true")
            mlflow.set_tag("selected_model", final_name)
            mlflow.log_metrics(
                {
                    **_log_metric_dict(test_metrics, "test_"),
                    "test_cost_per_1000_accounts": test_metrics["cost"][
                        "cost_per_1000_accounts"
                    ],
                    "test_pr_auc_ci_lower": test_metrics["pr_auc_ci"]["ci_lower"],
                    "test_pr_auc_ci_upper": test_metrics["pr_auc_ci"]["ci_upper"],
                    "test_roc_auc_ci_lower": test_metrics["roc_auc_ci"]["ci_lower"],
                    "test_roc_auc_ci_upper": test_metrics["roc_auc_ci"]["ci_upper"],
                }
            )
            mlflow.log_metric("prediction_threshold", threshold)
            mlflow.log_artifact(str(config.MODEL_PATH), artifact_path="artifacts")
            mlflow.log_artifact(str(config.MODEL_META_PATH), artifact_path="artifacts")
            mlflow.log_artifact(str(config.METRICS_PATH), artifact_path="reports")
            mlflow.log_artifact(
                str(config.EDA_SUMMARY_PATH), artifact_path="reports"
            )
            for path in diagnostics.values():
                mlflow.log_artifact(str(path), artifact_path="figures")
            mlflow.log_artifact(
                str(config.FIGURES_DIR / "model_error_analysis.png"),
                artifact_path="figures",
            )
            mlflow.log_artifact(
                str(config.REPORTS_DIR / "error_analysis.json"),
                artifact_path="reports",
            )
            mlflow.log_artifact(
                str(config.REPORTS_DIR / "error_analysis_by_segment.csv"),
                artifact_path="reports",
            )
            mlflow.log_artifact(
                str(config.REPORTS_DIR / "feature_importance.csv"),
                artifact_path="reports",
            )
            mlflow.log_artifact(str(comparison_path), artifact_path="figures")

    print(f"[save] model -> {config.MODEL_PATH}")
    print(f"[save] metadata -> {config.MODEL_META_PATH}")
    print(f"[save] metrics -> {config.METRICS_PATH}")
    print(f"[mlflow] final run id: {final_run_id}")
    return metrics


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
