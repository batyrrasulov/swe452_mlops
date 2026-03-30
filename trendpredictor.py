from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_TICKERS = ("AAPL",)
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2025-01-01"
DEFAULT_FORECAST_HORIZON = 10
DEFAULT_TARGET_RETURN_THRESHOLD = 0.01
DEFAULT_TRAIN_RATIO = 0.60
DEFAULT_VALIDATION_RATIO = 0.20
DEFAULT_PLOT_POINTS = 120
DEFAULT_RANDOM_STATE = 42
MODEL_NAMES = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class RunConfig:
    tickers: tuple[str, ...] = DEFAULT_TICKERS
    focus_ticker: str | None = None
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    target_return_threshold: float = DEFAULT_TARGET_RETURN_THRESHOLD
    train_ratio: float = DEFAULT_TRAIN_RATIO
    validation_ratio: float = DEFAULT_VALIDATION_RATIO
    plot_points: int = DEFAULT_PLOT_POINTS
    random_state: int = DEFAULT_RANDOM_STATE
    output_dir: Path = field(default_factory=lambda: Path("artifacts/latest"))
    save_plot: bool = True

    def resolved_focus_ticker(self) -> str:
        return self.focus_ticker or self.tickers[0]


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train and evaluate a stock trend prediction model.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="Ticker symbols to train on.")
    parser.add_argument("--focus-ticker", default=None, help="Ticker used for the saved prediction plot.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Historical download start date.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Historical download end date.")
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON,
        help="Number of trading days ahead used to define the target.",
    )
    parser.add_argument(
        "--target-return-threshold",
        type=float,
        default=DEFAULT_TARGET_RETURN_THRESHOLD,
        help="Positive class threshold for future return.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Training split ratio.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Validation split ratio.",
    )
    parser.add_argument("--plot-points", type=int, default=DEFAULT_PLOT_POINTS, help="Points shown in the saved plot.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    parser.add_argument("--output-dir", default="artifacts/latest", help="Directory for saved model artifacts.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    return RunConfig(
        tickers=tuple(dict.fromkeys(ticker.upper() for ticker in args.tickers)),
        focus_ticker=args.focus_ticker.upper() if args.focus_ticker else None,
        start_date=args.start_date,
        end_date=args.end_date,
        forecast_horizon=args.forecast_horizon,
        target_return_threshold=args.target_return_threshold,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        plot_points=args.plot_points,
        random_state=args.random_state,
        output_dir=Path(args.output_dir),
        save_plot=not args.no_plot,
    )


def validate_config(config: RunConfig) -> None:
    if not config.tickers:
        raise ValueError("At least one ticker is required.")

    if config.focus_ticker and config.focus_ticker not in config.tickers:
        raise ValueError("focus_ticker must be included in tickers.")

    if config.forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    if config.train_ratio <= 0 or config.validation_ratio <= 0:
        raise ValueError("train_ratio and validation_ratio must be positive.")

    if config.train_ratio + config.validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    if config.plot_points <= 0:
        raise ValueError("plot_points must be positive.")


def normalize_download_frame(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    return normalized


def download_price_history(tickers: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        data = normalize_download_frame(data)

        if data.empty:
            failed_tickers.append(ticker)
            continue

        prepared = data.reset_index()
        prepared["Ticker"] = ticker
        frames.append(prepared)

    if not frames:
        raise ValueError(f"No market data could be downloaded for {', '.join(tickers)}")

    if failed_tickers:
        print(f"Warning: skipped tickers with no data: {', '.join(failed_tickers)}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def engineer_features(
    market_data: pd.DataFrame,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    target_return_threshold: float = DEFAULT_TARGET_RETURN_THRESHOLD,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for _, ticker_frame in market_data.groupby("Ticker", sort=False):
        df = ticker_frame.sort_values("Date").copy()
        close = df["Close"]

        df["Return_1d"] = close.pct_change()
        df["Return_3d"] = close.pct_change(3)
        df["Return_5d"] = close.pct_change(5)
        df["Return_10d"] = close.pct_change(10)
        df["Momentum_5d"] = close / close.shift(5) - 1
        df["Momentum_10d"] = close / close.shift(10) - 1

        for window in (5, 10, 20, 50):
            df[f"MA_{window}"] = close.rolling(window).mean()

        for window in (5, 10, 20):
            df[f"Volatility_{window}"] = df["Return_1d"].rolling(window).std()

        df["MA_5_vs_20"] = df["MA_5"] / df["MA_20"] - 1
        df["MA_10_vs_50"] = df["MA_10"] / df["MA_50"] - 1
        df["Close_vs_MA_10"] = close / df["MA_10"] - 1
        df["Close_vs_MA_20"] = close / df["MA_20"] - 1
        df["Close_vs_MA_50"] = close / df["MA_50"] - 1
        df["RSI_14"] = compute_rsi(close, 14)

        rolling_high = df["High"].rolling(14).max()
        rolling_low = df["Low"].rolling(14).min()
        df["Stoch_K"] = 100 * (close - rolling_low) / (rolling_high - rolling_low)

        df["Intraday_Range"] = (df["High"] - df["Low"]) / close
        df["Open_Close_Gap"] = (df["Open"] - close.shift(1)) / close.shift(1)
        df["Volume_Change"] = df["Volume"].pct_change()
        rolling_volume_mean = df["Volume"].rolling(20).mean()
        rolling_volume_std = df["Volume"].rolling(20).std()
        df["Volume_Z"] = (df["Volume"] - rolling_volume_mean) / rolling_volume_std.replace(0, np.nan)
        df["Weekday"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month

        for lag in range(1, 6):
            df[f"Return_1d_lag_{lag}"] = df["Return_1d"].shift(lag)

        df["Future_Return"] = close.shift(-forecast_horizon) / close - 1
        df["Target"] = (df["Future_Return"] > target_return_threshold).astype(int)
        frames.append(df)

    feature_frame = pd.concat(frames, ignore_index=True)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feature_frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def build_feature_columns(model_data: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "Date",
        "Target",
        "Future_Return",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    }
    return [column for column in model_data.columns if column not in excluded_columns]


def split_dataset_by_date(
    model_data: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> DatasetSplit:
    unique_dates = np.array(sorted(model_data["Date"].unique()))
    if len(unique_dates) < 30:
        raise ValueError("Not enough observations to create reliable train/validation/test splits.")

    train_end = max(1, int(len(unique_dates) * train_ratio))
    validation_end = max(train_end + 1, int(len(unique_dates) * (train_ratio + validation_ratio)))

    if validation_end >= len(unique_dates):
        raise ValueError("Split configuration leaves no dates for the test set.")

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    train = model_data[model_data["Date"].isin(train_dates)].copy()
    validation = model_data[model_data["Date"].isin(validation_dates)].copy()
    test = model_data[model_data["Date"].isin(test_dates)].copy()
    return DatasetSplit(train=train, validation=validation, test=test)


def build_model_candidates(feature_columns: list[str], random_state: int) -> dict[str, Pipeline]:
    categorical_features = [column for column in feature_columns if column == "Ticker"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def choose_decision_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_score = float("-inf")

    for threshold in np.arange(0.35, 0.66, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, predictions)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def train_and_select_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> tuple[str, float, Pipeline, dict[str, dict[str, float]]]:
    candidates = build_model_candidates(feature_columns, random_state)
    candidate_metrics: dict[str, dict[str, float]] = {}
    best_name = ""
    best_threshold = 0.50
    best_score = float("-inf")

    for name, model in candidates.items():
        model.fit(train_frame[feature_columns], train_frame["Target"])
        validation_probabilities = model.predict_proba(validation_frame[feature_columns])[:, 1]
        threshold, score = choose_decision_threshold(validation_frame["Target"], validation_probabilities)

        candidate_metrics[name] = {
            "validation_balanced_accuracy": score,
            "decision_threshold": threshold,
        }

        if score > best_score:
            best_name = name
            best_threshold = threshold
            best_score = score

    selected_model = build_model_candidates(feature_columns, random_state)[best_name]
    train_validation_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    selected_model.fit(train_validation_frame[feature_columns], train_validation_frame["Target"])
    return best_name, best_threshold, selected_model, candidate_metrics


def evaluate_classification(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, digits=4, zero_division=0, output_dict=True),
    }

    if probabilities is not None and y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))

    return metrics


def baseline_scores(train_validation_targets: pd.Series, test_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    majority_class = int(train_validation_targets.mode().iloc[0])
    majority_predictions = np.full(len(test_frame), majority_class, dtype=int)
    baselines = {
        "majority_class": {
            "accuracy": float(accuracy_score(test_frame["Target"], majority_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_frame["Target"], majority_predictions)),
        }
    }

    if "Return_1d_lag_1" in test_frame.columns:
        momentum_predictions = (test_frame["Return_1d_lag_1"] > 0).astype(int).to_numpy()
        baselines["previous_day_direction"] = {
            "accuracy": float(accuracy_score(test_frame["Target"], momentum_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_frame["Target"], momentum_predictions)),
        }

    return baselines


def build_prediction_frame(
    test_frame: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    prediction_frame = test_frame[["Date", "Ticker", "Close", "Future_Return", "Target"]].copy()
    prediction_frame["Predicted_Probability"] = probabilities
    prediction_frame["Predicted_Target"] = predictions
    prediction_frame["Correct"] = (prediction_frame["Target"] == prediction_frame["Predicted_Target"]).astype(int)
    return prediction_frame


def per_ticker_metrics(prediction_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for ticker, frame in prediction_frame.groupby("Ticker"):
        metrics[ticker] = {
            "accuracy": float(accuracy_score(frame["Target"], frame["Predicted_Target"])),
            "balanced_accuracy": float(balanced_accuracy_score(frame["Target"], frame["Predicted_Target"])),
            "positive_rate": float(frame["Target"].mean()),
            "predicted_up_rate": float(frame["Predicted_Target"].mean()),
            "mean_future_return": float(frame["Future_Return"].mean()),
            "mean_future_return_when_predicted_up": float(
                frame.loc[frame["Predicted_Target"] == 1, "Future_Return"].mean()
            )
            if (frame["Predicted_Target"] == 1).any()
            else 0.0,
        }
    return metrics


def extract_top_features(model: Pipeline, top_n: int = 15) -> list[dict[str, float | str]]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "coef_"):
        importances = np.abs(np.ravel(estimator.coef_))
    elif hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    else:
        return []

    ranking = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"feature": name, "importance": float(score)} for name, score in ranking]


def save_prediction_plot(
    prediction_frame: pd.DataFrame,
    focus_ticker: str,
    threshold: float,
    output_path: Path,
    plot_points: int,
) -> None:
    plot_frame = prediction_frame[prediction_frame["Ticker"] == focus_ticker].sort_values("Date").tail(plot_points)
    if plot_frame.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(plot_frame["Date"], plot_frame["Predicted_Probability"], label="Predicted probability", color="#0b7285")
    plt.scatter(
        plot_frame["Date"],
        plot_frame["Target"],
        c=plot_frame["Target"],
        cmap="RdYlGn",
        label="Actual target",
        alpha=0.75,
    )
    plt.axhline(threshold, linestyle="--", color="#c92a2a", label=f"Decision threshold ({threshold:.2f})")
    plt.title(f"{focus_ticker} predicted up-move probability")
    plt.xlabel("Date")
    plt.ylabel("Probability / Actual label")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix_plot(confusion_matrix_values: list[list[int]], output_path: Path) -> None:
    matrix = np.asarray(confusion_matrix_values)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, int(matrix[row, column]), ha="center", va="center", color="#111111")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_feature_importance_plot(top_features: list[dict[str, float | str]], output_path: Path, top_n: int = 10) -> None:
    if not top_features:
        return

    feature_frame = pd.DataFrame(top_features[:top_n]).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(feature_frame["feature"], feature_frame["importance"], color="#0b7285")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_benchmark_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for name, metrics in summary["candidate_models"].items():
        rows.append(
            {
                "stage": "validation",
                "name": name,
                "accuracy": np.nan,
                "balanced_accuracy": metrics["validation_balanced_accuracy"],
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "decision_threshold": metrics["decision_threshold"],
            }
        )

    for name, metrics in summary["baselines"].items():
        rows.append(
            {
                "stage": "test_baseline",
                "name": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "decision_threshold": np.nan,
            }
        )

    rows.append(
        {
            "stage": "test_model",
            "name": summary["best_model_name"],
            "accuracy": summary["test_metrics"]["accuracy"],
            "balanced_accuracy": summary["test_metrics"]["balanced_accuracy"],
            "precision": summary["test_metrics"]["precision"],
            "recall": summary["test_metrics"]["recall"],
            "f1": summary["test_metrics"]["f1"],
            "roc_auc": summary["test_metrics"].get("roc_auc", np.nan),
            "decision_threshold": summary["decision_threshold"],
        }
    )

    return pd.DataFrame(rows)


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    return value


def save_artifacts(
    config: RunConfig,
    summary: dict[str, Any],
    model: Pipeline,
    raw_market_data: pd.DataFrame,
    model_data: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    scored_test_frame: pd.DataFrame,
    top_features: list[dict[str, float | str]],
) -> None:
    output_dir = ensure_output_dir(config.output_dir)
    raw_market_data.to_csv(output_dir / "raw_market_data.csv", index=False)
    model_data.to_csv(output_dir / "engineered_features.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    scored_test_frame.to_csv(output_dir / "scored_test_dataset.csv", index=False)
    pd.DataFrame(top_features).to_csv(output_dir / "top_features.csv", index=False)
    build_benchmark_frame(summary).to_csv(output_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(summary["test_metrics"]["confusion_matrix"], index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    joblib.dump(model, output_dir / "model.joblib")

    save_confusion_matrix_plot(summary["test_metrics"]["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_feature_importance_plot(top_features, output_dir / "feature_importance.png")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(to_json_ready(summary), metrics_file, indent=2)

    summary_text = [
        "Stock Price Trend Prediction Run Summary",
        f"Best model: {summary['best_model_name']}",
        f"Validation balanced accuracy: {summary['candidate_models'][summary['best_model_name']]['validation_balanced_accuracy']:.4f}",
        f"Test balanced accuracy: {summary['test_metrics']['balanced_accuracy']:.4f}",
        f"Decision threshold: {summary['decision_threshold']:.2f}",
        f"Tickers: {', '.join(config.tickers)}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_text), encoding="utf-8")


def run_pipeline(config: RunConfig, raw_data: pd.DataFrame | None = None) -> dict[str, Any]:
    validate_config(config)
    market_data = raw_data.copy() if raw_data is not None else download_price_history(config.tickers, config.start_date, config.end_date)
    market_data["Date"] = pd.to_datetime(market_data["Date"])

    model_data = engineer_features(
        market_data,
        forecast_horizon=config.forecast_horizon,
        target_return_threshold=config.target_return_threshold,
    )
    feature_columns = build_feature_columns(model_data)
    split = split_dataset_by_date(model_data, train_ratio=config.train_ratio, validation_ratio=config.validation_ratio)

    best_model_name, decision_threshold, model, candidate_models = train_and_select_model(
        split.train,
        split.validation,
        feature_columns,
        random_state=config.random_state,
    )

    test_probabilities = model.predict_proba(split.test[feature_columns])[:, 1]
    test_predictions = (test_probabilities >= decision_threshold).astype(int)
    prediction_frame = build_prediction_frame(split.test, test_probabilities, test_predictions)
    scored_test_frame = split.test.copy()
    scored_test_frame["Predicted_Probability"] = test_probabilities
    scored_test_frame["Predicted_Target"] = test_predictions
    scored_test_frame["Correct"] = (scored_test_frame["Target"] == scored_test_frame["Predicted_Target"]).astype(int)

    focus_ticker = config.resolved_focus_ticker()
    plot_path = config.output_dir / "prediction_plot.png"
    if config.save_plot:
        ensure_output_dir(config.output_dir)
        save_prediction_plot(prediction_frame, focus_ticker, decision_threshold, plot_path, config.plot_points)

    train_validation_targets = pd.concat([split.train["Target"], split.validation["Target"]], ignore_index=True)
    summary: dict[str, Any] = {
        "config": asdict(config),
        "dataset": {
            "raw_rows": int(len(market_data)),
            "model_rows": int(len(model_data)),
            "feature_count": len(feature_columns),
            "focus_ticker": focus_ticker,
            "date_range": {
                "start": model_data["Date"].min().strftime("%Y-%m-%d"),
                "end": model_data["Date"].max().strftime("%Y-%m-%d"),
            },
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "positive_class_rate": float(model_data["Target"].mean()),
        },
        "candidate_models": candidate_models,
        "best_model_name": best_model_name,
        "decision_threshold": float(decision_threshold),
        "baselines": baseline_scores(train_validation_targets, split.test),
        "test_metrics": evaluate_classification(split.test["Target"], test_predictions, test_probabilities),
        "per_ticker_metrics": per_ticker_metrics(prediction_frame),
        "signal_quality": {
            "mean_future_return": float(prediction_frame["Future_Return"].mean()),
            "mean_future_return_when_predicted_up": float(
                prediction_frame.loc[prediction_frame["Predicted_Target"] == 1, "Future_Return"].mean()
            )
            if (prediction_frame["Predicted_Target"] == 1).any()
            else 0.0,
            "predicted_up_rate": float(prediction_frame["Predicted_Target"].mean()),
        },
    }
    top_features = extract_top_features(model)
    summary["top_features"] = top_features

    save_artifacts(config, summary, model, market_data, model_data, prediction_frame, scored_test_frame, top_features)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    dataset = summary["dataset"]
    print("Stock Price Trend Prediction")
    print("=" * 40)
    print(f"Tickers: {', '.join(summary['config']['tickers'])}")
    print(f"Rows after feature engineering: {dataset['model_rows']}")
    print(f"Train / validation / test rows: {dataset['train_rows']} / {dataset['validation_rows']} / {dataset['test_rows']}")
    print(f"Feature count: {dataset['feature_count']}")
    print(f"Positive class rate: {dataset['positive_class_rate']:.4f}")
    print()
    print("Validation results")
    for name in MODEL_NAMES:
        if name in summary["candidate_models"]:
            validation = summary["candidate_models"][name]
            print(
                f"- {name}: balanced_accuracy={validation['validation_balanced_accuracy']:.4f}, "
                f"threshold={validation['decision_threshold']:.2f}"
            )
    print()
    print(f"Selected model: {summary['best_model_name']}")
    print(f"Decision threshold: {summary['decision_threshold']:.2f}")
    print(f"Test accuracy: {summary['test_metrics']['accuracy']:.4f}")
    print(f"Test balanced accuracy: {summary['test_metrics']['balanced_accuracy']:.4f}")
    print(f"Test precision: {summary['test_metrics']['precision']:.4f}")
    print(f"Test recall: {summary['test_metrics']['recall']:.4f}")
    print(f"Test F1: {summary['test_metrics']['f1']:.4f}")
    if "roc_auc" in summary["test_metrics"]:
        print(f"Test ROC AUC: {summary['test_metrics']['roc_auc']:.4f}")

    print()
    print("Baselines")
    for name, metrics in summary["baselines"].items():
        print(f"- {name}: accuracy={metrics['accuracy']:.4f}, balanced_accuracy={metrics['balanced_accuracy']:.4f}")

    print()
    print("Top features")
    for item in summary["top_features"][:10]:
        print(f"- {item['feature']}: {item['importance']:.6f}")

    print()
    print(f"Artifacts saved to: {summary['config']['output_dir']}")


def main() -> None:
    config = parse_args()
    summary = run_pipeline(config)
    print_summary(summary)


if __name__ == "__main__":
    main()
