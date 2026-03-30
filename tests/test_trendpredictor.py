import json

import numpy as np
import pandas as pd

import trendpredictor as tp


def make_synthetic_market_data(days: int = 260, tickers: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=days)
    frames: list[pd.DataFrame] = []

    for index, ticker in enumerate(tickers):
        time_index = np.arange(days)
        base = 100 + (0.04 + index * 0.01) * time_index
        cycle = 3.5 * np.sin(time_index / (7 + index) + index)
        close = base + cycle + rng.normal(0, 0.45, size=days)
        open_price = close * (1 + rng.normal(0, 0.0025, size=days))
        high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, size=days))
        low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, size=days))
        volume = 1_000_000 + index * 125_000 + rng.integers(0, 150_000, size=days)

        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "Ticker": ticker,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def test_engineer_features_creates_expected_columns() -> None:
    raw_data = make_synthetic_market_data(days=180, tickers=("AAPL",))
    features = tp.engineer_features(raw_data, forecast_horizon=5, target_return_threshold=0.003)

    expected_columns = {
        "Ticker",
        "Return_1d",
        "MA_20",
        "Volatility_10",
        "RSI_14",
        "Future_Return",
        "Target",
    }
    assert expected_columns.issubset(features.columns)
    assert set(features["Target"].unique()).issubset({0, 1})
    assert not features.isna().any().any()


def test_split_dataset_by_date_is_chronological() -> None:
    raw_data = make_synthetic_market_data(days=220)
    features = tp.engineer_features(raw_data, forecast_horizon=5, target_return_threshold=0.003)
    split = tp.split_dataset_by_date(features, train_ratio=0.6, validation_ratio=0.2)

    assert split.train["Date"].max() < split.validation["Date"].min()
    assert split.validation["Date"].max() < split.test["Date"].min()
    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.test) > 0


def test_run_pipeline_saves_expected_artifacts(tmp_path) -> None:
    raw_data = make_synthetic_market_data(days=240)
    config = tp.RunConfig(
        tickers=("AAPL", "MSFT", "NVDA"),
        focus_ticker="AAPL",
        forecast_horizon=5,
        target_return_threshold=0.003,
        output_dir=tmp_path / "artifacts",
        plot_points=40,
        random_state=11,
    )

    summary = tp.run_pipeline(config, raw_data=raw_data)

    assert summary["best_model_name"] in tp.MODEL_NAMES
    assert 0.0 <= summary["test_metrics"]["balanced_accuracy"] <= 1.0
    assert (config.output_dir / "metrics.json").exists()
    assert (config.output_dir / "model.joblib").exists()
    assert (config.output_dir / "predictions.csv").exists()
    assert (config.output_dir / "top_features.csv").exists()
    assert (config.output_dir / "prediction_plot.png").exists()

    saved_metrics = json.loads((config.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["best_model_name"] == summary["best_model_name"]
    assert "per_ticker_metrics" in saved_metrics