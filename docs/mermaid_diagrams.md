# Mermaid Diagrams

This file contains clean standalone Mermaid diagrams for the final report, README, presentation, and proof package.

## System Architecture

```mermaid
flowchart TD
    A[Yahoo Finance via yfinance] --> B[Raw Market Data]
    B --> C[Feature Engineering]
    C --> D[Chronological Split]
    D --> E[Candidate Models]
    E --> E1[Logistic Regression]
    E --> E2[Random Forest]
    E1 --> F[Validation Scoring]
    E2 --> F
    F --> G[Threshold Selection]
    G --> H[Final Refit]
    H --> I[Test Evaluation]
    I --> J[Artifacts]
```

## Data And Feature Flow

```mermaid
flowchart LR
    A[OHLCV Input] --> B[Returns]
    A --> C[Moving Averages]
    A --> D[Volatility]
    A --> E[Momentum]
    A --> F[RSI and Stochastic]
    A --> G[Volume Signals]
    B --> H[Feature Matrix]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I[Target Label]
```

## Execution Sequence

```mermaid
sequenceDiagram
    participant User
    participant Script as trendpredictor.py
    participant Data as yfinance
    participant Model as sklearn pipeline
    participant Output as artifacts/latest

    User->>Script: Run training command
    Script->>Data: Download AAPL data
    Data-->>Script: Historical OHLCV
    Script->>Script: Build features and target
    Script->>Model: Train candidate models
    Model-->>Script: Validation scores
    Script->>Script: Select best model and threshold
    Script->>Model: Refit and score test set
    Script->>Output: Save model, metrics, plots, datasets
    Script-->>User: Print final benchmark summary
```

## Artifact Traceability

```mermaid
flowchart LR
    A[raw_market_data.csv] --> B[engineered_features.csv]
    B --> C[scored_test_dataset.csv]
    C --> D[predictions.csv]
    C --> E[confusion_matrix.csv]
    C --> F[benchmark_summary.csv]
    B --> G[model.joblib]
    G --> H[top_features.csv]
    H --> I[feature_importance.png]
    C --> J[prediction_plot.png]
```

## Validation Summary Flow

```mermaid
flowchart TD
    A[Train Split] --> B[Validation Split]
    B --> C[Balanced Accuracy Comparison]
    C --> D[Decision Threshold = 0.35]
    D --> E[Best Model = Random Forest]
    E --> F[Test Balanced Accuracy = 0.5581]
    F --> G[Baseline Comparison]
    G --> H[Majority = 0.5000]
    G --> I[Previous Day Direction = 0.5121]
```