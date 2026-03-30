# Video Demo Runbook

This runbook gives a clean terminal sequence for recording the SDLC II project demo.

## 1. Open The Project

```bash
cd /Users/batyr.rasulov/sptp_mlops
source .venv/bin/activate
```

## 2. Show The Repository Contents

```bash
ls
ls docs
ls artifacts/latest
```

Talk track:

- Show that the repository contains the pipeline, tests, documentation, and generated artifacts.
- Mention that the final validated baseline is AAPL with a 10-day horizon and a 1% target threshold.

## 3. Show The Automated Tests

```bash
python -m pytest -q
```

Talk track:

- Explain that the tests verify feature engineering, chronological splitting, and end-to-end artifact generation.

## 4. Run The Full Pipeline

```bash
python trendpredictor.py --output-dir artifacts/latest
```

Talk track:

- Explain that this command downloads historical Yahoo Finance data, engineers features, trains the candidate models, selects the best one, evaluates on the held-out test set, and writes all artifacts.

## 5. Show The Generated Outputs

```bash
ls artifacts/latest
head -5 artifacts/latest/predictions.csv
head -5 artifacts/latest/benchmark_summary.csv
head -5 artifacts/latest/confusion_matrix.csv
```

Talk track:

- Show the row-level predictions.
- Show the benchmark comparison against baselines.
- Show the confusion matrix counts.

## 6. Open The Charts

```bash
open artifacts/latest/prediction_plot.png
open artifacts/latest/confusion_matrix.png
open artifacts/latest/feature_importance.png
```

Talk track:

- `prediction_plot.png` shows predicted probabilities and actual labels over time.
- `confusion_matrix.png` shows classification outcomes.
- `feature_importance.png` shows the top drivers used by the selected model.

## 7. Show The Documentation

```bash
open docs/final_project_report.pdf
open docs/proof_of_execution.pdf
```

Talk track:

- Explain that the final report contains the project overview, business value, architecture, implementation summary, validation results, and future directions.
- Explain that the proof document maps input data to engineered features, scored predictions, benchmarks, and saved artifacts.

## 8. Optional GitHub Push

If you want to show the GitHub publication step after committing:

```bash
git status
git push -u origin main
```

If your default branch is `master` instead of `main`, use:

```bash
git push -u origin master
```