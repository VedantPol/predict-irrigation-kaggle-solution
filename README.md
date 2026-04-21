# Predict Irrigation Need - 4-Level Architecture

Kaggle competition: `playground-series-s6e4`  
Metric: Balanced Accuracy (`Low`, `Medium`, `High`)

This repo uses the same architecture style as the fraud-risk project:

1. `prepare_data`
2. `feature_engineering`
3. `baseline` (Level-2 tree suite + Level-3 XGBoost + Level-4 logistic stacker)
4. `eda`
5. `hill_climb`
6. `stacking`
7. `pseudo_labeling`
8. `extra_training`

## Project Structure

- `src/fraud_risk_early_warning/pipeline.py`: end-to-end stage logic
- `src/fraud_risk_early_warning/advanced_features.py`: feature families
- `scripts/run_pipeline.py`: all runtime knobs
- `scripts/export_dashboard_data.py`: dashboard JSON export
- `outputs/submissions/submission_extra_training.csv`: Kaggle-ready submission

## Data

Expected files:

- `data/train.csv` (with `Irrigation_Need`)
- `data/test.csv` (with `id`, no target)
- `data/sample_submission.csv` (optional check)
- `data/irrigation_prediction.csv` (optional reference lookup)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

The default run order executes all stages through `extra_training`.
The runner now supports cached-resume for heavy stages (`prepare_data`, `feature_engineering`, `baseline`) via `scripts/run_pipeline.py`, so unchanged runs skip recomputation automatically.

## Validation Audit

To verify that reported validation metrics exactly match saved prediction files:

```bash
python scripts/audit_validation.py
```

This audit prints per-stage `reported` vs `calculated` score checks and a bootstrap confidence interval for `final_blend`.

## Kaggle Notebook (Public + Editable)

Use `notebooks/kaggle_solution_walkthrough.ipynb` as the shareable competition notebook.

What it does:

- Clones this repo in Kaggle runtime
- Copies competition CSVs from `/kaggle/input/playground-series-s6e4` into `data/`
- Runs staged training through `final_blend` with CPU-safe defaults
- Writes `/kaggle/working/submission.csv` for direct competition submission

How to submit and share:

1. Open the notebook on Kaggle and click **Save Version** -> **Save & Run All (Commit)**.
2. Click **Submit to Competition** and choose `/kaggle/working/submission.csv`.
3. In notebook **Settings**, set **Visibility** to **Public**.
4. Share the notebook URL. Others can copy it, run it, and edit their own version.

## Main Outputs

- Level-2/3/4 artifacts: `outputs/level2_results/...`
- Model OOF preds: `outputs/oof/`
- Test preds: `outputs/pred/`
- Pseudo labels: `outputs/level2_results/pseudo_labeling/pseudo_labels.parquet`
- Final submission: `outputs/submissions/submission_extra_training.csv`

## Config Knobs

Edit only `scripts/run_pipeline.py`:

- `PIPELINE_STAGE`
- `RESUME_FROM_ARTIFACTS` / `FORCE_RERUN_STAGES`
- `PREPARE_DATA_CONFIG`
- `FEATURE_ENGINEERING_CONFIG`
- `BASELINE_CONFIG`
