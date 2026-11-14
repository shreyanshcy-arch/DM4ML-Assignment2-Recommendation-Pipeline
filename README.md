# DM4ML Assignment 2 – End-to-End Recommendation Pipeline

**Student:** Shreyansh Chaudhary  
**Roll Number:** 2025AG05905  
**Dataset:** MovieLens `ml-latest-small`  
**Pipeline:** Ingestion → Validation → Preparation → Transformation → Feature Management → Training → Evaluation → Orchestration

## 1. Project overview
This repository implements a modular movie recommendation pipeline. It downloads MovieLens, validates interaction records, creates a chronological holdout split, builds user/item features, trains a popularity baseline and an item-item collaborative-filtering model, evaluates ranking and rating metrics, and records experiment results with MLflow.

## 2. Requirements
- Windows 10/11
- Python 3.10 or 3.11
- Internet connection for the initial MovieLens download

Node.js, fnm, Mermaid CLI, Airflow and a separate database server are **not required**.

## 3. Setup on a new Windows PC
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Run the complete pipeline
```powershell
python run_pipeline.py
```
On first execution the pipeline downloads the MovieLens `ml-latest-small` ZIP from GroupLens. Later runs reuse the extracted CSV files.

## 5. Outputs
- `data/raw/` – downloaded MovieLens source files
- `data/processed/` – clean interactions, chronological train/test data, user-item matrix and summaries
- `data/errors/invalid_ratings.csv` – rejected records with reasons
- `data/features/` – user, item and interaction feature tables
- `models/` – serialized popularity and item-item CF models
- `reports/validation_report.json` – validation summary
- `reports/feature_catalog.csv` – feature definitions
- `reports/model_comparison.csv` – comparison of both recommenders
- `reports/pipeline_run.txt` – pipeline execution log
- `mlflow.db` – local MLflow tracking database

## 6. Metrics and methodology
Ranking metrics use held-out interactions rated 4.0 or higher as relevant and K=5. RMSE and MAE are calculated using a transparent training-item-mean rating predictor with the global training mean as fallback; they are included as rating-prediction reference metrics and are not claimed to be model-specific CF estimates. Coverage is the fraction of catalog items recommended at least once. Diversity in this baseline implementation is a within-list uniqueness measure; interpret it as a simple diversity proxy, not a genre-distance measure.

## 7. Architecture
Open `architecture.png` directly. The editable Mermaid source is `architecture.mmd`. The PNG is included, so Mermaid CLI is unnecessary.

## 8. MLflow UI (optional)
After running the pipeline:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000`.

## 9. Reproducibility notes
The chronological split is global by timestamp: the earliest 80% of interactions are training data and the latest 20% are test data. The code computes metrics on the current downloaded dataset at execution time. Exact values may vary if the upstream MovieLens dataset changes.

## 10. Submission checklist
- [ ] Run `python run_pipeline.py` successfully on the submission PC
- [ ] Check the generated validation report and model comparison CSV
- [ ] Capture a screenshot of the successful terminal run
- [ ] Review the PDF report and architecture PNG
- [ ] Upload the repository and PDF report as required by the course portal
