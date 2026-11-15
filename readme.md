# Telco Customer Churn — From SQL Data to a Prediction App

A portfolio project exploring which customer profiles are associated with churn. A saved preprocessing and LightGBM pipeline is served through FastAPI and an interactive Streamlit form. The goal is to demonstrate data preparation, model comparison, and reproducible application delivery.

## What is implemented

- SQLite customer/service/billing dimensions joined to churn observations with SQL.
- Four numeric and fifteen categorical features; scaling and one-hot encoding inside the fitted pipeline.
- Sixteen experiment configurations: logistic regression, random forest, XGBoost, and LightGBM, each with/without PCA and with/without Optuna tuning.
- Batch prediction API, interactive customer form, health endpoints, and separate Docker images.
- Optional MLflow/DagsHub tracking. Credentials are not needed to run inference.

## Run the saved model locally

Use Python 3.12. The deployment artifact was serialized with scikit-learn 1.8.0; the API requirements match that version. Only load model artifacts from sources you trust.

```bash
git clone https://github.com/DoddapaneniSatish/Telco-customer-churn.git
cd Telco-customer-churn
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python test_inference.py
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs to try a prediction using the included 19-feature example. The request is `{"instances": [{...customer features...}]}` and the response contains `churn_prediction`, `churn_probability`, and `count`. Class `1` means churn. Each record must contain all required features without null values; invalid types and out-of-domain values need further validation.

In another terminal, activate the same environment and launch the frontend:

```bash
API_URL=http://localhost:8000 python -m streamlit run streamlit/app.py
# Windows PowerShell:
# $env:API_URL = 'http://localhost:8000'
# python -m streamlit run streamlit/app.py
```

Open http://localhost:8501. Set `API_URL` to use a different backend. Without it, the UI preserves the existing hosted API as a fallback. An optional `MODEL_PATH` overrides the API artifact path; the default is `api/models/telco_churn_best_model.pkl` relative to the application file, independent of the working directory.

## Docker and Render

```bash
docker compose up --build
```

The UI is at http://localhost:8501 and the API documentation at http://localhost:8000/docs. Compose waits for the API health check and supplies the internal backend URL. Both images include their runtime model/schema files and do not require host data volumes.

`render.yaml` defines two Docker services. When creating the Blueprint, set the frontend's `API_URL` to the actual public URL assigned to the API service. Do not assume a hostname from the service name. For an existing deployment, update that environment variable in Render before switching to the corrected configuration. Hosting availability and limits depend on your Render account and plan.

## Recorded model comparison

The committed experiment files record the following results on a 1,409-row split of the 7,043-customer dataset:

| Configuration | Churn F1 |
| --- | ---: |
| Logistic regression, no PCA, untuned | 0.6136 |
| Random forest, no PCA, tuned | 0.6284 |
| XGBoost, no PCA, tuned | 0.5870 |
| LightGBM, no PCA, tuned (selected) | 0.6286 |

For the selected run, recorded churn precision is **0.5336**, recall **0.7647**, and accuracy **0.7601**. Confusion counts are TN 785, FP 250, FN 88, TP 286. These are historical repository results, not a claim of business impact or a new benchmark.

**Evaluation limitation:** all sixteen configurations were compared on the same outer split, and `src/select_best_model.py` selects the maximum F1 from that comparison. That split therefore participates in model selection; its winning score is not an independent final holdout estimate. A future evaluation should reserve an untouched test set, select models and thresholds using training/validation data, and measure calibration and subgroup errors before drawing deployment conclusions. The demo's model scores are not calibrated guarantees or causal explanations.

## Verify changes

```bash
python -m pip install -r requirements-dev.txt
python test_inference.py
python -m pytest tests -q
```

The tests exercise the actual saved model through the API, reject incomplete records within a batch, and run the Streamlit form against that API using only the files bundled into the frontend image. Docker builds and live Render deployment require separate verification in those environments.

## Source map

| Path | Purpose |
| --- | --- |
| `src/data_loader.py` | Load SQL joins and split features/target |
| `src/preprocess.py` | Numeric/categorical preprocessing |
| `src/experiments.py` | Experiment runner, PCA and tuning |
| `src/select_best_model.py` | Select recorded best run and copy it to root and API artifact paths |
| `metrics/` | Committed comparison and per-run reports |
| `api/app.py`, `api/models/` | Serving code and deployment artifact |
| `streamlit/app.py`, `streamlit/data_schema.json` | Customer form and bundled input defaults |
| `tests/`, `test_inference.py` | Serving integration and inference checks |

Training is separate from serving: the experiment runner additionally requires Optuna, and remote tracking additionally requires MLflow plus a separately configured DagsHub credential. Never commit credentials. Selecting a model updates both `models/telco_churn_best_model.pkl` and the copy used by the API image. Older housing-named scripts/artifacts remain as legacy files and are not part of the Telco serving path.

## Next improvements

1. Independent final holdout evaluation, threshold selection, and calibration analysis.
2. Typed feature validation, batch limits, and clearer user-facing failure handling.
3. An analysis of false positives/negatives and a reproducible comparison report with model provenance.
4. A portfolio case study showing the data flow, tradeoffs, personal contribution, and measured results.

## Project origin

This repository is a fork of [mkzia/housing_app_fall25](https://github.com/mkzia/housing_app_fall25). The inherited housing scaffold explains the legacy files. The Telco-specific data, experiments, and serving changes are visible in this fork’s commit history.
