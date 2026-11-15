from pathlib import Path
import shutil
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from api.app import app, MODEL_PATH
from test_inference import sample_customer

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_real_model_batch(client):
    assert client.get("/health").json()["model_loaded"] is True
    customer = sample_customer()
    response = client.post("/predict", json={"instances": [customer, {**customer, "tenure": 1}]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["churn_probability"]) == 2
    assert set(data["churn_prediction"]) <= {0, 1}
    assert all(0 <= value <= 1 for value in data["churn_probability"])
    assert MODEL_PATH.is_file()

@pytest.mark.parametrize("kind", ["missing", "null"])
def test_each_record_must_have_complete_features(client, kind):
    customer = sample_customer()
    invalid = customer.copy()
    if kind == "missing":
        invalid.pop("tenure")
    else:
        invalid["tenure"] = None
    response = client.post("/predict", json={"instances": [customer, invalid]})
    assert response.status_code == 400
    assert response.json()["detail"]["instance"] == 1

def test_empty_batch(client):
    assert client.post("/predict", json={"instances": []}).status_code == 400

def test_ui_bundle_predicts_via_configured_api(client, monkeypatch, tmp_path):
    # Run with only the files copied by the frontend Dockerfile.
    for filename in ["app.py", "data_schema.json"]:
        shutil.copy(ROOT / "streamlit" / filename, tmp_path / filename)
    monkeypatch.setenv("API_URL", "http://configured-api:8000/")
    calls = []
    def post(url, *, json, timeout):
        calls.append((url, json))
        return client.post("/predict", json=json)
    monkeypatch.setattr("requests.post", post)
    ui = AppTest.from_file(str(tmp_path / "app.py")).run(timeout=30)
    assert not ui.exception
    assert len(ui.number_input) + len(ui.selectbox) == 19
    ui.button[0].click().run(timeout=30)
    assert not ui.exception
    assert calls[0][0] == "http://configured-api:8000/predict"
    assert len(ui.metric) == 2
    assert ui.success
