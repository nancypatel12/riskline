import pytest
from fastapi.testclient import TestClient

from riskline.schema import FEATURES
from riskline.serve import MAX_BATCH, create_app
from tests.conftest import synthetic


@pytest.fixture(scope="module")
def client(registry):
    with TestClient(create_app(registry=str(registry.root))) as c:
        yield c


@pytest.fixture
def applicant():
    return {k: (int(v) if float(v).is_integer() else float(v)) for k, v in synthetic(1).iloc[0][FEATURES].items()}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}


def test_predict(client, applicant):
    r = client.post("/predict", json=applicant)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["high_risk"] == (body["probability"] >= body["threshold"])
    assert float(r.headers["X-Latency-Ms"]) > 0


def test_batch_matches_single(client, applicant):
    single = client.post("/predict", json=applicant).json()
    batch = client.post("/predict/batch", json={"applicants": [applicant] * 3}).json()
    assert len(batch["predictions"]) == 3
    assert batch["predictions"][0]["probability"] == pytest.approx(single["probability"])


@pytest.mark.parametrize(
    "patch",
    [{"education": 9}, {"pay_1": 10}, {"limit_bal": -1}, {"pay_amt1": -5}, {"unknown": 1}, {"pay_1": "late"}],
)
def test_rejects_invalid_input(client, applicant, patch):
    assert client.post("/predict", json=applicant | patch).status_code == 422


def test_rejects_missing_field(client, applicant):
    applicant.pop("limit_bal")
    assert client.post("/predict", json=applicant).status_code == 422


@pytest.mark.parametrize("field", [{"sex": 2}, {"age": 35}, {"marriage": 1}])
def test_rejects_protected_attributes(client, applicant, field):
    assert client.post("/predict", json=applicant | field).status_code == 422


@pytest.mark.parametrize("n", [0, MAX_BATCH + 1])
def test_batch_size_limits(client, applicant, n):
    assert client.post("/predict/batch", json={"applicants": [applicant] * n}).status_code == 422


def test_model_card(client):
    card = client.get("/model").json()
    assert card["version"] == "v1"
    assert 0 < card["threshold"] < 1
