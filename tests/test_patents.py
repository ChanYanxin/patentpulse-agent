"""Patents-specific tests use synthetic inputs and do not validate legal novelty."""
import json
import pytest
from fastapi.testclient import TestClient
from techintel.app import app, get_agent
from techintel.patents import compare_claims, parse_claims, search_local_patents, search_patentsview

TARGET = """1. A robotic gripper comprising a tactile sensor and a torque-limited actuator.
2. The gripper of claim 1, wherein a feedback controller detects slip.
3. The gripper of claim 2, wherein an inertial sensor measures acceleration."""
REF = """1. A robotic gripper comprising a tactile sensor and a torque-limited actuator.
2. The robotic gripper of claim 1, wherein the feedback controller adjusts force based on slip."""


def test_parse_claim_graph():
    result = parse_claims(TARGET)
    assert result["claim_count"] == 3
    assert result["claims"][0]["independent"]
    assert result["claims"][1]["depends_on"] == [1]
    assert result["claims"][2]["depends_on"] == [2]
    assert not result["warnings"]


def test_invalid_claims():
    with pytest.raises(ValueError, match="No numbered claims"):
        parse_claims("A claim with no numbering")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_claims("1. One\n1. Two")
    result = parse_claims("1. A device.\n2. A device of claim 99, with a battery.")
    assert "missing claim 99" in result["warnings"][0]


def test_claim_comparison_is_screening_only():
    result = compare_claims(TARGET, REF)
    assert len(result["claim_chart"]) == 3
    assert result["claim_chart"][0]["lexical_coverage"] == 1.0
    assert result["claim_chart"][2]["lexical_coverage"] < 1
    assert "No legal novelty" in result["disclaimer"]


def test_local_search_labels_synthetic():
    get_agent.cache_clear()
    result = search_local_patents(get_agent().index, "robot gripper tactile sensor torque", k=3)
    assert result["source_type"] == "SYNTHETIC_DEMO_NOT_REAL_PATENTS"
    assert any("gripper" in r["doc_id"] for r in result["results"])
    assert all(r["doc_id"].startswith("patent_demo_") for r in result["results"])


def test_patent_endpoints_and_validation():
    c = TestClient(app)
    resp = c.post("/patents/search", json={"query": "organic photovoltaic donor acceptor"})
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1
    assert c.post("/patents/claims/parse", json={"claims_text": TARGET}).json()["claim_count"] == 3
    assert c.post("/patents/claims/compare", json={"target_claims": TARGET, "reference_claims": REF}).status_code == 200
    assert c.post("/patents/claims/parse", json={"claims_text": "bad claims"}).status_code == 422


def test_live_connector_needs_credential(monkeypatch):
    monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        search_patentsview("battery")


def test_live_connector_request_structure_mock(monkeypatch):
    import techintel.patents as patents
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, *args): return json.dumps({"error": False, "patents": [
            {"patent_id": "123456", "patent_title": "Illustrative", "patent_date": "2020-01-01", "patent_abstract": "Solid state"}]}).encode()
    def fake_urlopen(req, timeout):
        assert req.full_url.endswith("/patent/")
        assert json.loads(req.data)["q"] == {"_text_any": {"patent_abstract": "solid state"}}
        assert timeout == 12
        assert req.get_header("X-api-key") == "MOCK_KEY"
        return FakeResponse()
    monkeypatch.setenv("PATENTSVIEW_API_KEY", "MOCK_KEY")
    monkeypatch.setattr(patents, "urlopen", fake_urlopen)
    resp = search_patentsview("solid state")
    assert resp["count"] == 1
    assert resp["results"][0]["patent_id"] == "123456"


def test_patent_eval_reproducible():
    from techintel.patent_eval import evaluate_patent_demo
    report = evaluate_patent_demo()
    assert report['n_documents'] == 4
    assert report['n_queries'] == 16
    assert set(report['methods']) == {'word', 'hybrid', 'bm25'}
