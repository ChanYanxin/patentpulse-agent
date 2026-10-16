from fastapi.testclient import TestClient
from techintel.app import app, get_agent
from techintel.patents import parse_claims
from techintel.real_patent_eval import evaluate_real_patent_notes


def test_dependency_lists_and_ranges():
    claims = '''1. A device with a sensor.
2. A device of claim 1 with a controller.
3. The device of claim 1 or 2 with a battery.
4. The device of any one of claims 1 to 3 with a camera.
5. The device of claims 1, 2 and 4 with a recorder.'''
    parsed = parse_claims(claims)
    assert parsed["claims"][2]["depends_on"] == [1, 2]
    assert parsed["claims"][3]["depends_on"] == [1, 2, 3]
    assert parsed["claims"][4]["depends_on"] == [1, 2, 4]
    assert not parsed["warnings"]


def test_claim_bad_range():
    import pytest
    with pytest.raises(ValueError, match="descending"):
        parse_claims("1. A sensor.\n2. The sensor of claims 3 to 1.")


def test_curated_endpoint_provenance_and_source_ids():
    get_agent.cache_clear()
    response = TestClient(app).post("/patents/search", json={"query": "tactile gripper finger sensor",
                                      "source": "curated", "k": 3, "retrieval": "bm25"})
    assert response.status_code == 200
    value = response.json()
    assert value["source_type"] == "REAL_PATENT_CURATED_PARAPHRASE_NOT_FULL_TEXT"
    assert value["count"] > 0
    assert all(hit["doc_id"].startswith("patent_real_note_") for hit in value["results"])


def test_curated_eval_is_reproducible():
    report = evaluate_real_patent_notes()
    assert report["n_documents"] == 3
    assert report["n_queries"] == 13
    assert set(report["methods"]) == {"word", "hybrid", "bm25"}
    assert report["ablation_without_evidence_gate"]["bm25"]["ood_correct_rejection"] == 0.0
    assert report["methods"]["bm25"]["ood_correct_rejection"] == 1.0
