from fastapi.testclient import TestClient
from techintel.app import app


def test_demo_ui_loads_and_uses_safe_text_insertion():
    response = TestClient(app).get('/demo')
    assert response.status_code == 200
    assert 'PatentPulse-Agent' in response.text
    assert 'textContent' in response.text
    assert "'/patents/claims/parse'" in response.text
    assert '<script>' in response.text
