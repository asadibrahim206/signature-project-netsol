from fastapi.testclient import TestClient

from signature_project.main import app

client = TestClient(app)


def test_home_returns_running_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Signature API is running"


def test_home_lists_endpoints():
    response = client.get("/")

    endpoints = response.json()["endpoints"]
    assert "detection" in endpoints
    assert "verification" in endpoints
