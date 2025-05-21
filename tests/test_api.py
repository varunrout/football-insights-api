# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Welcome to Football Insights Lab API"}

@pytest.mark.parametrize("path,expected_min", [
    ("/api/teams", 1),
    ("/api/players", 1),
    ("/api/shot_events", 1),
    ("/api/team_performance", 1),
    ("/api/player_stats", 1),
])
def test_list_endpoints(path, expected_min):
    resp = client.get(path)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= expected_min

def test_get_team_by_id():
    # assumes team with id=1 exists in your StatsBomb load
    resp = client.get("/api/teams/1")
    assert resp.status_code == 200
    team = resp.json()
    assert team["id"] == 1
    assert "name" in team

def test_404():
    resp = client.get("/api/teams/999999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Team not found"
