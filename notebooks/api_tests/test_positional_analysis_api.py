import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_role_analysis_forward():
    response = client.get(
        "/positional-analysis/role-analysis",
        params={
            "position": "Forward",
            "competition_id": 43,
            "season_id": 2022,
            "min_minutes": 450
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "players" in data
    assert data["position"] == "Forward"
    assert isinstance(data["players"], list)
    # Check that at least one player has real per-90 metrics
    if data["players"]:
        player = data["players"][0]
        assert "key_metrics" in player
        assert player["key_metrics"]["goals_per_90"] >= 0
        assert player["minutes"] >= 450
