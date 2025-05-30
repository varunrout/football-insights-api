#!/usr/bin/env python3
import requests
import pandas as pd
import json

# Base URL for your running FastAPI app
BASE_URL = "http://localhost:8000/api/v1"

# Define each endpoint and the query parameters you want to test it with
ENDPOINTS = {
    # Dashboard
    "dashboard_summary": {
        "path": "/dashboard/summary",
        "params": {"competition_id": 1, "team_id": 42, "season_id": 2024}
    },
    "xg_timeline": {
        "path": "/dashboard/xg-timeline",
        "params": {"competition_id": 1, "team_id": 42, "season_id": 2024}
    },
    "shot_map": {
        "path": "/dashboard/shot-map",
        "params": {"competition_id": 1, "team_id": 42}
    },

    # Player Analysis
    "player_profile": {
        "path": "/player-analysis/profile",
        "params": {"player_id": 9, "competition_id": 1, "season_id": 2024}
    },
    "performance_trend": {
        "path": "/player-analysis/performance-trend",
        "params": {"player_id": 9, "metric": "goals", "timeframe": "last5", "competition_id": 1}
    },
    "event_map": {
        "path": "/player-analysis/event-map",
        "params": {"player_id": 9, "event_type": "shots", "competition_id": 1}
    },

    # Player Comparison
    "radar": {
        "path": "/player-comparison/radar",
        "params": {"player_ids": [9, 10], "competition_id": 1, "season_id": 2024}
    },
    "bar_chart": {
        "path": "/player-comparison/bar-chart",
        "params": {"player_ids": [9, 10], "competition_id": 1, "season_id": 2024, "metric": "goals"}
    },
    "scatter_plot": {
        "path": "/player-comparison/scatter-plot",
        "params": {"competition_id": 1, "season_id": 2024, "x_metric": "xg", "y_metric": "shots"}
    },
    "similarity_map": {
        "path": "/player-comparison/similarity-map",
        "params": {"player_id": 9, "competition_id": 1}
    },

    # Positional Analysis
    "role_analysis": {
        "path": "/positional-analysis/role-analysis",
        "params": {"position": "Midfielder", "competition_id": 1, "season_id": 2024}
    },
    "zone_analysis": {
        "path": "/positional-analysis/zone-analysis",
        "params": {"team_id": 42, "competition_id": 1, "zone_type": "grid"}
    },
    "heat_map": {
        "path": "/positional-analysis/heat-map",
        "params": {"player_id": 9, "competition_id": 1, "event_type": "Pass"}
    },

    # xT Analytics
    "xt_model": {
        "path": "/xt-analytics/model",
        "params": {}
    },
    "player_xt_rankings": {
        "path": "/xt-analytics/player-rankings",
        "params": {"competition_id": 1}
    },
    "xt_pass_map": {
        "path": "/xt-analytics/pass-map",
        "params": {"match_id": 1001}
    },
    "team_xt_contribution": {
        "path": "/xt-analytics/team-contribution",
        "params": {"competition_id": 1, "team_id": 42}
    },

    # Tactical Insights
    "defensive_metrics": {
        "path": "/tactical-insights/defensive-metrics",
        "params": {"team_id": 42, "competition_id": 1}
    },
    "offensive_metrics": {
        "path": "/tactical-insights/offensive-metrics",
        "params": {"team_id": 42, "competition_id": 1}
    },
    "pass_network": {
        "path": "/tactical-insights/pass-network",
        "params": {"team_id": 42, "match_id": 1001}
    },

    # Matchup Analysis
    "head_to_head": {
        "path": "/matchup-analysis/head-to-head",
        "params": {"team1_id": 42, "team2_id": 55, "competition_id": 1}
    },
    "team_style": {
        "path": "/matchup-analysis/team-style",
        "params": {"team_id": 42, "competition_id": 1}
    },
    "matchup_prediction": {
        "path": "/matchup-analysis/matchup-prediction",
        "params": {"team1_id": 42, "team2_id": 55, "competition_id": 1}
    },
}


def fetch_all(endpoints: dict) -> pd.DataFrame:
    records = []

    for name, info in endpoints.items():
        url = BASE_URL + info["path"]
        try:
            resp = requests.get(url, params=info["params"], timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            data = {"error": str(e)}

        # Flatten JSON into one-level dict
        flat = pd.json_normalize(data)
        flat["endpoint"] = name
        flat["url"] = url

        # Serialize each param as a JSON string to avoid length mismatches
        for k, v in info["params"].items():
            flat[f"param_{k}"] = json.dumps(v)

        records.append(flat)

    # Concatenate all into one DataFrame
    return pd.concat(records, ignore_index=True, sort=False)


if __name__ == "__main__":
    df = fetch_all(ENDPOINTS)
    print(df.head(20))
    df.to_csv("api_responses.csv", index=False)