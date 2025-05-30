import requests
import pandas as pd

BASE_URL = "http://localhost:8000/api/v1"

results = []

def get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params)
    try:
        data = response.json()
    except Exception:
        data = response.text
    print(f"GET {url} | Status: {response.status_code}")
    print(data)
    print("-" * 60)
    # Save result for DataFrame
    results.append({
        "endpoint": endpoint,
        "params": str(params),
        "status_code": response.status_code,
        "response": str(data)
    })
    return response

if __name__ == "__main__":
    # --- New selector endpoints ---
    print("Testing selector endpoints (new logic):")
    get("core-selectors/competitions")
    get("core-selectors/seasons", params={"competition_id": 1})
    get("core-selectors/teams", params={"competition_id": 1, "season_id": 2020})
    get("core-selectors/matches", params={"competition_id": 1, "season_id": 2020, "team_id": 42})
    get("core-selectors/players", params={"competition_id": 1, "season_id": 2020, "team_id": 42})

    # --- New xT analytics endpoints ---
    print("Testing xT analytics endpoints (new logic):")
    get("xt-analytics/model")
    get("xt-analytics/player-rankings", params={"competition_id": 1, "season_id": 2020, "team_id": 42})
    get("xt-analytics/pass-map", params={"match_id": 1001})
    get("xt-analytics/team-contribution", params={"competition_id": 1, "season_id": 2020, "team_id": 42})

    # --- New dashboard endpoints ---
    print("Testing dashboard endpoints (new logic):")
    get("dashboard/summary", params={"competition_id": 1, "season_id": 2020, "team_id": 42})
    get("dashboard/xg-timeline", params={"competition_id": 1, "season_id": 2020, "team_id": 42})
    get("dashboard/shot-map", params={"competition_id": 1, "season_id": 2020, "team_id": 42})

    # --- New player analysis endpoints ---
    print("Testing player analysis endpoints (new logic):")
    get("player-analysis/profile", params={"player_id": 9, "competition_id": 1, "season_id": 2020})
    get("player-analysis/performance-trend", params={"player_id": 9, "metric": "goals", "timeframe": "last5", "competition_id": 1, "season_id": 2020})
    get("player-analysis/event-map", params={"player_id": 9, "event_type": "shots", "competition_id": 1, "season_id": 2020})

    # --- New player comparison endpoints ---
    print("Testing player comparison endpoints (new logic):")
    get("player-comparison/radar", params={"player_ids": [9, 10], "competition_id": 1, "season_id": 2020})
    get("player-comparison/bar-chart", params={"player_ids": [9, 10], "competition_id": 1, "season_id": 2020, "metric": "goals"})
    get("player-comparison/scatter-plot", params={"competition_id": 1, "season_id": 2020, "x_metric": "xg", "y_metric": "shots"})
    get("player-comparison/similarity-map", params={"player_id": 9, "competition_id": 1, "season_id": 2020})

    # --- New tactical insights endpoints ---
    print("Testing tactical insights endpoints (new logic):")
    get("tactical-insights/defensive-metrics", params={"team_id": 42, "competition_id": 1, "season_id": 2020})
    get("tactical-insights/offensive-metrics", params={"team_id": 42, "competition_id": 1, "season_id": 2020})
    get("tactical-insights/pass-network", params={"team_id": 42, "match_id": 1001})

    # --- New matchup analysis endpoints ---
    print("Testing matchup analysis endpoints (new logic):")
    get("matchup-analysis/head-to-head", params={"team1_id": 42, "team2_id": 55, "competition_id": 1, "season_id": 2020})
    get("matchup-analysis/team-style", params={"team_id": 42, "competition_id": 1, "season_id": 2020})
    get("matchup-analysis/matchup-prediction", params={"team1_id": 42, "team2_id": 55, "competition_id": 1, "season_id": 2020})

    # --- Older logic: direct endpoints with hardcoded/test IDs ---
    print("Testing endpoints with older logic (direct/hardcoded IDs):")
    get("dashboard/summary", params={"competition_id": 2, "season_id": 3, "team_id": 217})
    get("dashboard/xg-timeline", params={"competition_id": 2, "season_id": 3, "team_id": 217})
    get("dashboard/shot-map", params={"competition_id": 2, "season_id": 3, "team_id": 217})

    get("player-analysis/profile", params={"player_id": 5246, "competition_id": 2, "season_id": 3})
    get("player-analysis/performance-trend", params={"player_id": 5246, "metric": "goals", "timeframe": "last5", "competition_id": 2, "season_id": 3})
    get("player-analysis/event-map", params={"player_id": 5246, "event_type": "shots", "competition_id": 2, "season_id": 3})

    get("player-comparison/radar", params={"player_ids": [5246, 302], "competition_id": 2, "season_id": 3})
    get("player-comparison/bar-chart", params={"player_ids": [5246, 302], "competition_id": 2, "season_id": 3, "metric": "goals"})
    get("player-comparison/scatter-plot", params={"competition_id": 2, "season_id": 3, "x_metric": "xg", "y_metric": "shots"})
    get("player-comparison/similarity-map", params={"player_id": 5246, "competition_id": 2, "season_id": 3})

    get("tactical-insights/defensive-metrics", params={"team_id": 217, "competition_id": 2, "season_id": 3})
    get("tactical-insights/offensive-metrics", params={"team_id": 217, "competition_id": 2, "season_id": 3})
    get("tactical-insights/pass-network", params={"team_id": 217, "match_id": 8658})

    get("matchup-analysis/head-to-head", params={"team1_id": 217, "team2_id": 218, "competition_id": 2, "season_id": 3})
    get("matchup-analysis/team-style", params={"team_id": 217, "competition_id": 2, "season_id": 3})
    get("matchup-analysis/matchup-prediction", params={"team1_id": 217, "team2_id": 218, "competition_id": 2, "season_id": 3})

    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv("api_responses.csv", index=False)
    print("All API responses saved to api_responses.csv")