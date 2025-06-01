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
    get("core-selectors/seasons", params={"competition_id": 9})
    get("core-selectors/teams", params={"competition_id": 9, "season_id": 281, "team_id": 904})
    get("core-selectors/matches", params={"competition_id": 9, "season_id": 281})
    get("core-selectors/players", params={"competition_id": 9, "season_id": 281, "team_id": 904})

    # --- New xT analytics endpoints ---
    print("Testing xT analytics endpoints (new logic):")
    get("xt-analytics/model")
    get("xt-analytics/player-rankings", params={"competition_id": 9, "season_id": 281, "team_id": 904})
    get("xt-analytics/pass-map", params={"match_id": 3895302})
    get("xt-analytics/team-contribution", params={"competition_id": 9, "season_id": 281, "team_id": 904})

    # --- New dashboard endpoints ---
    print("Testing dashboard endpoints (new logic):")
    get("dashboard/summary", params={"competition_id": 9, "season_id": 281, "team_id": 904})
    get("dashboard/xg-timeline", params={"competition_id": 9, "season_id": 281, "team_id": 904})
    get("dashboard/shot-map", params={"competition_id": 9, "season_id": 281, "team_id": 904})

    # --- New player analysis endpoints ---
    print("Testing player analysis endpoints (new logic):")
    get("player-analysis/profile", params={"player_id": 40724, "competition_id": 9, "season_id": 281})
    get("player-analysis/performance-trend", params={"player_id": 32712, "metric": "goals", "timeframe": "last5", "competition_id": 9, "season_id": 281})
    get("player-analysis/event-map", params={"player_id": 32712, "event_type": "shots", "competition_id": 9, "season_id": 281})

    # --- New player comparison endpoints ---
    print("Testing player comparison endpoints (new logic):")
    get("player-comparison/radar", params={"player_ids": [40724, 32712, 38004], "competition_id": 9, "season_id": 281})
    get("player-comparison/bar-chart", params={"player_ids": [40724, 32712, 38004], "competition_id": 9, "season_id": 281, "metric": "goals"})
    get("player-comparison/scatter-plot", params={"competition_id": 9, "season_id": 281, "x_metric": "xg", "y_metric": "shots"})
    get("player-comparison/similarity-map", params={"player_id": 38004, "competition_id": 9, "season_id": 281})

    # --- New tactical insights endpoints ---
    print("Testing tactical insights endpoints (new logic):")
    get("tactical-insights/defensive-metrics", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/offensive-metrics", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/pass-network", params={"team_id": 904, "match_id": 3895302})
    get("tactical-insights/build-up-analysis", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/pressing-analysis", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/transitions", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/set-pieces", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/formation-analysis", params={"team_id": 904, "match_id": 3895302})
    get("tactical-insights/team-style", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("tactical-insights/style-comparison", params={"team_id1": 904, "team_id2": 169, "competition_id": 9, "season_id": 281})

    # --- New matchup analysis endpoints ---
    print("Testing matchup analysis endpoints (new logic):")
    get("matchup-analysis/head-to-head", params={"team1_id": 904, "team2_id": 169, "competition_id": 9, "season_id": 281})
    get("matchup-analysis/team-style", params={"team_id": 904, "competition_id": 9, "season_id": 281})
    get("matchup-analysis/matchup-prediction", params={"team1_id": 904, "team2_id": 169, "competition_id": 9, "season_id": 281})


    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv("api_responses.csv", index=False)
    print("All API responses saved to api_responses.csv")