# app/services/dashboard_main.py
from typing import List, Dict
import pandas as pd
from collections import defaultdict

# Example utility imports (you should define actual ones in util/data_loader.py)
from app.util.data_loader import load_events, load_matches
from app.util.ml_models import load_xt_model


def get_kpi_summary(events: pd.DataFrame, matches: pd.DataFrame) -> Dict:
    shots = events[events['type.name'] == 'Shot']
    passes = events[events['type.name'] == 'Pass']
    pressures = events[events['type.name'] == 'Pressure']

    goals = shots[shots['shot.outcome.name'] == 'Goal'].shape[0]
    xg = shots['shot.statsbomb_xg'].sum()

    assists = passes[passes['pass.assist'] == True].shape[0]
    xa = passes['pass.xa'].sum()

    total_pressures = pressures.shape[0]
    total_def_actions = events[events['team.name'] == 'Bayer 04 Leverkusen'].shape[0]
    ppda = total_def_actions / total_pressures if total_pressures else 0

    xga = events[(events['type.name'] == 'Shot') & (events['team.name'] != 'Bayer 04 Leverkusen')]['shot.statsbomb_xg'].sum()

    clean_sheets = matches[matches['opponent_goals'] == 0].shape[0]
    possession = matches['possession'].mean()  # Assume this field exists or calculate it

    set_piece_goals = shots[(shots['shot.set_piece'] != 'None') & (shots['shot.outcome.name'] == 'Goal')].shape[0]

    return {
        "goals": goals,
        "xg": round(xg, 2),
        "assists": assists,
        "xa": round(xa, 2),
        "ppda": round(ppda, 2),
        "xga": round(xga, 2),
        "clean_sheets": clean_sheets,
        "possession": round(possession, 2),
        "set_piece_goals": set_piece_goals
    }


def get_xg_goals_trend(events: pd.DataFrame, matches: pd.DataFrame) -> List[Dict]:
    result = []
    grouped = events[events['type.name'] == 'Shot'].groupby('match_id')

    for match_id, group in grouped:
        xg = group['shot.statsbomb_xg'].sum()
        goals = group[group['shot.outcome.name'] == 'Goal'].shape[0]
        date = matches[matches['match_id'] == match_id]['match_date'].values[0]
        result.append({"match_id": match_id, "date": date, "xg": round(xg, 2), "goals": goals})

    return result


def get_xt_zone_heatmap(events: pd.DataFrame, xt_model) -> List[Dict]:
    zone_data = defaultdict(float)
    for _, row in events.iterrows():
        x, y = row['location'][0], row['location'][1]
        xt_val = xt_model.get_value(x, y)
        zone_x, zone_y = int(x // 10), int(y // 10)
        zone_data[(zone_x, zone_y)] += xt_val

    return [{"zone": [k[0], k[1]], "xt": round(v, 3)} for k, v in zone_data.items()]


def get_top_player_contributions(events: pd.DataFrame, model: str = 'xt') -> List[Dict]:
    player_scores = defaultdict(float)
    if model == 'xt':
        for _, row in events.iterrows():
            if row['type.name'] in ['Pass', 'Carry']:
                player = row['player.name']
                x, y = row['location'][0], row['location'][1]
                xt_val = load_xt_model().get_value(x, y)
                player_scores[player] += xt_val

    sorted_scores = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"player_name": k, "value": round(v, 2), "metric": model} for k, v in sorted_scores]


def get_team_radar(events: pd.DataFrame, league_stats: Dict) -> Dict:
    # Example: Let's define a few metrics
    metrics = ['xG', 'Shots', 'xGA', 'PPDA']
    values = []

    xg = events[(events['team.name'] == 'Bayer 04 Leverkusen') & (events['type.name'] == 'Shot')]['shot.statsbomb_xg'].sum()
    shots = events[(events['team.name'] == 'Bayer 04 Leverkusen') & (events['type.name'] == 'Shot')].shape[0]
    xga = events[(events['team.name'] != 'Bayer 04 Leverkusen') & (events['type.name'] == 'Shot')]['shot.statsbomb_xg'].sum()
    ppda = events[events['type.name'] == 'Pressure'].shape[0] / events[events['team.name'] == 'Bayer 04 Leverkusen'].shape[0]

    team_values = [xg, shots, xga, ppda]

    for i, m in enumerate(metrics):
        mean = league_stats[m]['mean']
        std = league_stats[m]['std']
        val = team_values[i]
        zscore = (val - mean) / std
        percentile = min(max(0, (zscore + 3) / 6), 1)  # Rough normalization between 0–1
        values.append(round(percentile, 2))

    return {"metrics": metrics, "values": values}


def get_possession_flow(events: pd.DataFrame) -> List[Dict]:
    # Dummy logic for possession phases, replace with actual chaining logic
    flow = [
        {"source": "Defensive Third", "target": "Midfield", "value": 182},
        {"source": "Midfield", "target": "Final Third", "value": 145},
        {"source": "Final Third", "target": "Shot", "value": 86},
    ]
    return flow
