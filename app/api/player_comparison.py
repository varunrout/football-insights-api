from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
from app.services.metric_calculator import calculate_xt_added
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/radar")
async def get_player_radar_comparison(
    player_ids: List[int] = Query(..., description="Player IDs to compare"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    metrics: Optional[List[str]] = Query(None, description="Specific metrics to include"),
    normalized: bool = Query(True, description="Whether to normalize the metrics"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get radar chart data for player comparison
    
    Returns normalized metrics for multiple players in a format suitable for radar charts
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if not metrics:
            metrics = [
                "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90", 
                "progressive_passes_per_90", "successful_dribbles_per_90",
                "defensive_actions_per_90", "pressures_per_90"
            ]
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        else:
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
            if not all_events:
                return {"error": "No data found for competition/season."}
            events = pd.concat(all_events)
        if events is None or events.empty:
            return {"error": "No data found for competition/season/team."}
        player_data = []
        for player_id in player_ids:
            pe = events[events['player_id'] == player_id]
            minutes = pe['minute'].sum()
            # Calculate metrics
            goals = len(pe[(pe['type_name'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
            assists = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            xg = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
            # For advanced metrics, use pass/carry logic as needed
            progressive_passes = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            successful_dribbles = len(pe[pe['type_name'] == 'Carry'])  # Placeholder
            defensive_actions = len(pe[pe['type_name'].isin(['Duel', 'Interception', 'Tackle', 'Block'])])
            pressures = len(pe[pe['type_name'] == 'Pressure'])
            # Per 90
            per_90 = lambda v: v / (minutes / 90) if minutes > 0 else 0
            player_metrics = {
                "goals_per_90": per_90(goals),
                "assists_per_90": per_90(assists),
                "xg_per_90": per_90(xg),
                "xa_per_90": 0,  # Placeholder
                "progressive_passes_per_90": per_90(progressive_passes),
                "successful_dribbles_per_90": per_90(successful_dribbles),
                "defensive_actions_per_90": per_90(defensive_actions),
                "pressures_per_90": per_90(pressures)
            }
            player_data.append({
                "player_id": player_id,
                "player_name": pe.iloc[0]["player_name"] if not pe.empty and "player_name" in pe.columns else f"Player {player_id}",
                "team": pe.iloc[0]["team_name"] if not pe.empty and "team_name" in pe.columns else None,
                "metrics": {k: player_metrics[k] for k in metrics}
            })
        # Normalization
        metric_ranges = {}
        for metric in metrics:
            values = [p["metrics"].get(metric, 0) for p in player_data]
            metric_ranges[metric] = {
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            if normalized and metric_ranges[metric]["max"] > metric_ranges[metric]["min"]:
                for p in player_data:
                    val = p["metrics"][metric]
                    p["metrics"][metric] = (val - metric_ranges[metric]["min"]) / (metric_ranges[metric]["max"] - metric_ranges[metric]["min"])
        for p in player_data:
            p["player_id"] = int(p["player_id"])
            if p["team"] is not None:
                p["team"] = str(p["team"])
            if "metrics" in p:
                for k, v in p["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        p["metrics"][k] = float(v)
        for k, v in metric_ranges.items():
            metric_ranges[k]["min"] = float(v["min"])
            metric_ranges[k]["max"] = float(v["max"])
        return {
            "players": player_data,
            "metrics": metrics,
            "metric_ranges": metric_ranges,
            "normalized": normalized
        }
    except Exception as e:
        logger.error(f"Error getting player radar comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/bar-chart")
async def get_player_bar_comparison(
    player_ids: List[int] = Query(..., description="Player IDs to compare"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    metric: str = Query(..., description="Metric to compare"),
    per_90: bool = Query(True, description="Whether to show per 90 minutes values"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get bar chart data for player comparison
    
    Returns data for a single metric across multiple players
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        else:
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
            if not all_events:
                return {"error": "No data found for competition/season."}
            events = pd.concat(all_events)
        if events is None or events.empty:
            return {"error": "No data found for competition/season/team."}
        player_data = []
        for player_id in player_ids:
            pe = events[events['player_id'] == player_id]
            minutes = pe['minute'].sum()
            # Calculate metric value
            if metric == "goals":
                value = len(pe[(pe['type_name'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
            elif metric == "assists":
                value = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            elif metric == "xg":
                value = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
            elif metric == "xa":
                value = 0  # Placeholder
            else:
                value = len(pe)
            per_90_value = value / (minutes / 90) if per_90 and minutes > 0 else value
            player_data.append({
                "player_id": player_id,
                "player_name": pe.iloc[0]["player_name"] if not pe.empty and "player_name" in pe.columns else f"Player {player_id}",
                "team": pe.iloc[0]["team_name"] if not pe.empty and "team_name" in pe.columns else None,
                "value": value,
                "per_90_value": per_90_value,
                "minutes": minutes
            })
        for p in player_data:
            p["player_id"] = int(p["player_id"])
            if p["team"] is not None:
                p["team"] = str(p["team"])
            if "value" in p and isinstance(p["value"], (np.integer, np.floating)):
                p["value"] = float(p["value"])
            if "per_90_value" in p and isinstance(p["per_90_value"], (np.integer, np.floating)):
                p["per_90_value"] = float(p["per_90_value"])
            if "minutes" in p and isinstance(p["minutes"], (np.integer, np.floating)):
                p["minutes"] = float(p["minutes"])
        return {
            "players": player_data,
            "metric": metric,
            "per_90": per_90
        }
    except Exception as e:
        logger.error(f"Error getting player bar comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/scatter-plot")
async def get_player_scatter_comparison(
    competition_id: int = Query(..., description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    x_metric: str = Query(..., description="Metric for x-axis"),
    y_metric: str = Query(..., description="Metric for y-axis"),
    min_minutes: int = Query(450, description="Minimum minutes played"),
    position_group: Optional[str] = Query(None, description="Filter by position group"),
    highlighted_player_ids: Optional[List[int]] = Query(None, description="Players to highlight"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get scatter plot data for player comparison
    
    Returns data for comparing all players in a competition across two metrics
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        else:
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
            if not all_events:
                return {"error": "No data found for competition/season."}
            events = pd.concat(all_events)
        if events is None or events.empty:
            return {"error": "No data found for competition/season/team."}
        # Group by player
        players = []
        for player_id, pe in events.groupby('player_id'):
            minutes = pe['minute'].sum()
            if minutes < min_minutes:
                continue
            if position_group and pe.iloc[0].get('position', None) != position_group:
                continue
            # Calculate metrics
            def get_metric(metric):
                if metric == "goals":
                    return len(pe[(pe['type_name'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
                elif metric == "assists":
                    return len(pe[pe['type_name'] == 'Pass'])  # Placeholder
                elif metric == "xg":
                    return pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
                elif metric == "xa":
                    return 0  # Placeholder
                else:
                    return len(pe)
            x_value = get_metric(x_metric)
            y_value = get_metric(y_metric)
            is_highlighted = highlighted_player_ids and player_id in highlighted_player_ids
            players.append({
                "player_id": player_id,
                "player_name": pe.iloc[0]["player_name"] if not pe.empty and "player_name" in pe.columns else f"Player {player_id}",
                "team": pe.iloc[0]["team_name"] if not pe.empty and "team_name" in pe.columns else None,
                "position_group": pe.iloc[0].get('position', None),
                "x_value": x_value,
                "y_value": y_value,
                "minutes": minutes,
                "highlighted": is_highlighted
            })
        for p in players:
            p["player_id"] = int(p["player_id"])
            if p["team"] is not None:
                p["team"] = str(p["team"])
            if "x_value" in p and isinstance(p["x_value"], (np.integer, np.floating)):
                p["x_value"] = float(p["x_value"])
            if "y_value" in p and isinstance(p["y_value"], (np.integer, np.floating)):
                p["y_value"] = float(p["y_value"])
            if "minutes" in p and isinstance(p["minutes"], (np.integer, np.floating)):
                p["minutes"] = float(p["minutes"])
            if "highlighted" in p:
                p["highlighted"] = bool(p["highlighted"])
        # Compute averages safely
        if players:
            x_avg = float(np.mean([p["x_value"] for p in players]))
            y_avg = float(np.mean([p["y_value"] for p in players]))
        else:
            x_avg = 0.0
            y_avg = 0.0
        return {
            "players": players,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "x_average": x_avg,
            "y_average": y_avg,
            "min_minutes": min_minutes,
            "position_group": position_group
        }
    except Exception as e:
        logger.error(f"Error getting player scatter comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/similarity-map")
async def get_player_similarity_map(
    player_id: int = Query(..., description="Reference player ID"),
    competition_id: int = Query(..., description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    min_minutes: int = Query(450, description="Minimum minutes played"),
    limit: int = Query(10, description="Number of similar players to return"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player similarity map data
    Returns a list of the most similar players to the reference player
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        else:
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
            if not all_events:
                return {"error": "No data found for competition/season."}
            events = pd.concat(all_events)
        if events is None or events.empty:
            return {"error": "No data found for competition/season/team."}
        # Calculate per-90 metrics for all players
        metrics = [
            "goals_per_90", "assists_per_90", "xg_per_90", "progressive_passes_per_90",
            "successful_dribbles_per_90", "defensive_actions_per_90", "pressures_per_90"
        ]
        player_vectors = {}
        for pid, pe in events.groupby('player_id'):
            minutes = pe['minute'].sum()
            if minutes < min_minutes:
                continue
            goals = len(pe[(pe['type_name'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
            assists = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            xg = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
            progressive_passes = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            successful_dribbles = len(pe[pe['type_name'] == 'Carry'])  # Placeholder
            defensive_actions = len(pe[pe['type_name'].isin(['Duel', 'Interception', 'Tackle', 'Block'])])
            pressures = len(pe[pe['type_name'] == 'Pressure'])
            per_90 = lambda v: v / (minutes / 90) if minutes > 0 else 0
            player_vectors[pid] = np.array([
                per_90(goals),
                per_90(assists),
                per_90(xg),
                per_90(progressive_passes),
                per_90(successful_dribbles),
                per_90(defensive_actions),
                per_90(pressures)
            ])
        # Reference player vector
        ref_vector = player_vectors.get(player_id)
        if ref_vector is None:
            return {"error": "Reference player not found or insufficient minutes."}
        # Calculate similarity (cosine similarity)
        def cosine_similarity(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else 0
        similarities = []
        for pid, vec in player_vectors.items():
            if pid == player_id:
                continue
            sim = cosine_similarity(ref_vector, vec)
            similarities.append((pid, sim))
        similarities.sort(key=lambda x: -x[1])
        similar_players = []
        for pid, sim in similarities[:limit]:
            pe = events[events['player_id'] == pid]
            similar_players.append({
                "player_id": pid,
                "player_name": pe.iloc[0]["player_name"] if not pe.empty and "player_name" in pe.columns else f"Player {pid}",
                "team": pe.iloc[0]["team_name"] if not pe.empty and "team_name" in pe.columns else None,
                "position": pe.iloc[0].get('position', None),
                "minutes": pe['minute'].sum(),
                "similarity_score": sim,
                "key_metrics": {m: float(player_vectors[pid][i]) for i, m in enumerate(metrics)}
            })
        pe_ref = events[events['player_id'] == player_id]
        reference_player = {
            "player_id": player_id,
            "player_name": pe_ref.iloc[0]["player_name"] if not pe_ref.empty and "player_name" in pe_ref.columns else f"Player {player_id}",
            "team": pe_ref.iloc[0]["team_name"] if not pe_ref.empty and "team_name" in pe_ref.columns else None,
            "position": pe_ref.iloc[0].get('position', None),
            "minutes": pe_ref['minute'].sum(),
            "key_metrics": {m: float(ref_vector[i]) for i, m in enumerate(metrics)}
        }
        for p in similar_players:
            p["player_id"] = int(p["player_id"])
            if p["team"] is not None:
                p["team"] = str(p["team"])
            if "minutes" in p and isinstance(p["minutes"], (np.integer, np.floating)):
                p["minutes"] = float(p["minutes"])
            if "similarity_score" in p and isinstance(p["similarity_score"], (np.integer, np.floating)):
                p["similarity_score"] = float(p["similarity_score"])
            if "key_metrics" in p:
                for k, v in p["key_metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        p["key_metrics"][k] = float(v)
        if reference_player:
            reference_player["player_id"] = int(reference_player["player_id"])
            if reference_player["team"] is not None:
                reference_player["team"] = str(reference_player["team"])
            if "minutes" in reference_player and isinstance(reference_player["minutes"], (np.integer, np.floating)):
                reference_player["minutes"] = float(reference_player["minutes"])
            if "key_metrics" in reference_player:
                for k, v in reference_player["key_metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        reference_player["key_metrics"][k] = float(v)
        return {
            "reference_player": reference_player,
            "similar_players": similar_players,
            "competition_id": competition_id,
            "season_id": season_id,
            "min_minutes": min_minutes
        }
    except Exception as e:
        logger.error(f"Error getting player similarity map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
