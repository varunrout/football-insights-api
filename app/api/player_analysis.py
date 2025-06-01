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

@router.get("/profile")
async def get_player_profile(
    player_id: int = Query(..., description="Player ID"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get comprehensive player profile
    
    Returns player info, key metrics, and performance summary
    """
    try:
        # Validate selectors
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        # Use team_id if provided for faster lookup
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
            if events is None or events.empty:
                return {"error": "No data found for team in this competition/season."}
            player_events = events[events['player_id'] == player_id]
        else:
            # Fallback: aggregate all events for the competition/season
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = []
            for _, match in matches_df.iterrows():
                ev = fdm.get_events(match['match_id'])
                if player_id in ev['player_id'].unique():
                    all_events.append(ev)
            if not all_events:
                return {"error": "No data found for player."}
            player_events = pd.concat(all_events)
        if player_events.empty:
            return {"error": "No data found for player in this context."}
        
        # Basic info (fallbacks if not available)
        first_event = player_events.iloc[0]
        player_info = {
            "player_id": player_id,
            "name": first_event.get("player", f"Player {player_id}"),
            "team": first_event.get("team", "Unknown"),
            "position": first_event.get("position", "Unknown"),
        }
        
        # Playing time
        minutes_played = player_events['minute'].sum()  # Approximation
        matches_played = player_events['match_id'].nunique()
        playing_time = {
            "matches_played": matches_played,
            "minutes_played": minutes_played,
            "minutes_per_match": minutes_played / matches_played if matches_played > 0 else 0
        }
        
        # Performance metrics
        goals = len(player_events[(player_events['type_name'] == 'Shot') & (player_events['shot_outcome'] == 'Goal')])
        assists = len(player_events[player_events['type_name'] == 'Pass'])  # Placeholder: refine with assist logic
        xg = player_events['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in player_events else 0
        shots = len(player_events[player_events['type_name'] == 'Shot'])
        passes_completed = len(player_events[(player_events['type_name'] == 'Pass') & (player_events['pass_outcome'].isna())])
        pass_accuracy = passes_completed / max(1, len(player_events[player_events['type_name'] == 'Pass'])) * 100
        performance_metrics = {
            "goals": goals,
            "assists": assists,
            "xg": xg,
            "shots": shots,
            "passes_completed": passes_completed,
            "pass_accuracy": pass_accuracy
        }
        
        # Per 90 metrics
        per_90_metrics = {k + '_per_90': v / (minutes_played / 90) if minutes_played > 0 else 0 for k, v in performance_metrics.items()}
        
        # Form (last 5 matches)
        form_data = []
        for match_id in player_events['match_id'].drop_duplicates().tail(5):
            match_events = player_events[player_events['match_id'] == match_id]
            form_data.append({
                "match_id": match_id,
                "goals": len(match_events[(match_events['type_name'] == 'Shot') & (match_events['shot_outcome'] == 'Goal')]),
                "shots": len(match_events[match_events['type_name'] == 'Shot']),
                "passes_completed": len(match_events[(match_events['type_name'] == 'Pass') & (match_events['pass_outcome'].isna())]),
                "minutes_played": match_events['minute'].sum()
            })
        
        # Convert numpy types to native Python types for JSON serialization
        player_info["player_id"] = int(player_info["player_id"])
        if isinstance(minutes_played, (np.integer, np.floating)):
            minutes_played = float(minutes_played)
        if isinstance(matches_played, (np.integer, np.floating)):
            matches_played = int(matches_played)
        playing_time["matches_played"] = int(playing_time["matches_played"])
        playing_time["minutes_played"] = float(playing_time["minutes_played"])
        playing_time["minutes_per_match"] = float(playing_time["minutes_per_match"])
        for k in performance_metrics:
            if isinstance(performance_metrics[k], (np.integer, np.floating)):
                performance_metrics[k] = float(performance_metrics[k])
        for k in per_90_metrics:
            if isinstance(per_90_metrics[k], (np.integer, np.floating)):
                per_90_metrics[k] = float(per_90_metrics[k])
        for f in form_data:
            if "goals" in f and isinstance(f["goals"], (np.integer, np.floating)):
                f["goals"] = float(f["goals"])
            if "shots" in f and isinstance(f["shots"], (np.integer, np.floating)):
                f["shots"] = float(f["shots"])
            if "passes_completed" in f and isinstance(f["passes_completed"], (np.integer, np.floating)):
                f["passes_completed"] = float(f["passes_completed"])
            if "minutes_played" in f and isinstance(f["minutes_played"], (np.integer, np.floating)):
                f["minutes_played"] = float(f["minutes_played"])
        return {
            "player_info": player_info,
            "playing_time": playing_time,
            "performance_metrics": performance_metrics,
            "per_90_metrics": per_90_metrics,
            "form": form_data,
            "competition_id": competition_id,
            "season_id": season_id
        }
    except Exception as e:
        logger.error(f"Error getting player profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/performance-trend")
async def get_player_performance_trend(
    player_id: int = Query(..., description="Player ID"),
    metric: str = Query(..., description="Metric to track over time"),
    timeframe: str = Query("season", description="Timeframe: 'season', 'last10', 'last5'"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player performance trend over time
    Returns time series data for a specific metric
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
            if events is None or events.empty:
                return {"error": "No data found for team in this competition/season."}
            player_events = events[events['player_id'] == player_id]
            matches = player_events['match_id'].unique()
        else:
            matches_df = fdm.get_matches(competition_id, season_id)
            matches = []
            for _, match in matches_df.iterrows():
                ev = fdm.get_events(match['match_id'])
                if player_id in ev['player_id'].unique():
                    matches.append((match, ev))
        if not matches or (isinstance(matches, list) and not matches):
            return {"error": "No data found for player."}
        # Determine number of matches for timeframe
        if timeframe == "season":
            num_matches = len(matches)
        elif timeframe == "last10":
            num_matches = min(10, len(matches))
        else:
            num_matches = min(5, len(matches))
        matches = matches[-num_matches:]
        # Aggregate metric per match
        trend_data = []
        for i, (match, events) in enumerate(matches):
            player_events = events[events['player_id'] == player_id]
            if metric == "goals":
                value = len(player_events[(player_events['type_name'] == 'Shot') & (player_events['shot_outcome'] == 'Goal')])
            elif metric == "assists":
                value = len(player_events[player_events['type_name'] == 'Pass'])  # Placeholder: refine with assist logic
            elif metric == "xg":
                value = player_events['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in player_events else 0
            elif metric == "shots":
                value = len(player_events[player_events['type_name'] == 'Shot'])
            elif metric == "key_passes":
                value = len(player_events[player_events['type_name'] == 'Pass'])  # Placeholder: refine with key pass logic
            elif metric == "tackles":
                value = len(player_events[player_events['type_name'] == 'Duel'])
            elif metric == "pass_accuracy":
                passes = player_events[player_events['type_name'] == 'Pass']
                completed = passes[passes['pass_outcome'].isna()]
                value = len(completed) / max(1, len(passes)) * 100
            else:
                value = None
            trend_data.append({
                "match_number": i + 1,
                "match_id": match['match_id'],
                "opponent": match.get('away_team', 'Unknown'),
                "value": value,
                "match_result": match.get('result', None)
            })
        # Rolling average (window=5)
        window_size = min(5, len(trend_data))
        rolling_avg = []
        for i in range(len(trend_data)):
            if i < window_size - 1:
                continue
            window_avg = sum(point["value"] for point in trend_data[i-(window_size-1):i+1]) / window_size
            rolling_avg.append({
                "match_number": trend_data[i]["match_number"],
                "value": window_avg
            })
        return {
            "player_id": player_id,
            "metric": metric,
            "timeframe": timeframe,
            "trend_data": trend_data,
            "rolling_average": rolling_avg,
            "competition_id": competition_id
        }
    except Exception as e:
        logger.error(f"Error getting player performance trend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/event-map")
async def get_player_event_map(
    player_id: int = Query(..., description="Player ID"),
    event_type: str = Query("all", description="Event type: 'all', 'passes', 'shots', 'defensive'"),
    match_id: Optional[int] = Query(None, description="Filter by match ID"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player event map
    Returns location data for player events for pitch visualization
    """
    try:
        if match_id:
            events = fdm.get_events(match_id)
        elif competition_id and season_id and team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        elif competition_id and season_id:
            matches_df = fdm.get_matches(competition_id, season_id)
            all_events = []
            for _, match in matches_df.iterrows():
                ev = fdm.get_events(match['match_id'])
                if player_id in ev['player_id'].unique():
                    all_events.append(ev)
            if not all_events:
                return {"error": "No data found for player."}
            events = pd.concat(all_events)
        else:
            return {"error": "Insufficient selector parameters. Provide match_id or (competition_id, season_id[, team_id])."}
        player_events = events[events['player_id'] == player_id]
        # Filter by event type
        if event_type == "passes":
            player_events = player_events[player_events['type_name'] == 'Pass']
        elif event_type == "shots":
            player_events = player_events[player_events['type_name'] == 'Shot']
        elif event_type == "defensive":
            player_events = player_events[player_events['type_name'].isin(['Duel', 'Interception', 'Tackle', 'Block'])]
        # Build event list
        event_list = []
        for _, row in player_events.iterrows():
            event = {
                "id": row.name,
                "type": row['type_name'],
                "x": row['location'][0] if isinstance(row['location'], list) else None,
                "y": row['location'][1] if isinstance(row['location'], list) else None,
                "minute": row['minute'],
                "success": row.get('pass_outcome', None) is None if row['type_name'] == 'Pass' else None,
                "end_x": row['pass_end_location'][0] if row['type_name'] == 'Pass' and isinstance(row.get('pass_end_location'), list) else None,
                "end_y": row['pass_end_location'][1] if row['type_name'] == 'Pass' and isinstance(row.get('pass_end_location'), list) else None
            }
            event_list.append(event)
        return {
            "player_id": player_id,
            "event_type": event_type,
            "match_id": match_id,
            "competition_id": competition_id,
            "events": event_list
        }
    except Exception as e:
        logger.error(f"Error getting player event map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/percentile-ranks")
async def get_player_percentile_ranks(
    player_id: int = Query(..., description="Player ID"),
    position_group: Optional[str] = Query(None, description="Position group to compare against"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    min_minutes: int = Query(450, description="Minimum minutes played for comparison group"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional, for faster lookup)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player percentile rankings compared to position peers
    Returns percentile ranks for various metrics
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
        # Determine position group if not specified
        if not position_group:
            player_row = events[events['player_id'] == player_id].iloc[0]
            position_group = player_row.get('position', 'Unknown')
        # Filter to players in position group with min_minutes
        player_minutes = events.groupby('player_id')['minute'].sum()
        eligible_players = player_minutes[player_minutes >= min_minutes].index
        group_events = events[(events['player_id'].isin(eligible_players)) & (events['position'] == position_group)]
        # Define metrics
        metrics = [
            "goals", "assists", "xg", "shots", "passes_completed", "pass_accuracy"
        ]
        # Calculate raw values for all players
        player_stats = {}
        for player in group_events['player_id'].unique():
            pe = group_events[group_events['player_id'] == player]
            goals = len(pe[(pe['type_name'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
            assists = len(pe[pe['type_name'] == 'Pass'])  # Placeholder
            xg = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
            shots = len(pe[pe['type_name'] == 'Shot'])
            passes_completed = len(pe[(pe['type_name'] == 'Pass') & (pe['pass_outcome'].isna())])
            pass_accuracy = passes_completed / max(1, len(pe[pe['type_name'] == 'Pass'])) * 100
            player_stats[player] = {
                "goals": goals,
                "assists": assists,
                "xg": xg,
                "shots": shots,
                "passes_completed": passes_completed,
                "pass_accuracy": pass_accuracy
            }
        # Calculate percentiles for the requested player
        percentiles = {}
        comparison_values = {}
        for metric in metrics:
            values = np.array([stats[metric] for stats in player_stats.values()])
            player_value = player_stats.get(player_id, {}).get(metric, 0)
            percentile = float(np.sum(values < player_value)) / len(values) * 100 if len(values) > 0 else 0
            percentiles[metric] = percentile
            comparison_values[metric] = {
                "player": player_value,
                "average": float(np.mean(values)) if len(values) > 0 else 0,
                "max": float(np.max(values)) if len(values) > 0 else 0
            }
        return {
            "player_id": player_id,
            "position_group": position_group,
            "percentile_ranks": percentiles,
            "comparison_values": comparison_values,
            "metrics": metrics,
            "competition_id": competition_id,
            "season_id": season_id,
            "min_minutes": min_minutes,
            "comparison_group_size": len(player_stats)
        }
    except Exception as e:
        logger.error(f"Error getting player percentile ranks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
