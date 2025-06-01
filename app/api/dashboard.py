from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
from app.services.metric_calculator import calculate_basic_match_metrics
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/summary")
async def get_dashboard_summary(
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get summary KPIs for dashboard (real data)
    """
    try:
        # Load all matches for the team/competition/season
        matches = fdm.get_matches_for_team(competition_id=competition_id, season_id=season_id, team_id=team_id)
        if matches is None or matches.empty:
            return {"metrics": {}, "competition_id": competition_id, "team_id": team_id, "season_id": season_id}
        # For each match, load all events (not just for the team)
        all_events = []
        for _, match in matches.iterrows():
            match_events = fdm.get_events(match["match_id"])
            if match_events is not None:
                all_events.append(match_events)
        if not all_events:
            return {"metrics": {}, "competition_id": competition_id, "team_id": team_id, "season_id": season_id}
        events = pd.concat(all_events)
        metrics = calculate_basic_match_metrics(events)
        # Use the first team in the metrics dict if team_id is None
        team_key = team_id if team_id in metrics else (list(metrics.keys())[0] if metrics else None)
        team_metrics = metrics.get(team_key, {})
        return {
            "metrics": {
                "goals": team_metrics.get("goals", 0),
                "xG": team_metrics.get("xg", 0),
                "possession": team_metrics.get("possession_pct", 0),
                "ppda": team_metrics.get("ppda", 0),
                "pass_success": team_metrics.get("pass_completion", 0)
            },
            "competition_id": competition_id,
            "team_id": team_id,
            "season_id": season_id,
            "events" : events['team_id'].unique().tolist() if 'team_id' in events else events['team'].unique().tolist()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/xg-timeline")
async def get_xg_timeline(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: int = Query(..., description="Team ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get xG vs Goals timeline for dashboard (real data)
    """
    try:
        matches = fdm.get_matches_for_team(competition_id=competition_id, season_id=season_id, team_id=team_id)
        timeline = []
        cumulative_xg = []
        cumulative_goals = []
        total_xg = 0
        total_goals = 0
        for _, match in matches.iterrows():
            events = fdm.get_events(match["match_id"])
            metrics = calculate_basic_match_metrics(events)
            team_name = team_id
            team_metrics = metrics.get(team_name, {})
            xg = team_metrics.get("xg", 0)
            goals = team_metrics.get("goals", 0)
            timeline.append({
                "match_id": match["match_id"],
                "opponent": match.get("opponent", "N/A"),
                "xg": xg,
                "goals": goals
            })
            total_xg += xg
            total_goals += goals
            cumulative_xg.append(total_xg)
            cumulative_goals.append(total_goals)
        return {
            "matches": timeline,
            "cumulative": {"xg": cumulative_xg, "goals": cumulative_goals}
        }
    except Exception as e:
        logger.error(f"Error getting xG timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/shot-map")
async def get_shot_map(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    season_id: Optional[int] = Query(None, description="Season ID (required if match_id is not provided)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get shot map data for dashboard (real data)
    """
    try:
        if match_id:
            if match_id is None:
                return {"shots": []}
            events = fdm.get_events(match_id)
        else:
            if season_id is None:
                return {"error": "season_id is required if match_id is not provided."}
            events = fdm.get_events_for_team(competition_id=competition_id, season_id=season_id, team_id=team_id)
        if events is None or len(events) == 0:
            return {"shots": []}
        # Try both 'team_id' and 'team' columns for compatibility
        if 'team_id' in events:
            shots = events[(events["team_id"] == team_id) & (events["type_name"] == "Shot")]
        else:
            shots = events[(events["team"] == team_id) & (events["type"] == "Shot")]
        shot_list = []
        for _, shot in shots.iterrows():
            shot_list.append({
                "x": shot["location"][0] if isinstance(shot["location"], list) else None,
                "y": shot["location"][1] if isinstance(shot["location"], list) else None,
                "xg": shot.get("shot_statsbomb_xg", 0),
                "outcome": shot.get("shot_outcome", "Unknown"),
                "minute": shot.get("minute", None),
                "player_id": shot.get("player_id", None)
            })
        return {"shots": shot_list}
    except Exception as e:
        logger.error(f"Error getting shot map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
