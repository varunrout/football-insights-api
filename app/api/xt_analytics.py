from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from app.util.football_data_manager import FootballDataManager
from app.util.metrics.expected_threat import ExpectedThreatModel
from app.services.metric_calculator import load_xt_model, calculate_xt_added, get_player_xt_contributions

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/model")
async def get_xt_model():
    """
    Get the Expected Threat (xT) model grid values and pitch details

    Returns the xT values for each grid cell on the pitch, along with pitch metadata
    """
    try:
        xt_model = load_xt_model()
        grid_info = xt_model.get_xt_grid()
        # Add detailed pitch info (StatsBomb standard, in yards)
        pitch_info = {
            "length": 120,
            "width": 80,
            "units": "yards",
            "coordinate_system": "StatsBomb",
            "origin": "bottom left",
            "x_range": [0, 120],
            "y_range": [0, 80],
            "penalty_area": {
                "width": 44,
                "length": 18
            },
            "goal_area": {
                "width": 20,
                "length": 6
            },
            "goal": {
                "width": 8,
                "depth": 0
            },
            "center_circle_radius": 10,
            "corner_radius": 1
        }
        return {"grid": grid_info, "pitch": pitch_info}
    except Exception as e:
        logger.error(f"Error getting xT model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/player-rankings")
async def get_player_xt_rankings(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: Optional[int] = Query(None, description="Team ID (if None, returns all teams)"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    min_minutes: int = Query(90, description="Minimum minutes played"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player rankings by xT contribution
    
    Returns players sorted by their xT contribution
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        if team_id:
            matches = fdm.get_matches_for_team(competition_id, season_id, team_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
        else:
            matches = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
        if not all_events:
            return {"players": []}
        events_df = pd.concat(all_events)
        # Calculate player xT contributions
        player_xt_df = get_player_xt_contributions(events_df)
        # Filter by min_minutes if possible (fallback to min_actions)
        # Here, positive_actions is a proxy for minutes if minutes not available
        player_xt_df = player_xt_df[player_xt_df['positive_actions'] >= min_minutes // 10]  # Approx: 10 actions per 90 min
        # Merge player names if available in events_df
        if "player_id" in events_df.columns and "player_name" in events_df.columns:
            player_names = events_df[["player_id", "player_name"]].drop_duplicates()
            player_xt_df = player_xt_df.merge(player_names, on="player_id", how="left")
        # Sort and format
        player_xt_df = player_xt_df.sort_values("total_xt_added", ascending=False)
        players = [
            {
                "player_id": row["player_id"],
                "team_name": row["team_name"],
                "total_xt": row["total_xt_added"],
                "xt_per_action": row["avg_xt_per_action"],
                "positive_actions": row["positive_actions"],
                "player_name": row["player_name"] if "player_name" in row and pd.notnull(row["player_name"]) else "Unknown"
            }
            for _, row in player_xt_df.iterrows()
        ]
        return {"players": players}
    except Exception as e:
        logger.error(f"Error getting player xT rankings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/pass-map")
async def get_xt_pass_map(
    match_id: Optional[int] = Query(None, description="Match ID (optional)"),
    team_id: Optional[int] = Query(None, description="Team ID (optional)"),
    player_id: Optional[int] = Query(None, description="Player ID (optional)"),
    season_id: Optional[int] = Query(None, description="Season ID (required if no match_id)"),
    competition_id: Optional[int] = Query(None, description="Competition ID (required if no match_id)"),
    min_xt: float = Query(0.05, description="Minimum xT value for passes to include"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get xT pass map data
    Returns passes with xT values for visualization, filtered by match, team, or player as needed
    """
    try:
        # Determine which events to load
        if match_id:
            events = fdm.get_events(match_id)
        elif team_id and competition_id and season_id:
            matches = fdm.get_matches_for_team(competition_id, season_id, team_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
            events = pd.concat(all_events) if all_events else None
        elif player_id and competition_id and season_id:
            matches = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
            events = pd.concat(all_events) if all_events else None
        else:
            return {"passes": []}
        if events is None or events.empty:
            return {"passes": []}
        xt_model = load_xt_model()
        events_with_xt = calculate_xt_added(events, xt_model)
        passes = events_with_xt[(events_with_xt['type_name'] == 'Pass') & (events_with_xt['xt_added'] >= min_xt)]
        # Apply filters
        if match_id and team_id:
            passes = passes[passes['team_id'] == team_id]
        if match_id and player_id:
            passes = passes[passes['player_id'] == player_id]
        if not match_id and team_id:
            passes = passes[passes['team_id'] == team_id]
        if not match_id and player_id:
            passes = passes[passes['player_id'] == player_id]
        # Build pass list with player_name and outcome if available
        pass_list = []
        accurate_count = 0
        for _, row in passes.iterrows():
            outcome = row.get("pass_outcome_name") or row.get("pass_outcome") or "Complete"
            if outcome in [None, "", "Complete", "Successful", "Accurate", "Pass"]:
                accurate_count += 1
            pass_list.append({
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name", "Unknown"),
                "team_id": row.get("team_id"),
                "team_name": row.get("team_name"),
                "start_x": row["location"][0] if isinstance(row["location"], list) else None,
                "start_y": row["location"][1] if isinstance(row["location"], list) else None,
                "end_x": row["pass_end_location"][0] if isinstance(row.get("pass_end_location"), list) else None,
                "end_y": row["pass_end_location"][1] if isinstance(row.get("pass_end_location"), list) else None,
                "xt_value": row["xt_added"],
                "outcome": outcome
            })
        total_passes = len(pass_list)
        pass_info = {
            "total": total_passes,
            "accurate": accurate_count,
            "accuracy": round(accurate_count / total_passes, 3) if total_passes > 0 else 0.0
        }
        return {"passes": pass_list, "pass_info": pass_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/team-contribution")
async def get_team_xt_contribution(
        competition_id: int = Query(..., description="Competition ID"),
        team_id: int = Query(..., description="Team ID"),
        match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
        season_id: Optional[int] = Query(None, description="Season ID"),
        fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get xT contribution breakdown by team

    Returns xT contribution by team areas and phases of play
    """
    try:
        xt_model = load_xt_model()
        if match_id:
            events = fdm.get_events(match_id)
        else:
            if not competition_id or not season_id:
                return {"error": "competition_id and season_id are required if match_id is not provided."}
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        if events is None or events.empty:
            return {}
        # Filter for team
        team_events = events[events['team_id'] == team_id]
        xt_events = calculate_xt_added(team_events, xt_model)

        # By zone (vertical thirds)
        def get_zone(x):
            if x < 40:
                return 'defensive_third'
            elif x < 80:
                return 'middle_third'
            else:
                return 'final_third'

        xt_events = xt_events.copy()
        xt_events['zone'] = xt_events['location'].apply(lambda loc: get_zone(loc[0]) if isinstance(loc, list) else None)
        by_zone = xt_events.groupby('zone')['xt_added'].sum().to_dict()
        # By player
        by_player = xt_events.groupby('player_id')['xt_added'].sum().reset_index().rename(columns={'xt_added': 'xt'})
        by_player = by_player.sort_values('xt', ascending=False).to_dict(orient='records')
        # By action type
        by_action_type = xt_events.groupby('type_name')['xt_added'].sum().to_dict()
        # By time period (15-min bins)
        xt_events['period'] = xt_events['minute'].apply(lambda m: (m // 15) * 15)
        by_time_period = xt_events.groupby('period')['xt_added'].sum().reset_index()
        by_time_period = [
            {"period": f"{int(row['period']) + 1}-{int(row['period']) + 15}", "xt": row['xt_added']} for _, row in
            by_time_period.iterrows()
        ]
        return {
            "team_id": team_id,
            "competition_id": competition_id,
            "match_id": match_id,
            "total_xt": xt_events['xt_added'].sum(),
            "by_zone": by_zone,
            "by_player": by_player,
            "by_action_type": by_action_type,
            "by_time_period": by_time_period
        }
    except Exception as e:
        logger.error(f"Error getting team xT contribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/zone-effectiveness")
async def get_zone_effectiveness(
        competition_id: int = Query(..., description="Competition ID"),
        team_id: Optional[int] = Query(None, description="Team ID (if None, returns league average)"),
        match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
        season_id: Optional[int] = Query(None, description="Season ID"),
        fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get zone effectiveness analysis

    Returns xT effectiveness by pitch zone
    """
    try:
        xt_model = load_xt_model()
        if match_id:
            events = fdm.get_events(match_id)
        else:
            if not competition_id or not season_id:
                return {"error": "competition_id and season_id are required if match_id is not provided."}
            if team_id:
                events = fdm.get_events_for_team(competition_id, season_id, team_id)
            else:
                matches = fdm.get_matches(competition_id, season_id)
                all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
                if not all_events:
                    return {}
                events = pd.concat(all_events)
        if events is None or events.empty:
            return {}
        xt_events = calculate_xt_added(events, xt_model)
        # Define 4x6 grid
        grid_x, grid_y = 6, 4
        x_bins = np.linspace(0, 120, grid_x + 1)
        y_bins = np.linspace(0, 80, grid_y + 1)
        xt_events = xt_events[xt_events['type_name'].isin(['Pass', 'Carry'])]
        xt_events = xt_events[xt_events['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)]
        xt_events['zone_x'] = xt_events['location'].apply(lambda loc: np.digitize(loc[0], x_bins) - 1)
        xt_events['zone_y'] = xt_events['location'].apply(lambda loc: np.digitize(loc[1], y_bins) - 1)
        pitch_zones = []
        for i in range(grid_y):
            for j in range(grid_x):
                zone_events = xt_events[(xt_events['zone_x'] == j) & (xt_events['zone_y'] == i)]
                effectiveness = zone_events['xt_added'].sum()
                actions = len(zone_events)
                pitch_zones.append({
                    "zone_id": f"{i + 1}_{j + 1}",
                    "x_start": x_bins[j],
                    "y_start": y_bins[i],
                    "x_end": x_bins[j + 1],
                    "y_end": y_bins[i + 1],
                    "effectiveness": effectiveness,
                    "xt_generated": effectiveness,
                    "actions": actions
                })
        return {
            "team_id": team_id,
            "competition_id": competition_id,
            "match_id": match_id,
            "pitch_zones": pitch_zones,
            "pitch_dimensions": {
                "length": 120,
                "width": 80
            }
        }
    except Exception as e:
        logger.error(f"Error getting zone effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")