from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
from app.util.metrics.expected_threat import ExpectedThreatModel

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/model")
async def get_xt_model():
    """
    Get the Expected Threat (xT) model grid values
    
    Returns the xT values for each grid cell on the pitch
    """
    try:
        # This would normally load a trained model
        # Placeholder response with sample data
        return {
            "grid_size": {"x": 12, "y": 8},
            "pitch_dimensions": {"x": 120, "y": 80},
            "grid_values": [
                [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.01, 0.008],
                [0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.015, 0.01],
                [0.003, 0.005, 0.008, 0.01, 0.02, 0.025, 0.025, 0.02],
                [0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03],
                [0.008, 0.01, 0.02, 0.03, 0.05, 0.06, 0.06, 0.05],
                [0.01, 0.015, 0.025, 0.04, 0.06, 0.08, 0.08, 0.06],
                [0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.1, 0.08],
                [0.02, 0.03, 0.04, 0.07, 0.1, 0.15, 0.15, 0.1],
                [0.03, 0.04, 0.06, 0.1, 0.15, 0.2, 0.2, 0.15],
                [0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.3, 0.2],
                [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.4, 0.3],
                [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.6, 0.4]
            ]
        }
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
        # Placeholder response
        return {
            "players": [
                {"player_id": 1, "name": "Player A", "team": "Team 1", "total_xt": 4.2, "xt_per_90": 0.45, "minutes": 840},
                {"player_id": 2, "name": "Player B", "team": "Team 2", "total_xt": 3.8, "xt_per_90": 0.42, "minutes": 810},
                {"player_id": 3, "name": "Player C", "team": "Team 1", "total_xt": 3.5, "xt_per_90": 0.39, "minutes": 900},
                {"player_id": 4, "name": "Player D", "team": "Team 3", "total_xt": 3.2, "xt_per_90": 0.36, "minutes": 720},
                {"player_id": 5, "name": "Player E", "team": "Team 2", "total_xt": 2.9, "xt_per_90": 0.32, "minutes": 810}
            ]
        }
    except Exception as e:
        logger.error(f"Error getting player xT rankings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/pass-map")
async def get_xt_pass_map(
    match_id: int = Query(..., description="Match ID"),
    team_id: Optional[int] = Query(None, description="Team ID (if None, returns both teams)"),
    min_xt: float = Query(0.05, description="Minimum xT value for passes to include"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get xT pass map data
    
    Returns passes with xT values for visualization
    """
    try:
        # Placeholder response
        return {
            "passes": [
                {"player_id": 1, "player_name": "Player A", "start_x": 35, "start_y": 30, "end_x": 70, "end_y": 40, "xt_value": 0.08},
                {"player_id": 2, "player_name": "Player B", "start_x": 50, "start_y": 20, "end_x": 85, "end_y": 30, "xt_value": 0.12},
                {"player_id": 3, "player_name": "Player C", "start_x": 70, "start_y": 40, "end_x": 95, "end_y": 45, "xt_value": 0.18},
                {"player_id": 1, "player_name": "Player A", "start_x": 60, "start_y": 10, "end_x": 90, "end_y": 5, "xt_value": 0.09},
                {"player_id": 4, "player_name": "Player D", "start_x": 80, "start_y": 30, "end_x": 105, "end_y": 40, "xt_value": 0.25}
            ]
        }
    except Exception as e:
        logger.error(f"Error getting xT pass map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/team-contribution")
async def get_team_xt_contribution(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get xT contribution breakdown by team
    
    Returns xT contribution by team areas and phases of play
    """
    try:
        # Placeholder response
        return {
            "team_id": team_id,
            "competition_id": competition_id,
            "match_id": match_id,
            "total_xt": 14.2,
            "by_zone": {
                "defensive_third": 1.2,
                "middle_third": 4.8,
                "final_third": 8.2
            },
            "by_player": [
                {"player_id": 1, "player_name": "Player A", "xt": 3.2},
                {"player_id": 2, "player_name": "Player B", "xt": 2.8},
                {"player_id": 3, "player_name": "Player C", "xt": 2.5},
                {"player_id": 4, "player_name": "Player D", "xt": 2.1},
                {"player_id": 5, "player_name": "Player E", "xt": 1.9}
            ],
            "by_action_type": {
                "passes": 8.5,
                "carries": 4.2,
                "take_ons": 1.5
            },
            "by_time_period": [
                {"period": "0-15", "xt": 1.2},
                {"period": "16-30", "xt": 1.8},
                {"period": "31-45", "xt": 2.3},
                {"period": "46-60", "xt": 3.1},
                {"period": "61-75", "xt": 2.7},
                {"period": "76-90", "xt": 3.1}
            ]
        }
    except Exception as e:
        logger.error(f"Error getting team xT contribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/zone-effectiveness")
async def get_zone_effectiveness(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: Optional[int] = Query(None, description="Team ID (if None, returns league average)"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get zone effectiveness analysis
    
    Returns xT effectiveness by pitch zone
    """
    try:
        # Placeholder response
        pitch_zones = []
        
        # Create 4x6 grid of zones
        for i in range(4):  # 4 vertical zones
            for j in range(6):  # 6 horizontal zones
                zone_x = j * 20
                zone_y = i * 20
                
                # Effectiveness increases as we move up the pitch
                effectiveness = (j + 1) * 20
                
                # Add some variation
                import random
                random.seed(f"{team_id}_{i}_{j}")
                variation = random.uniform(-10, 10)
                effectiveness = max(0, min(100, effectiveness + variation))
                
                pitch_zones.append({
                    "zone_id": f"{i+1}_{j+1}",
                    "x_start": zone_x,
                    "y_start": zone_y,
                    "x_end": zone_x + 20,
                    "y_end": zone_y + 20,
                    "effectiveness": effectiveness,
                    "xt_generated": (j + 1) * (i + 1) * 0.15,
                    "actions": (j + 1) * (i + 1) * 10
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
