from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager

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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get radar chart data for player comparison
    
    Returns normalized metrics for multiple players in a format suitable for radar charts
    """
    try:
        # Default metrics if none specified
        if not metrics:
            metrics = [
                "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90", 
                "progressive_passes_per_90", "successful_dribbles_per_90",
                "defensive_actions_per_90", "pressures_per_90"
            ]
        
        # Placeholder response with sample data
        player_data = []
        for player_id in player_ids:
            # In a real implementation, this would fetch actual player data
            player_data.append({
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "team": f"Team {player_id % 5}",
                "metrics": {
                    "goals_per_90": 0.3 + (player_id % 5) * 0.1,
                    "assists_per_90": 0.2 + (player_id % 4) * 0.1,
                    "xg_per_90": 0.35 + (player_id % 3) * 0.15,
                    "xa_per_90": 0.25 + (player_id % 4) * 0.12,
                    "progressive_passes_per_90": 3.5 + (player_id % 10) * 0.5,
                    "successful_dribbles_per_90": 1.2 + (player_id % 5) * 0.3,
                    "defensive_actions_per_90": 5.0 + (player_id % 8) * 0.6,
                    "pressures_per_90": 15.0 + (player_id % 7) * 1.2
                }
            })
        
        # Generate min and max values for normalization
        metric_ranges = {}
        for metric in metrics:
            values = [p["metrics"].get(metric, 0) for p in player_data]
            metric_ranges[metric] = {
                "min": min(values),
                "max": max(values)
            }
        
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get bar chart data for player comparison
    
    Returns data for a single metric across multiple players
    """
    try:
        # Placeholder response with sample data
        player_data = []
        for player_id in player_ids:
            # In a real implementation, this would fetch actual player data
            value = 0.0
            if metric == "goals":
                value = 5 + (player_id % 10)
            elif metric == "assists":
                value = 3 + (player_id % 7)
            elif metric == "xg":
                value = 5.5 + (player_id % 8) * 0.8
            elif metric == "xa":
                value = 3.2 + (player_id % 6) * 0.7
            else:
                value = 10 + (player_id % 15)
                
            # Adjust to per 90 if requested
            minutes = 1500 + (player_id % 800)  # Simulated minutes played
            per_90_value = value / (minutes / 90) if per_90 else value
                
            player_data.append({
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "team": f"Team {player_id % 5}",
                "value": value,
                "per_90_value": per_90_value,
                "minutes": minutes
            })
        
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get scatter plot data for player comparison
    
    Returns data for comparing all players in a competition across two metrics
    """
    try:
        # Placeholder response with sample data for 40 players
        players = []
        for i in range(1, 41):
            # Generate random-ish data for each player
            player_id = 1000 + i
            x_value = (i % 10) * 0.5 + (i % 7) * 0.3
            y_value = (i % 8) * 0.4 + (i % 5) * 0.6
            minutes = min_minutes + (i % 15) * 100
            
            # Determine position group
            pos_groups = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
            player_position_group = pos_groups[i % 4]
            
            # Skip if filtered by position group
            if position_group and player_position_group != position_group:
                continue
            
            # Determine if this player should be highlighted
            is_highlighted = highlighted_player_ids and player_id in highlighted_player_ids
            
            players.append({
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "team": f"Team {i % 10}",
                "position_group": player_position_group,
                "x_value": x_value,
                "y_value": y_value,
                "minutes": minutes,
                "highlighted": is_highlighted
            })
        
        # Calculate league averages
        x_avg = sum(p["x_value"] for p in players) / len(players) if players else 0
        y_avg = sum(p["y_value"] for p in players) / len(players) if players else 0
        
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player similarity map data
    
    Returns a list of the most similar players to the reference player
    """
    try:
        # Placeholder response with sample data
        reference_player = {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team": f"Team {player_id % 10}",
            "position": "Forward" if player_id % 4 == 0 else 
                       "Midfielder" if player_id % 4 == 1 else 
                       "Defender" if player_id % 4 == 2 else "Goalkeeper",
            "minutes": 1800 + (player_id % 500),
            "key_metrics": {
                "goals_per_90": 0.45,
                "assists_per_90": 0.23,
                "xg_per_90": 0.52,
                "xa_per_90": 0.31,
                "progressive_passes_per_90": 4.2,
                "successful_dribbles_per_90": 1.8,
                "defensive_actions_per_90": 6.5,
                "pressures_per_90": 18.3
            }
        }
        
        # Generate similar players with varying similarity scores
        similar_players = []
        for i in range(1, limit + 1):
            similar_id = 1000 + (player_id + i) % 100
            similarity_score = 0.95 - (i * 0.05)
            
            similar_player = {
                "player_id": similar_id,
                "player_name": f"Player {similar_id}",
                "team": f"Team {similar_id % 10}",
                "position": reference_player["position"],  # Same position for similarity
                "minutes": 1500 + (similar_id % 800),
                "similarity_score": similarity_score,
                "key_metrics": {
                    "goals_per_90": reference_player["key_metrics"]["goals_per_90"] * (1 + (i % 5 - 2) * 0.1),
                    "assists_per_90": reference_player["key_metrics"]["assists_per_90"] * (1 + (i % 4 - 2) * 0.1),
                    "xg_per_90": reference_player["key_metrics"]["xg_per_90"] * (1 + (i % 6 - 3) * 0.1),
                    "xa_per_90": reference_player["key_metrics"]["xa_per_90"] * (1 + (i % 5 - 2) * 0.1),
                    "progressive_passes_per_90": reference_player["key_metrics"]["progressive_passes_per_90"] * (1 + (i % 4 - 2) * 0.1),
                    "successful_dribbles_per_90": reference_player["key_metrics"]["successful_dribbles_per_90"] * (1 + (i % 6 - 3) * 0.1),
                    "defensive_actions_per_90": reference_player["key_metrics"]["defensive_actions_per_90"] * (1 + (i % 5 - 2) * 0.1),
                    "pressures_per_90": reference_player["key_metrics"]["pressures_per_90"] * (1 + (i % 4 - 2) * 0.1)
                }
            }
            
            similar_players.append(similar_player)
        
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
