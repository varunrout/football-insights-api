from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager

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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get comprehensive player profile
    
    Returns player info, key metrics, and performance summary
    """
    try:
        # Generate player info (in real implementation would get from DB)
        player_info = {
            "player_id": player_id,
            "name": f"Player {player_id}",
            "team": f"Team {player_id % 10}",
            "age": 20 + (player_id % 15),
            "nationality": "Country " + chr(65 + (player_id % 26)),
            "position": "Forward" if player_id % 4 == 0 else 
                       "Midfielder" if player_id % 4 == 1 else 
                       "Defender" if player_id % 4 == 2 else "Goalkeeper",
            "preferred_foot": "Right" if player_id % 3 != 0 else "Left",
            "height": 175 + (player_id % 30),
            "contract_expires": f"202{4 + (player_id % 4)}"
        }
        
        # Generate playing time data
        playing_time = {
            "matches_played": 20 + (player_id % 15),
            "matches_started": 15 + (player_id % 20),
            "minutes_played": 1500 + (player_id % 1500),
            "minutes_per_match": 75 + (player_id % 20),
            "matches_completed": 10 + (player_id % 15)
        }
        
        # Generate performance metrics based on position
        performance_metrics = {}
        
        if player_info["position"] == "Forward":
            performance_metrics = {
                "goals": 8 + (player_id % 15),
                "assists": 4 + (player_id % 8),
                "xg": 7.5 + (player_id % 10) * 0.5,
                "xa": 3.5 + (player_id % 7) * 0.5,
                "shots": 45 + (player_id % 30),
                "shots_on_target": 20 + (player_id % 15),
                "shot_accuracy": 40 + (player_id % 25),
                "conversion_rate": 15 + (player_id % 10),
                "successful_dribbles": 25 + (player_id % 20),
                "key_passes": 15 + (player_id % 20),
                "penalties_scored": 1 + (player_id % 3),
                "penalties_taken": 1 + (player_id % 4)
            }
        elif player_info["position"] == "Midfielder":
            performance_metrics = {
                "goals": 3 + (player_id % 8),
                "assists": 5 + (player_id % 10),
                "xg": 2.5 + (player_id % 6) * 0.5,
                "xa": 4.5 + (player_id % 8) * 0.5,
                "passes_completed": 800 + (player_id % 400),
                "pass_accuracy": 80 + (player_id % 15),
                "key_passes": 25 + (player_id % 25),
                "progressive_passes": 120 + (player_id % 80),
                "successful_dribbles": 30 + (player_id % 25),
                "tackles_won": 40 + (player_id % 30),
                "interceptions": 30 + (player_id % 20),
                "ball_recoveries": 80 + (player_id % 50)
            }
        elif player_info["position"] == "Defender":
            performance_metrics = {
                "goals": 1 + (player_id % 4),
                "assists": 2 + (player_id % 5),
                "xg": 1.0 + (player_id % 3) * 0.5,
                "xa": 1.5 + (player_id % 4) * 0.5,
                "clean_sheets": 5 + (player_id % 10),
                "tackles_won": 60 + (player_id % 40),
                "interceptions": 50 + (player_id % 30),
                "clearances": 80 + (player_id % 50),
                "blocks": 30 + (player_id % 20),
                "aerial_duels_won": 70 + (player_id % 50),
                "pass_accuracy": 75 + (player_id % 15),
                "errors_leading_to_shot": (player_id % 5)
            }
        else:  # Goalkeeper
            performance_metrics = {
                "clean_sheets": 5 + (player_id % 10),
                "saves": 70 + (player_id % 50),
                "save_percentage": 65 + (player_id % 25),
                "goals_conceded": 20 + (player_id % 15),
                "penalties_saved": 1 + (player_id % 2),
                "penalties_faced": 3 + (player_id % 5),
                "passes_attempted": 450 + (player_id % 250),
                "pass_accuracy": 70 + (player_id % 20),
                "catches": 15 + (player_id % 20),
                "punches": 8 + (player_id % 10),
                "high_claims": 12 + (player_id % 15),
                "errors_leading_to_goal": (player_id % 3)
            }
        
        # Calculate per 90 metrics
        per_90_metrics = {}
        minutes_per_90 = playing_time["minutes_played"] / 90
        
        for key, value in performance_metrics.items():
            # Only calculate per 90 for count metrics, not percentages
            if key not in ["pass_accuracy", "shot_accuracy", "conversion_rate", "save_percentage"]:
                per_90_metrics[f"{key}_per_90"] = value / minutes_per_90
        
        # Generate percentile rankings (comparison to other players in position)
        percentile_rankings = {}
        import random
        random.seed(player_id)
        
        for key in performance_metrics:
            # Generate random percentile between 50-99 for better visualization
            percentile_rankings[key] = 50 + random.randint(0, 49)
        
        # Generate form data (last 5 matches)
        form_data = []
        for i in range(5):
            match_data = {
                "match_id": 1000 + i,
                "opponent": f"Opponent {player_id % 20 + i}",
                "result": "W" if i % 3 == 0 else "D" if i % 3 == 1 else "L",
                "minutes_played": 90 if i % 4 != 0 else 70 + (i * 5),
                "rating": 6.5 + (i % 4) * 0.5 + random.random(),
                "goals": 1 if i % 3 == 0 and player_info["position"] != "Goalkeeper" else 0,
                "assists": 1 if i % 4 == 1 and player_info["position"] not in ["Goalkeeper", "Defender"] else 0
            }
            
            # Add position-specific metrics
            if player_info["position"] == "Forward":
                match_data.update({
                    "shots": 2 + i,
                    "shots_on_target": 1 if i % 2 == 0 else 0,
                    "xg": 0.3 + (i * 0.1)
                })
            elif player_info["position"] == "Midfielder":
                match_data.update({
                    "key_passes": 1 + i,
                    "pass_accuracy": 75 + (i * 3),
                    "tackles": 2 + (i % 3)
                })
            elif player_info["position"] == "Defender":
                match_data.update({
                    "tackles": 3 + i,
                    "interceptions": 2 + (i % 3),
                    "clearances": 4 + (i * 2)
                })
            else:  # Goalkeeper
                match_data.update({
                    "saves": 2 + i,
                    "goals_conceded": 1 if i % 2 == 0 else 0,
                    "clean_sheet": i % 2 == 1
                })
            
            form_data.append(match_data)
        
        return {
            "player_info": player_info,
            "playing_time": playing_time,
            "performance_metrics": performance_metrics,
            "per_90_metrics": per_90_metrics,
            "percentile_rankings": percentile_rankings,
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player performance trend over time
    
    Returns time series data for a specific metric
    """
    try:
        import random
        random.seed(player_id + hash(metric))
        
        # Generate timeline based on timeframe
        if timeframe == "season":
            num_matches = 30
        elif timeframe == "last10":
            num_matches = 10
        else:  # last5
            num_matches = 5
        
        # Generate trend data
        trend_data = []
        
        # Base value depends on metric
        if metric == "goals":
            base_value = 0.3
            max_value = 3
        elif metric == "assists":
            base_value = 0.2
            max_value = 2
        elif metric == "xg":
            base_value = 0.3
            max_value = 1.2
        elif metric == "shots":
            base_value = 2.0
            max_value = 8.0
        elif metric == "key_passes":
            base_value = 1.5
            max_value = 6.0
        elif metric == "tackles":
            base_value = 2.0
            max_value = 7.0
        elif metric == "pass_accuracy":
            base_value = 70.0
            max_value = 100.0
        else:
            base_value = 1.0
            max_value = 5.0
        
        # Generate data points with some randomness but maintaining a trend
        trend_direction = random.choice([1, -1])  # Improving or declining
        trend_factor = 0.05  # How strong the trend is
        
        for i in range(num_matches):
            match_number = i + 1
            # Calculate value with trend and random noise
            trend_component = trend_direction * i * trend_factor * base_value
            random_component = (random.random() - 0.5) * base_value
            value = min(max(base_value + trend_component + random_component, 0), max_value)
            
            # Add data point
            trend_data.append({
                "match_number": match_number,
                "match_id": 1000 + i,
                "opponent": f"Opponent {i + 1}",
                "value": value,
                "match_result": random.choice(["W", "D", "L"])
            })
        
        # Calculate rolling average
        window_size = min(5, num_matches)
        rolling_avg = []
        
        for i in range(num_matches):
            if i < window_size - 1:
                # Not enough data points for full window
                continue
                
            # Calculate average of last window_size matches
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player event map
    
    Returns location data for player events for pitch visualization
    """
    try:
        import random
        random.seed(player_id)
        
        # Generate player position to influence event locations
        player_position = random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"])
        
        # Number of events depends on event type
        if event_type == "all":
            num_events = 50
        elif event_type == "passes":
            num_events = 30
        elif event_type == "shots":
            num_events = 5 if player_position in ["Forward", "Midfielder"] else 1
        else:  # defensive
            num_events = 20 if player_position in ["Defender", "Midfielder"] else 5
        
        # Generate events
        events = []
        
        for i in range(num_events):
            # Generate location based on player position
            if player_position == "Forward":
                x = random.uniform(60, 115)
                y = random.uniform(10, 70)
            elif player_position == "Midfielder":
                x = random.uniform(40, 90)
                y = random.uniform(10, 70)
            elif player_position == "Defender":
                x = random.uniform(10, 60)
                y = random.uniform(10, 70)
            else:  # Goalkeeper
                x = random.uniform(0, 30)
                y = random.uniform(20, 60)
            
            # Determine specific event type
            if event_type == "all":
                specific_type = random.choice(["Pass", "Shot", "Tackle", "Interception", "Duel", "Ball Recovery"])
            elif event_type == "passes":
                specific_type = random.choice(["Short Pass", "Long Pass", "Cross", "Through Ball", "Key Pass"])
            elif event_type == "shots":
                specific_type = random.choice(["Shot", "Goal", "Shot on Target", "Shot off Target"])
            else:  # defensive
                specific_type = random.choice(["Tackle", "Interception", "Clearance", "Block", "Ball Recovery"])
            
            # Add success/failure status
            success = random.random() > 0.3  # 70% success rate
            
            # Add event
            events.append({
                "id": i + 1,
                "type": specific_type,
                "x": x,
                "y": y,
                "success": success,
                "minute": random.randint(1, 90),
                # For passes, add end location
                "end_x": x + random.uniform(5, 30) if "Pass" in specific_type else None,
                "end_y": y + random.uniform(-15, 15) if "Pass" in specific_type else None
            })
        
        return {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "position": player_position,
            "event_type": event_type,
            "match_id": match_id,
            "competition_id": competition_id,
            "events": events
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
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player percentile rankings compared to position peers
    
    Returns percentile ranks for various metrics
    """
    try:
        import random
        random.seed(player_id)
        
        # Determine player position if not specified
        if not position_group:
            position_group = random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"])
        
        # Define metrics based on position
        if position_group == "Forward":
            metrics = [
                "goals", "assists", "xg", "xa", "shots", "shots_on_target", 
                "shot_accuracy", "conversion_rate", "successful_dribbles", 
                "key_passes", "progressive_carries", "touches_in_box"
            ]
        elif position_group == "Midfielder":
            metrics = [
                "goals", "assists", "xg", "xa", "passes_completed", 
                "pass_accuracy", "key_passes", "progressive_passes", 
                "successful_dribbles", "tackles_won", "interceptions", 
                "ball_recoveries", "distance_covered"
            ]
        elif position_group == "Defender":
            metrics = [
                "clean_sheets", "tackles_won", "interceptions", "clearances", 
                "blocks", "aerial_duels_won", "pass_accuracy", "progressive_passes", 
                "errors_leading_to_shot", "ball_recoveries", "minutes_per_goal_conceded"
            ]
        else:  # Goalkeeper
            metrics = [
                "clean_sheets", "saves", "save_percentage", "goals_conceded", 
                "penalties_saved", "passes_attempted", "pass_accuracy", 
                "catches", "punches", "high_claims", "errors_leading_to_goal"
            ]
        
        # Generate percentile ranks
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = random.randint(1, 99)
        
        # Include comparison metrics to show raw values
        comparison_values = {}
        for metric in metrics:
            # Generate reasonable values based on metric
            if metric == "goals":
                comparison_values[metric] = {
                    "player": random.randint(5, 25),
                    "average": 10.5,
                    "max": 30
                }
            elif metric == "assists":
                comparison_values[metric] = {
                    "player": random.randint(3, 15),
                    "average": 7.2,
                    "max": 20
                }
            elif metric == "xg":
                comparison_values[metric] = {
                    "player": round(random.uniform(5.0, 20.0), 2),
                    "average": 9.8,
                    "max": 25.5
                }
            elif metric in ["pass_accuracy", "shot_accuracy", "conversion_rate", "save_percentage"]:
                comparison_values[metric] = {
                    "player": random.randint(60, 95),
                    "average": 78.5,
                    "max": 97
                }
            else:
                comparison_values[metric] = {
                    "player": random.randint(10, 100),
                    "average": 42.5,
                    "max": 120
                }
        
        return {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "position_group": position_group,
            "percentile_ranks": percentile_ranks,
            "comparison_values": comparison_values,
            "metrics": metrics,
            "competition_id": competition_id,
            "season_id": season_id,
            "min_minutes": min_minutes,
            "comparison_group_size": random.randint(30, 100)  # Number of players in comparison group
        }
    except Exception as e:
        logger.error(f"Error getting player percentile ranks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
