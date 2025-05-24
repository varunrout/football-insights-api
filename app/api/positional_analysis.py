from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/role-analysis")
async def get_role_analysis(
    position: str = Query(..., description="Position to analyze (e.g., 'Forward', 'Midfielder')"),
    competition_id: int = Query(..., description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    min_minutes: int = Query(450, description="Minimum minutes played"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get role analysis data for a specific position
    
    Returns metrics and player clusters for the specified position
    """
    try:
        # Map of position to specialized roles
        position_roles = {
            "Forward": ["Target Man", "False 9", "Inside Forward", "Winger"],
            "Midfielder": ["Deep-Lying Playmaker", "Box-to-Box", "Attacking Midfielder", "Defensive Midfielder"],
            "Defender": ["Ball-Playing CB", "Defensive CB", "Attacking FB", "Defensive FB"],
            "Goalkeeper": ["Sweeper Keeper", "Traditional Keeper"]
        }
        
        # Get roles for the requested position
        roles = position_roles.get(position, [])
        
        # Generate sample players for each role
        players_by_role = {}
        for role in roles:
            players = []
            for i in range(1, 6):  # 5 players per role
                player_id = 1000 + len(players_by_role) * 10 + i
                players.append({
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "team": f"Team {player_id % 10}",
                    "minutes": min_minutes + (i * 200),
                    "role_score": 0.8 + (i * 0.03),  # How well they fit the role
                    "key_metrics": generate_metrics_for_role(role)
                })
            players_by_role[role] = players
        
        # Generate role definitions
        role_definitions = {}
        for role in roles:
            role_definitions[role] = {
                "description": f"Players who primarily {role_description(role)}",
                "key_metrics": key_metrics_for_role(role)
            }
        
        return {
            "position": position,
            "roles": roles,
            "role_definitions": role_definitions,
            "players_by_role": players_by_role,
            "competition_id": competition_id,
            "season_id": season_id,
            "min_minutes": min_minutes
        }
    except Exception as e:
        logger.error(f"Error getting role analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/zone-analysis")
async def get_zone_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Competition ID"),
    zone_type: str = Query("vertical", description="Zone type: 'vertical', 'horizontal', or 'grid'"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get zone analysis data for a team
    
    Returns metrics by pitch zone (vertical thirds, horizontal thirds, or grid)
    """
    try:
        # Determine zones based on type
        zones = []
        if zone_type == "vertical":
            zones = ["Defensive Third", "Middle Third", "Attacking Third"]
        elif zone_type == "horizontal":
            zones = ["Left Channel", "Central Channel", "Right Channel"]
        elif zone_type == "grid":
            # 3x3 grid
            vertical_zones = ["Defensive", "Middle", "Attacking"]
            horizontal_zones = ["Left", "Central", "Right"]
            for v in vertical_zones:
                for h in horizontal_zones:
                    zones.append(f"{v} {h}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid zone_type: {zone_type}")
        
        # Generate metrics for each zone
        zone_metrics = {}
        for zone in zones:
            zone_metrics[zone] = {
                "possession_pct": 40 + (hash(zone) % 40),  # Random-ish percentage between 40-80%
                "passes": 100 + (hash(zone) % 200),
                "pass_success_rate": 70 + (hash(zone) % 20),
                "shots": 1 + (hash(zone) % 10),
                "xg": (1 + (hash(zone) % 10)) * 0.1,
                "ball_recoveries": 5 + (hash(zone) % 15),
                "defensive_actions": 10 + (hash(zone) % 20)
            }
        
        # Generate player involvement by zone
        player_involvement = {}
        for zone in zones:
            players = []
            for i in range(1, 4):  # Top 3 players per zone
                player_id = 1000 + (hash(zone) % 100) + i
                players.append({
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "position": "Forward" if player_id % 4 == 0 else 
                              "Midfielder" if player_id % 4 == 1 else 
                              "Defender" if player_id % 4 == 2 else "Goalkeeper",
                    "actions": 10 + (hash(f"{zone}_{player_id}") % 50),
                    "success_rate": 60 + (hash(f"{zone}_{player_id}") % 30)
                })
            player_involvement[zone] = players
        
        return {
            "team_id": team_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "zone_type": zone_type,
            "zones": zones,
            "zone_metrics": zone_metrics,
            "player_involvement": player_involvement
        }
    except Exception as e:
        logger.error(f"Error getting zone analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/heat-map")
async def get_player_heat_map(
    player_id: int = Query(..., description="Player ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Competition ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g., 'Pass', 'Shot')"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get player heat map data
    
    Returns player action locations for generating a heat map
    """
    try:
        # Generate sample player info
        player_info = {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team": f"Team {player_id % 10}",
            "position": "Forward" if player_id % 4 == 0 else 
                      "Midfielder" if player_id % 4 == 1 else 
                      "Defender" if player_id % 4 == 2 else "Goalkeeper",
        }
        
        # Generate sample action locations
        # Distribution will vary based on position
        actions = []
        num_actions = 150  # Sample size
        
        # Adjust position bias based on player position
        if player_info["position"] == "Forward":
            x_mean, y_mean = 80, 40  # Higher up the pitch
        elif player_info["position"] == "Midfielder":
            x_mean, y_mean = 60, 40  # Middle of the pitch
        elif player_info["position"] == "Defender":
            x_mean, y_mean = 30, 40  # Lower down the pitch
        else:  # Goalkeeper
            x_mean, y_mean = 10, 40  # Own goal area
        
        # Generate semi-random action locations with position bias
        import random
        random.seed(player_id)  # For reproducibility
        
        for i in range(num_actions):
            # Add some noise to the position
            x = max(0, min(120, x_mean + random.normalvariate(0, 20)))
            y = max(0, min(80, y_mean + random.normalvariate(0, 15)))
            
            # Determine event type (if not filtered)
            if event_type:
                action_type = event_type
            else:
                if player_info["position"] == "Forward":
                    types = ["Pass", "Shot", "Carry", "Ball Receipt"]
                    weights = [0.5, 0.2, 0.2, 0.1]
                elif player_info["position"] == "Midfielder":
                    types = ["Pass", "Shot", "Carry", "Ball Receipt", "Pressure"]
                    weights = [0.6, 0.05, 0.2, 0.1, 0.05]
                elif player_info["position"] == "Defender":
                    types = ["Pass", "Clearance", "Pressure", "Duel", "Carry"]
                    weights = [0.5, 0.1, 0.2, 0.1, 0.1]
                else:  # Goalkeeper
                    types = ["Pass", "Goal Keeper", "Clearance"]
                    weights = [0.6, 0.3, 0.1]
                
                action_type = random.choices(types, weights=weights)[0]
            
            # Add the action
            actions.append({
                "x": x,
                "y": y,
                "type": action_type,
                "minute": random.randint(1, 90),
                "outcome": random.choice(["Successful", "Unsuccessful"]) if random.random() < 0.8 else None
            })
        
        # If specific event type was requested, filter actions
        if event_type:
            actions = [a for a in actions if a["type"] == event_type]
        
        return {
            "player": player_info,
            "match_id": match_id,
            "competition_id": competition_id,
            "event_type": event_type,
            "actions": actions,
            "action_count": len(actions)
        }
    except Exception as e:
        logger.error(f"Error getting player heat map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Helper functions for role analysis
def role_description(role: str) -> str:
    """Return a description for a player role"""
    descriptions = {
        "Target Man": "act as a focal point in attack, winning aerial duels and holding up play",
        "False 9": "drop deep to create space and link play rather than staying high as a traditional striker",
        "Inside Forward": "cut inside from wide positions to create and score goals",
        "Winger": "stay wide to deliver crosses and beat defenders one-on-one",
        "Deep-Lying Playmaker": "dictate play from deep positions with precise passing",
        "Box-to-Box": "contribute both defensively and offensively across the entire pitch",
        "Attacking Midfielder": "create chances and score goals from central positions behind the striker",
        "Defensive Midfielder": "shield the defense and disrupt opposition attacks",
        "Ball-Playing CB": "initiate attacks with precise passing from defense",
        "Defensive CB": "focus on defending through tackles, interceptions and aerial duels",
        "Attacking FB": "provide width and attacking support from full-back positions",
        "Defensive FB": "focus on defensive duties with limited attacking involvement",
        "Sweeper Keeper": "act as an additional outfield player by coming off their line",
        "Traditional Keeper": "focus on shot-stopping and command of the penalty area"
    }
    return descriptions.get(role, "perform their role effectively")

def key_metrics_for_role(role: str) -> Dict[str, str]:
    """Return key metrics that define a role"""
    metrics = {
        "Target Man": {
            "aerial_duels_won": "High",
            "headed_goals": "High",
            "ball_retention": "Medium",
            "key_passes": "Low"
        },
        "False 9": {
            "progressive_passes": "High",
            "progressive_carries": "High",
            "key_passes": "High",
            "aerial_duels_won": "Low"
        },
        "Inside Forward": {
            "shots": "High",
            "xG": "High",
            "successful_dribbles": "High",
            "crosses": "Low"
        },
        "Winger": {
            "crosses": "High",
            "successful_dribbles": "High",
            "progressive_carries": "High",
            "defensive_actions": "Low"
        },
        "Deep-Lying Playmaker": {
            "progressive_passes": "Very High",
            "pass_accuracy": "Very High",
            "tackles": "Medium",
            "shots": "Low"
        },
        "Box-to-Box": {
            "distance_covered": "Very High",
            "tackles": "High",
            "progressive_carries": "High",
            "shots": "Medium"
        },
        "Attacking Midfielder": {
            "key_passes": "Very High",
            "xA": "High",
            "shots": "High",
            "defensive_actions": "Low"
        },
        "Defensive Midfielder": {
            "interceptions": "High",
            "tackles": "High",
            "pass_accuracy": "Medium",
            "shots": "Low"
        },
        "Ball-Playing CB": {
            "pass_accuracy": "High",
            "progressive_passes": "High",
            "defensive_actions": "High",
            "successful_dribbles": "Medium"
        },
        "Defensive CB": {
            "clearances": "High",
            "aerial_duels_won": "High",
            "blocks": "High",
            "progressive_passes": "Low"
        },
        "Attacking FB": {
            "crosses": "High",
            "progressive_carries": "High",
            "key_passes": "Medium",
            "defensive_actions": "Medium"
        },
        "Defensive FB": {
            "tackles": "High",
            "interceptions": "High",
            "progressive_carries": "Low",
            "crosses": "Low"
        },
        "Sweeper Keeper": {
            "defensive_actions_outside_box": "High",
            "pass_accuracy": "High",
            "long_pass_accuracy": "High",
            "saves": "Medium"
        },
        "Traditional Keeper": {
            "saves": "High",
            "aerial_claims": "High",
            "pass_accuracy": "Medium",
            "defensive_actions_outside_box": "Low"
        }
    }
    return metrics.get(role, {})

def generate_metrics_for_role(role: str) -> Dict[str, float]:
    """Generate sample metrics for a player of a given role"""
    # Base metrics that all roles have
    metrics = {
        "goals_per_90": 0.1,
        "assists_per_90": 0.1,
        "xg_per_90": 0.15,
        "xa_per_90": 0.15,
        "progressive_passes_per_90": 3.0,
        "successful_dribbles_per_90": 1.0,
        "defensive_actions_per_90": 5.0,
        "pressures_per_90": 15.0,
        "aerial_duels_won_per_90": 1.0,
        "pass_accuracy": 75.0,
        "tackles_per_90": 1.5,
        "interceptions_per_90": 1.2,
        "crosses_per_90": 1.0,
        "progressive_carries_per_90": 2.0,
        "key_passes_per_90": 1.0,
        "shots_per_90": 1.0,
        "clearances_per_90": 1.0,
        "blocks_per_90": 0.5
    }
    
    # Adjust metrics based on role
    if role == "Target Man":
        metrics.update({
            "goals_per_90": 0.5,
            "aerial_duels_won_per_90": 4.5,
            "headed_goals_per_90": 0.2
        })
    elif role == "False 9":
        metrics.update({
            "goals_per_90": 0.4,
            "assists_per_90": 0.3,
            "progressive_passes_per_90": 5.0,
            "key_passes_per_90": 2.5
        })
    elif role == "Inside Forward":
        metrics.update({
            "goals_per_90": 0.6,
            "shots_per_90": 3.5,
            "xg_per_90": 0.55,
            "successful_dribbles_per_90": 3.0
        })
    elif role == "Winger":
        metrics.update({
            "assists_per_90": 0.4,
            "crosses_per_90": 5.0,
            "successful_dribbles_per_90": 3.5,
            "progressive_carries_per_90": 4.5
        })
    elif role == "Deep-Lying Playmaker":
        metrics.update({
            "progressive_passes_per_90": 8.0,
            "pass_accuracy": 92.0,
            "key_passes_per_90": 1.8,
            "defensive_actions_per_90": 4.0
        })
    elif role == "Box-to-Box":
        metrics.update({
            "distance_covered_per_90": 12.0,
            "progressive_carries_per_90": 3.5,
            "tackles_per_90": 2.5,
            "shots_per_90": 1.5
        })
    elif role == "Attacking Midfielder":
        metrics.update({
            "key_passes_per_90": 3.0,
            "xa_per_90": 0.4,
            "shots_per_90": 2.5,
            "successful_dribbles_per_90": 2.5
        })
    elif role == "Defensive Midfielder":
        metrics.update({
            "interceptions_per_90": 2.5,
            "tackles_per_90": 3.0,
            "pressures_per_90": 25.0,
            "aerial_duels_won_per_90": 2.0
        })
    
    # Add some random variation
    import random
    random.seed(hash(role))
    for key in metrics:
        metrics[key] *= random.uniform(0.9, 1.1)
    
    return metrics
