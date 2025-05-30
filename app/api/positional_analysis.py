from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
import pandas as pd
import numpy as np

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
        matches_df = fdm.get_matches(competition_id, season_id)
        all_events = []
        for _, match in matches_df.iterrows():
            all_events.append(fdm.get_events(match['match_id']))
        if not all_events:
            return {"error": "No data found for competition/season."}
        events = pd.concat(all_events)
        # Filter to players in position group with min_minutes
        player_minutes = events.groupby('player')['minute'].sum()
        eligible_players = player_minutes[player_minutes >= min_minutes].index
        group_events = events[(events['player'].isin(eligible_players)) & (events['position'] == position)]
        # Calculate per-90 metrics for each player
        player_data = []
        for player_id, pe in group_events.groupby('player'):
            minutes = pe['minute'].sum()
            goals = len(pe[(pe['type'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
            assists = len(pe[pe['type'] == 'Pass'])  # Placeholder
            xg = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
            progressive_passes = len(pe[pe['type'] == 'Pass'])  # Placeholder
            successful_dribbles = len(pe[pe['type'] == 'Carry'])  # Placeholder
            defensive_actions = len(pe[pe['type'].isin(['Duel', 'Interception', 'Tackle', 'Block'])])
            pressures = len(pe[pe['type'] == 'Pressure'])
            per_90 = lambda v: v / (minutes / 90) if minutes > 0 else 0
            player_data.append({
                "player_id": player_id,
                "player_name": pe.iloc[0]["player"] if not pe.empty else f"Player {player_id}",
                "team": pe.iloc[0]["team"] if not pe.empty else None,
                "minutes": minutes,
                "key_metrics": {
                    "goals_per_90": per_90(goals),
                    "assists_per_90": per_90(assists),
                    "xg_per_90": per_90(xg),
                    "progressive_passes_per_90": per_90(progressive_passes),
                    "successful_dribbles_per_90": per_90(successful_dribbles),
                    "defensive_actions_per_90": per_90(defensive_actions),
                    "pressures_per_90": per_90(pressures)
                }
            })
        return {
            "position": position,
            "players": player_data,
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
        if match_id:
            events = fdm.get_events(match_id)
        elif competition_id:
            matches_df = fdm.get_matches(competition_id, None)
            all_events = []
            for _, match in matches_df.iterrows():
                all_events.append(fdm.get_events(match['match_id']))
            if not all_events:
                return {"error": "No data found for competition."}
            events = pd.concat(all_events)
        else:
            return {"error": "No match or competition specified."}
        team_events = events[events['team_id'] == team_id]
        # Define zones
        zones = []
        if zone_type == "vertical":
            zones = ["Defensive Third", "Middle Third", "Attacking Third"]
            def get_zone(x, y):
                if x < 40:
                    return "Defensive Third"
                elif x < 80:
                    return "Middle Third"
                else:
                    return "Attacking Third"
        elif zone_type == "horizontal":
            zones = ["Left Channel", "Central Channel", "Right Channel"]
            def get_zone(x, y):
                if y < 26.6:
                    return "Left Channel"
                elif y < 53.3:
                    return "Central Channel"
                else:
                    return "Right Channel"
        elif zone_type == "grid":
            zones = [f"{v} {h}" for v in ["Defensive", "Middle", "Attacking"] for h in ["Left", "Central", "Right"]]
            def get_zone(x, y):
                if x < 40:
                    v = "Defensive"
                elif x < 80:
                    v = "Middle"
                else:
                    v = "Attacking"
                if y < 26.6:
                    h = "Left"
                elif y < 53.3:
                    h = "Central"
                else:
                    h = "Right"
                return f"{v} {h}"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid zone_type: {zone_type}")
        # Assign zone to each event
        team_events = team_events[team_events['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)]
        team_events = team_events.copy()
        team_events['zone'] = team_events['location'].apply(lambda loc: get_zone(loc[0], loc[1]))
        # Aggregate metrics by zone
        zone_metrics = {}
        for zone in zones:
            zone_df = team_events[team_events['zone'] == zone]
            zone_metrics[zone] = {
                "possession_pct": 100 * len(zone_df[zone_df['type'] == 'Pass']) / max(1, len(team_events[team_events['type'] == 'Pass'])),
                "passes": len(zone_df[zone_df['type'] == 'Pass']),
                "pass_success_rate": 100 * len(zone_df[(zone_df['type'] == 'Pass') & (zone_df['pass_outcome'].isna())]) / max(1, len(zone_df[zone_df['type'] == 'Pass'])),
                "shots": len(zone_df[zone_df['type'] == 'Shot']),
                "xg": zone_df['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in zone_df else 0,
                "ball_recoveries": len(zone_df[zone_df['type'] == 'Ball Recovery']),
                "defensive_actions": len(zone_df[zone_df['type'].isin(['Duel', 'Interception', 'Tackle', 'Block'])])
            }
        # Player involvement by zone
        player_involvement = {}
        for zone in zones:
            zone_df = team_events[team_events['zone'] == zone]
            top_players = zone_df['player'].value_counts().head(3)
            players = []
            for player, actions in top_players.items():
                players.append({
                    "player": player,
                    "actions": actions
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
        if match_id:
            events = fdm.get_events(match_id)
        elif competition_id:
            matches_df = fdm.get_matches(competition_id, None)
            all_events = []
            for _, match in matches_df.iterrows():
                ev = fdm.get_events(match['match_id'])
                if player_id in ev['player'].unique():
                    all_events.append(ev)
            if not all_events:
                return {"error": "No data found for player."}
            events = pd.concat(all_events)
        else:
            return {"error": "No match or competition specified."}
        player_events = events[events['player'] == player_id]
        if event_type:
            player_events = player_events[player_events['type'] == event_type]
        actions = []
        for _, row in player_events.iterrows():
            if isinstance(row['location'], list) and len(row['location']) == 2:
                actions.append({
                    "x": row['location'][0],
                    "y": row['location'][1],
                    "type": row['type'],
                    "minute": row['minute'],
                    "outcome": row.get('pass_outcome', None) if row['type'] == 'Pass' else None
                })
        player_info = {
            "player_id": player_id,
            "player_name": player_events.iloc[0]["player"] if not player_events.empty else f"Player {player_id}",
            "team": player_events.iloc[0]["team"] if not player_events.empty else None,
            "position": player_events.iloc[0].get("position", None) if not player_events.empty else None
        }
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

def generate_metrics_for_role(
    position: str,
    competition_id: int,
    season_id: Optional[int],
    min_minutes: int,
    fdm: FootballDataManager
) -> Dict[str, float]:
    """
    Aggregate real per-90 metrics for all players in the given role/position.
    Returns the average per-90 metrics for the role.
    """
    matches_df = fdm.get_matches(competition_id, season_id)
    all_events = []
    for _, match in matches_df.iterrows():
        all_events.append(fdm.get_events(match['match_id']))
    if not all_events:
        return {}
    events = pd.concat(all_events)
    # Filter to players in position group with min_minutes
    player_minutes = events.groupby('player')['minute'].sum()
    eligible_players = player_minutes[player_minutes >= min_minutes].index
    group_events = events[(events['player'].isin(eligible_players)) & (events['position'] == position)]
    # Calculate per-90 metrics for each player
    per90_metrics = []
    for player_id, pe in group_events.groupby('player'):
        minutes = pe['minute'].sum()
        goals = len(pe[(pe['type'] == 'Shot') & (pe['shot_outcome'] == 'Goal')])
        assists = len(pe[pe['type'] == 'Pass'])  # Placeholder: refine with real assist logic if available
        xg = pe['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in pe else 0
        progressive_passes = len(pe[pe['type'] == 'Pass'])  # Placeholder
        successful_dribbles = len(pe[pe['type'] == 'Carry'])  # Placeholder
        defensive_actions = len(pe[pe['type'].isin(['Duel', 'Interception', 'Tackle', 'Block'])])
        pressures = len(pe[pe['type'] == 'Pressure'])
        per_90 = lambda v: v / (minutes / 90) if minutes > 0 else 0
        per90_metrics.append({
            "goals_per_90": per_90(goals),
            "assists_per_90": per_90(assists),
            "xg_per_90": per_90(xg),
            "progressive_passes_per_90": per_90(progressive_passes),
            "successful_dribbles_per_90": per_90(successful_dribbles),
            "defensive_actions_per_90": per_90(defensive_actions),
            "pressures_per_90": per_90(pressures)
        })
    if not per90_metrics:
        return {}
    # Compute average per-90 metrics for the role
    df = pd.DataFrame(per90_metrics)
    avg_metrics = df.mean().to_dict()
    return avg_metrics
