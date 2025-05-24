from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
from app.util.metrics.ppda import calculate_ppda, calculate_team_match_ppda
from app.util.metrics.pass_network import calculate_pass_network, analyze_team_structure

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/pass-network")
async def get_pass_network(
    match_id: int = Query(..., description="Match ID"),
    team_id: int = Query(..., description="Team ID"),
    min_passes: int = Query(3, description="Minimum passes between players to include in network"),
    include_subs: bool = Query(False, description="Include substitute players in the network"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get pass network data for a team in a specific match
    
    Returns data for visualizing player positions and pass connections
    """
    try:
        # Get match events
        events = fdm.get_events(match_id)
        
        # Get team name from team_id
        team_name = None
        for _, event in events.iterrows():
            if event.get('team_id') == team_id:
                team_name = event.get('team')
                break
        
        if not team_name:
            raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found in match {match_id}")
        
        # Calculate pass network
        pass_network = calculate_pass_network(events, team_name, min_passes, include_subs)
        
        # Analyze team structure
        structure_analysis = analyze_team_structure(pass_network)
        
        # Combine data for response
        response = {
            "match_id": match_id,
            "team_id": team_id,
            "team_name": team_name,
            "network": pass_network,
            "analysis": structure_analysis
        }
        
        return response
    except Exception as e:
        logger.error(f"Error getting pass network: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/ppda")
async def get_ppda_analysis(
    competition_id: int = Query(..., description="Competition ID"),
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    opposition_half_only: bool = Query(True, description="Consider only actions in opposition half"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get PPDA (Passes Per Defensive Action) analysis for a team
    
    Returns PPDA values which measure pressing intensity (lower = more intense pressing)
    """
    try:
        # If a specific match is requested
        if match_id:
            events = fdm.get_events(match_id)
            team_name = None
            
            # Find the team name from the events
            for _, event in events.iterrows():
                if event.get('team_id') == team_id:
                    team_name = event.get('team')
                    break
            
            if not team_name:
                raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found in match {match_id}")
                
            ppda = calculate_ppda(events, team_name, opposition_half_only)
            
            # Get opponent PPDA too
            teams = events['team'].unique()
            opponent = [t for t in teams if t != team_name][0] if len(teams) > 1 else "Unknown"
            opponent_ppda = calculate_ppda(events, opponent, opposition_half_only)
            
            return {
                "match_id": match_id,
                "team": team_name,
                "team_id": team_id,
                "ppda": ppda,
                "opponent": opponent,
                "opponent_ppda": opponent_ppda,
                "opposition_half_only": opposition_half_only
            }
        
        # For all matches in a competition for this team
        dataset_path = "data_cache/top_10_competitions"  # Path to the preprocessed dataset
        analysis_data = fdm.load_analysis_dataset(dataset_path, load_data=True)
        
        # Find matches for this team in the competition
        matches_data = {}
        team_name = None
        
        for comp_id, comp_data in analysis_data['competitions'].items():
            if comp_id != competition_id:
                continue
                
            for match_id, match_data in comp_data['matches'].items():
                events_df = match_data['events']
                
                # Check if team played in this match
                team_events = events_df[events_df['team_id'] == team_id]
                if not team_events.empty:
                    team_name = team_events.iloc[0]['team']
                    matches_data[match_id] = {
                        'home_team': match_data['home_team'],
                        'away_team': match_data['away_team'],
                        'events': events_df
                    }
        
        if not team_name:
            raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found in competition {competition_id}")
        
        # Calculate PPDA for each match
        ppda_by_match = []
        for match_id, match_data in matches_data.items():
            events_df = match_data['events']
            ppda = calculate_ppda(events_df, team_name, opposition_half_only)
            
            # Determine opponent
            is_home = match_data['home_team'] == team_name
            opponent = match_data['away_team'] if is_home else match_data['home_team']
            
            ppda_by_match.append({
                "match_id": match_id,
                "opponent": opponent,
                "ppda": ppda
            })
        
        # Calculate average PPDA
        avg_ppda = sum(m['ppda'] for m in ppda_by_match) / len(ppda_by_match) if ppda_by_match else 0
        
        return {
            "team_id": team_id,
            "team_name": team_name,
            "competition_id": competition_id,
            "average_ppda": avg_ppda,
            "ppda_by_match": ppda_by_match,
            "opposition_half_only": opposition_half_only
        }
    except Exception as e:
        logger.error(f"Error calculating PPDA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/shot-creation")
async def get_shot_creation_analysis(
    match_id: int = Query(..., description="Match ID"),
    team_id: int = Query(..., description="Team ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get shot creation analysis data for a team in a specific match
    
    Returns data on shot creation patterns, key passes, and assist locations
    """
    try:
        # Get match events
        events = fdm.get_events(match_id)
        
        # Get team name from team_id
        team_name = None
        for _, event in events.iterrows():
            if event.get('team_id') == team_id:
                team_name = event.get('team')
                break
        
        if not team_name:
            raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found in match {match_id}")
        
        # Filter to team's events
        team_events = events[events['team'] == team_name]
        
        # Get shots
        shots = team_events[team_events['type'] == 'Shot']
        
        # Collect shot creation sequences
        shot_sequences = []
        
        for _, shot in shots.iterrows():
            sequence = {
                "shot_id": shot.name,
                "minute": shot.get('minute', 0),
                "player": shot.get('player', 'Unknown'),
                "shot_location": shot.get('location', [0, 0]),
                "shot_outcome": shot.get('shot_outcome', 'Unknown'),
                "xg": shot.get('shot_statsbomb_xg', 0),
                "key_pass_id": shot.get('shot_key_pass_id'),
                "key_pass": None
            }
            
            # If there's a key pass, find it
            if pd.notna(shot.get('shot_key_pass_id')):
                key_pass_id = shot['shot_key_pass_id']
                
                # Look up the key pass event
                key_pass_events = events[events.index == key_pass_id]
                if not key_pass_events.empty:
                    key_pass = key_pass_events.iloc[0]
                    sequence["key_pass"] = {
                        "player": key_pass.get('player', 'Unknown'),
                        "location": key_pass.get('location', [0, 0]),
                        "pass_end_location": key_pass.get('pass_end_location', [0, 0]),
                        "pass_type": key_pass.get('pass_type', 'Regular')
                    }
            
            shot_sequences.append(sequence)
        
        # Aggregate key pass locations and types
        key_pass_locations = []
        for seq in shot_sequences:
            if seq["key_pass"]:
                key_pass_locations.append({
                    "location": seq["key_pass"]["location"],
                    "end_location": seq["key_pass"]["pass_end_location"],
                    "pass_type": seq["key_pass"]["pass_type"],
                    "player": seq["key_pass"]["player"],
                    "resulting_shot_outcome": seq["shot_outcome"],
                    "xg": seq["xg"]
                })
        
        # Compile shot creation zones (divide pitch into 6 zones and count)
        zones = {
            "defensive_third": 0,
            "middle_third": 0,
            "final_third": 0,
            "left_channel": 0,
            "central_channel": 0,
            "right_channel": 0
        }
        
        for kp in key_pass_locations:
            if isinstance(kp["location"], list) and len(kp["location"]) >= 2:
                x, y = kp["location"][0], kp["location"][1]
                
                # Horizontal zones (pitch length = 120)
                if x < 40:
                    zones["defensive_third"] += 1
                elif x < 80:
                    zones["middle_third"] += 1
                else:
                    zones["final_third"] += 1
                
                # Vertical zones (pitch width = 80)
                if y < 26.6:
                    zones["left_channel"] += 1
                elif y < 53.3:
                    zones["central_channel"] += 1
                else:
                    zones["right_channel"] += 1
        
        # Calculate shot creation metrics
        total_shots = len(shots)
        key_pass_shots = sum(1 for s in shot_sequences if s["key_pass"] is not None)
        key_pass_percentage = (key_pass_shots / total_shots * 100) if total_shots > 0 else 0
        
        return {
            "match_id": match_id,
            "team_id": team_id,
            "team_name": team_name,
            "total_shots": total_shots,
            "key_pass_shots": key_pass_shots,
            "key_pass_percentage": key_pass_percentage,
            "shot_sequences": shot_sequences,
            "key_pass_locations": key_pass_locations,
            "creation_zones": zones
        }
    except Exception as e:
        logger.error(f"Error analyzing shot creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
