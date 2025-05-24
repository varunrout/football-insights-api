from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from app.services.metric_calculator import (
    load_xt_model,
    calculate_xt_added,
    get_player_xt_contributions,
    identify_progressive_passes
)
from app.util.football_data_manager import FootballDataManager

router = APIRouter(
    prefix="/api/xt",
    tags=["xT Analytics"],
    responses={404: {"description": "Not found"}},
)

# Initialize data manager
data_manager = FootballDataManager()

# Try to load the xT model
try:
    xt_model = load_xt_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    xt_model = None


@router.get("/model-status")
async def get_model_status():
    """Check if the xT model is loaded and ready."""
    return {"model_loaded": model_loaded}


@router.get("/heatmap")
async def get_xt_heatmap():
    """Get xT values for visualizing as a heatmap."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="xT model not loaded. Run the metric engineering notebook first.")
    
    # Generate grid coordinates
    n_grid_cells_x = xt_model.n_grid_cells_x
    n_grid_cells_y = xt_model.n_grid_cells_y
    cell_length_x = xt_model.cell_length_x
    cell_length_y = xt_model.cell_length_y
    
    # Create coordinates for grid centers
    x_centers = [cell_length_x/2 + i*cell_length_x for i in range(n_grid_cells_x)]
    y_centers = [cell_length_y/2 + i*cell_length_y for i in range(n_grid_cells_y)]
    
    # Create grid data for frontend visualization
    grid_data = []
    for i in range(n_grid_cells_x):
        for j in range(n_grid_cells_y):
            grid_data.append({
                "x": x_centers[i],
                "y": y_centers[j],
                "value": float(xt_model.grid[i, j])
            })
    
    return {
        "grid_data": grid_data,
        "grid_dimensions": {
            "x_cells": n_grid_cells_x,
            "y_cells": n_grid_cells_y,
            "cell_width": cell_length_x,
            "cell_height": cell_length_y
        }
    }


@router.get("/player-rankings")
async def get_player_xt_rankings(
    competition_id: Optional[int] = None,
    match_id: Optional[int] = None,
    team_name: Optional[str] = None,
    min_actions: int = Query(5, ge=1),
    limit: int = Query(100, ge=1, le=500)
):
    """Get player xT contribution rankings."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="xT model not loaded. Run the metric engineering notebook first.")
    
    # Load the dataset (or the specific competition/match if specified)
    try:
        dataset_path = "data_cache/top_10_competitions"
        analysis_data = data_manager.load_analysis_dataset(dataset_path)
        
        # Prepare events data based on filters
        events_data = []
        
        for comp_id, comp_data in analysis_data["competitions"].items():
            # Filter by competition if specified
            if competition_id is not None and comp_id != competition_id:
                continue
                
            for match_id_iter, match_data in comp_data["matches"].items():
                # Filter by match if specified
                if match_id is not None and match_id_iter != match_id:
                    continue
                    
                if "events" in match_data:
                    events = match_data["events"]
                    
                    # Filter by team if specified
                    if team_name is not None:
                        events = events[events["team"] == team_name]
                    
                    events_data.append(events)
        
        if not events_data:
            return {"players": [], "message": "No data found matching the specified filters."}
        
        # Combine events data
        all_events = pd.concat(events_data)
        
        # Calculate xT contributions
        player_contributions = get_player_xt_contributions(all_events, xt_model, min_actions=min_actions)
        
        # Sort by total xT added and limit results
        top_players = player_contributions.sort_values("total_xt_added", ascending=False).head(limit)
        
        # Convert to list of dictionaries for JSON response
        players = top_players.to_dict(orient="records")
        
        return {"players": players, "total_players": len(players)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating player xT rankings: {str(e)}")


@router.get("/match-analysis/{match_id}")
async def get_match_xt_analysis(match_id: int):
    """Get xT analysis for a specific match."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="xT model not loaded. Run the metric engineering notebook first.")
    
    try:
        # Load the dataset
        dataset_path = "data_cache/top_10_competitions"
        analysis_data = data_manager.load_analysis_dataset(dataset_path)
        
        # Find the match
        match_data = None
        comp_id = None
        
        for c_id, comp_data in analysis_data["competitions"].items():
            if match_id in comp_data["matches"]:
                match_data = comp_data["matches"][match_id]
                comp_id = c_id
                break
        
        if match_data is None:
            raise HTTPException(status_code=404, detail=f"Match ID {match_id} not found")
        
        # Check if events data is available
        if "events" not in match_data:
            raise HTTPException(status_code=404, detail=f"Events data for match ID {match_id} not found")
        
        events_df = match_data["events"]
        
        # Calculate xT added for all events
        events_with_xt = calculate_xt_added(events_df, xt_model)
        
        # Find progressive passes
        progressive_passes = identify_progressive_passes(events_df)
        
        # Get teams
        teams = events_df["team"].unique()
        
        # Calculate xT summary by team
        team_summaries = {}
        for team in teams:
            team_events = events_with_xt[events_with_xt["team"] == team]
            
            # Total xT added by the team
            total_xt = team_events[team_events["xt_added"] > 0]["xt_added"].sum()
            
            # xT added by event type
            xt_by_type = team_events[team_events["xt_added"] > 0].groupby("type")["xt_added"].sum().to_dict()
            
            # Top players by xT added
            top_players = (
                team_events[team_events["xt_added"] > 0]
                .groupby("player")["xt_added"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .to_dict()
            )
            
            # Calculate progressive passes for team
            team_progressive = progressive_passes[progressive_passes["team"] == team]
            prog_pass_count = len(team_progressive)
            
            team_summaries[team] = {
                "total_xt_added": float(total_xt),
                "xt_by_event_type": xt_by_type,
                "top_players_by_xt": top_players,
                "progressive_passes": prog_pass_count
            }
        
        # Get top xT actions
        top_actions = (
            events_with_xt[events_with_xt["xt_added"] > 0]
            .sort_values("xt_added", ascending=False)
            .head(10)
        )
        
        top_actions_list = []
        for _, action in top_actions.iterrows():
            top_actions_list.append({
                "player": action["player"],
                "team": action["team"],
                "type": action["type"],
                "minute": int(action["minute"]),
                "xt_added": float(action["xt_added"]),
                "start_location": action["location"],
                "end_location": action.get("pass_end_location") if action["type"] == "Pass" else action.get("carry_end_location")
            })
        
        return {
            "match_info": {
                "home_team": match_data["home_team"],
                "away_team": match_data["away_team"],
                "score": match_data["score"],
                "competition_id": comp_id
            },
            "team_summaries": team_summaries,
            "top_xt_actions": top_actions_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing match: {str(e)}")
