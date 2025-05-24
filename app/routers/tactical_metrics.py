from fastapi import APIRouter, HTTPException, Query, Depends, Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from app.services.metric_calculator import (
    calculate_basic_match_metrics,
    calculate_ppda,
    identify_progressive_passes
)
from app.util.football_data_manager import FootballDataManager

router = APIRouter(
    prefix="/api/tactical",
    tags=["Tactical Metrics"],
    responses={404: {"description": "Not found"}},
)

# Initialize data manager
data_manager = FootballDataManager()


@router.get("/match-metrics/{match_id}")
async def get_match_metrics(match_id: int):
    """Get basic tactical metrics for a specific match."""
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
        
        # Calculate basic match metrics
        metrics = calculate_basic_match_metrics(events_df)
        
        # Get team names
        teams = events_df["team"].unique()
        
        # Calculate additional metrics
        team_metrics = {}
        for team in teams:
            # Get PPDA for each team
            ppda = calculate_ppda(events_df, team)
            
            # Find progressive passes
            team_events = events_df[events_df["team"] == team]
            progressive_passes = identify_progressive_passes(team_events)
            
            # Calculate shots metrics
            shots = team_events[team_events["type"] == "Shot"]
            shots_data = {
                "total_shots": len(shots),
                "shots_on_target": len(shots[shots["shot_outcome"].isin(["Goal", "Saved"])]),
                "goals": len(shots[shots["shot_outcome"] == "Goal"]),
                "xg": float(shots["shot_statsbomb_xg"].sum())
            }
            
            # Add to team metrics
            team_metrics[team] = {
                **metrics[team],  # Include basic metrics
                "ppda": float(ppda),  # Add PPDA
                "progressive_passes": len(progressive_passes),  # Add progressive passes count
                "shots_data": shots_data  # Add shots data
            }
        
        return {
            "match_info": {
                "match_id": match_id,
                "competition_id": comp_id,
                "home_team": match_data["home_team"],
                "away_team": match_data["away_team"],
                "score": match_data["score"]
            },
            "team_metrics": team_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating match metrics: {str(e)}")


@router.get("/competition-summary/{competition_id}")
async def get_competition_summary(
    competition_id: int,
    team_name: Optional[str] = None
):
    """Get tactical metrics summary for a competition or specific team within a competition."""
    try:
        # Load the dataset
        dataset_path = "data_cache/top_10_competitions"
        analysis_data = data_manager.load_analysis_dataset(dataset_path)
        
        # Check if competition exists
        if competition_id not in analysis_data["competitions"]:
            raise HTTPException(status_code=404, detail=f"Competition ID {competition_id} not found")
        
        comp_data = analysis_data["competitions"][competition_id]
        
        # Collect metrics for all matches in the competition
        all_team_metrics = {}
        match_count = {}  # Count matches per team
        
        for match_id, match_data in comp_data["matches"].items():
            if "events" not in match_data:
                continue
                
            events_df = match_data["events"]
            
            # Filter by team if specified
            if team_name is not None:
                if team_name not in events_df["team"].unique():
                    continue
                    
            # Calculate metrics for this match
            match_metrics = calculate_basic_match_metrics(events_df)
            
            # Aggregate metrics by team
            for team, metrics in match_metrics.items():
                if team_name is not None and team != team_name:
                    continue
                    
                if team not in all_team_metrics:
                    all_team_metrics[team] = {k: 0 for k in metrics.keys()}
                    match_count[team] = 0
                
                # Sum metrics
                for metric, value in metrics.items():
                    all_team_metrics[team][metric] += value
                
                # Increment match count
                match_count[team] += 1
                
                # Calculate PPDA for this team in this match
                ppda = calculate_ppda(events_df, team)
                if "ppda_sum" not in all_team_metrics[team]:
                    all_team_metrics[team]["ppda_sum"] = 0
                
                all_team_metrics[team]["ppda_sum"] += ppda
        
        # Calculate averages
        avg_team_metrics = {}
        for team, metrics in all_team_metrics.items():
            if match_count[team] == 0:
                continue
                
            avg_metrics = {k: v / match_count[team] for k, v in metrics.items()}
            avg_metrics["matches_played"] = match_count[team]
            avg_metrics["ppda"] = avg_metrics.pop("ppda_sum") / match_count[team]
            
            avg_team_metrics[team] = avg_metrics
            
        return {
            "competition_info": {
                "competition_id": competition_id,
                "name": comp_data["name"],
                "season": comp_data["season"],
                "total_matches": len(comp_data["matches"])
            },
            "team_metrics": avg_team_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating competition summary: {str(e)}")
