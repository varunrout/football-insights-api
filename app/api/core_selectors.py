from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from app.util.football_data_manager import FootballDataManager
import pandas as pd

router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/competitions")
async def list_competitions(
    only_with_360: bool = Query(True, description="Only competitions with 360 data"),
    exclude_women: bool = Query(True, description="Exclude women's competitions"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List available competitions
    """
    try:
        df = fdm.get_competitions(only_with_360=only_with_360, exclude_women=exclude_women)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing competitions: {str(e)}")

@router.get("/seasons")
async def list_seasons(
    competition_id: int = Query(..., description="Competition ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List seasons for a competition
    """
    try:
        df = fdm.get_competitions()
        comp = df[df['competition_id'] == competition_id]
        if comp.empty:
            return []
        # Return all unique seasons for this competition
        return comp[['season_id', 'season_name']].drop_duplicates().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing seasons: {str(e)}")

@router.get("/teams")
async def list_teams(
    competition_id: int = Query(..., description="Competition ID"),
    season_id: int = Query(..., description="Season ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List teams for a season
    """
    try:
        matches = fdm.get_matches(competition_id, season_id)
        teams = set(matches['home_team_id']).union(set(matches['away_team_id']))
        team_info = []
        for tid in teams:
            row = matches[(matches['home_team_id'] == tid) | (matches['away_team_id'] == tid)].iloc[0]
            name = row['home_team'] if row['home_team_id'] == tid else row['away_team']
            team_info.append({"team_id": tid, "team_name": name})
        return team_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing teams: {str(e)}")

@router.get("/matches")
async def list_matches(
    competition_id: int = Query(..., description="Competition ID"),
    season_id: int = Query(..., description="Season ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List matches for a competition/season, optionally filtered by team
    """
    try:
        matches = fdm.get_matches(competition_id, season_id)
        if team_id:
            matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
        return matches.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing matches: {str(e)}")

@router.get("/players")
async def list_players(
    competition_id: int = Query(..., description="Competition ID"),
    season_id: int = Query(..., description="Season ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID (optional)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List players for a competition/season/team
    """
    try:
        if team_id:
            events = fdm.get_events_for_team(competition_id, season_id, team_id)
        else:
            matches = fdm.get_matches(competition_id, season_id)
            all_events = [fdm.get_events(match['match_id']) for _, match in matches.iterrows()]
            events = pd.concat(all_events) if all_events else None
        if events is None or events.empty:
            return []
        players = events[['player', 'player_id']].drop_duplicates().to_dict(orient="records")
        return players
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing players: {str(e)}")
