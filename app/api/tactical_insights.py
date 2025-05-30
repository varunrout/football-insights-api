from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from app.util.football_data_manager import FootballDataManager
from app.services.metrics_engine import MetricsEngine
from app.services.tactical_analyzer import TacticalAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

def get_metrics_engine(fdm: FootballDataManager = Depends(get_data_manager)):
    """Dependency to get MetricsEngine instance"""
    return MetricsEngine(fdm)

def get_tactical_analyzer(metrics_engine: MetricsEngine = Depends(get_metrics_engine)):
    """Dependency to get TacticalAnalyzer instance"""
    return TacticalAnalyzer(metrics_engine)

@router.get("/defensive-metrics")
async def get_defensive_metrics(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get defensive metrics for a team
    Returns PPDA, defensive actions, and pressure metrics
    """
    try:
        if not team_id:
            return {"error": "team_id is required."}
        if not match_id and (not competition_id or not season_id):
            return {"error": "competition_id and season_id are required if match_id is not provided."}
        metrics = analyzer.get_defensive_metrics(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )
        return metrics
    except Exception as e:
        logger.error(f"Error getting defensive metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/offensive-metrics")
async def get_offensive_metrics(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get offensive metrics for a team
    Returns possession metrics, attacking patterns, and shot creation
    """
    try:
        if not team_id:
            return {"error": "team_id is required."}
        if not match_id and (not competition_id or not season_id):
            return {"error": "competition_id and season_id are required if match_id is not provided."}
        metrics = analyzer.get_offensive_metrics(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )
        return metrics
    except Exception as e:
        logger.error(f"Error getting offensive metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/pass-network")
async def get_pass_network(
    team_id: int = Query(..., description="Team ID"),
    match_id: int = Query(..., description="Match ID"),
    min_passes: int = Query(3, description="Minimum number of passes between players to include"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get pass network data for a team in a specific match

    Returns nodes (players) and edges (passes) for network visualization
    """
    try:
        # Get pass network using the tactical analyzer
        network = analyzer.get_pass_network(
            team_id=team_id,
            match_id=match_id,
            min_passes=min_passes
        )

        return network
    except Exception as e:
        logger.error(f"Error getting pass network: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/build-up-analysis")
async def get_build_up_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get build-up play analysis for a team

    Returns metrics on how the team progresses the ball from defense to attack
    """
    try:
        # Get build-up analysis using the tactical analyzer
        analysis = analyzer.get_build_up_analysis(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting build-up analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/pressing-analysis")
async def get_pressing_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get pressing analysis for a team

    Returns metrics on the team's pressing approach and effectiveness
    """
    try:
        # Get pressing analysis using the tactical analyzer
        analysis = analyzer.get_pressing_analysis(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting pressing analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/transitions")
async def get_transition_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get transition analysis for a team

    Returns metrics on the team's offensive and defensive transitions
    """
    try:
        # Get transition analysis using the tactical analyzer
        analysis = analyzer.get_transition_analysis(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting transition analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/set-pieces")
async def get_set_piece_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: Optional[int] = Query(None, description="Match ID (if None, returns data for all matches)"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get set piece analysis for a team

    Returns metrics on the team's set piece effectiveness (corners, free kicks, etc.)
    """
    try:
        # Get set piece analysis using the tactical analyzer
        analysis = analyzer.get_set_piece_analysis(
            team_id=team_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting set piece analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/formation-analysis")
async def get_formation_analysis(
    team_id: int = Query(..., description="Team ID"),
    match_id: int = Query(..., description="Match ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get formation analysis for a team in a specific match

    Returns the team's formation, player positions, and formation shifts over time
    """
    try:
        # Get formation analysis using the tactical analyzer
        analysis = analyzer.get_formation_analysis(
            team_id=team_id,
            match_id=match_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting formation analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/team-style")
async def get_team_style(
    team_id: int = Query(..., description="Team ID"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Get team playing style analysis

    Returns metrics and classifications of the team's playing style
    """
    try:
        # Get team style analysis using the tactical analyzer
        analysis = analyzer.get_team_style(
            team_id=team_id,
            competition_id=competition_id,
            season_id=season_id
        )

        return analysis
    except Exception as e:
        logger.error(f"Error getting team style analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/style-comparison")
async def get_style_comparison(
    team_id1: int = Query(..., description="First team ID"),
    team_id2: int = Query(..., description="Second team ID"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    analyzer: TacticalAnalyzer = Depends(get_tactical_analyzer)
):
    """
    Compare playing styles between two teams

    Returns comparative metrics and style differences
    """
    try:
        # Get style comparison using the tactical analyzer
        comparison = analyzer.compare_team_styles(
            team_id1=team_id1,
            team_id2=team_id2,
            competition_id=competition_id,
            season_id=season_id
        )

        return comparison
    except Exception as e:
        logger.error(f"Error getting style comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
