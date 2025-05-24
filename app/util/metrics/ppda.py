"""
PPDA (Passes Per Defensive Action) metric implementation.

This module provides functions to calculate PPDA, which is a measure of pressing intensity.
Lower values indicate more intense pressing (fewer passes allowed before a defensive action).
"""

import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_ppda(events_df: pd.DataFrame, team: str, opposition_half_only: bool = True) -> float:
    """Calculate PPDA (Passes Allowed Per Defensive Action) for a team.
    
    Args:
        events_df: DataFrame of match events
        team: Team to calculate PPDA for
        opposition_half_only: If True, only consider defensive actions in the opposition half
        
    Returns:
        PPDA value
    """
    # Get opposition team
    teams = events_df['team'].unique()
    opposition_team = [t for t in teams if t != team][0] if len(teams) > 1 else None
    
    if not opposition_team:
        return float('inf')  # Return infinity if no opposition team found
    
    # Get opposition passes
    opposition_passes = events_df[(events_df['team'] == opposition_team) & (events_df['type'] == 'Pass')]
    
    # Filter passes to only those in team's defensive half if opposition_half_only=True
    if opposition_half_only:
        # For the opposition, their attacking half is the team's defensive half
        opposition_passes = opposition_passes[
            opposition_passes['location'].apply(
                lambda x: x[0] >= 60 if isinstance(x, list) and len(x) >= 2 else False
            )
        ]
    
    # Get defensive actions by team
    defensive_actions = events_df[
        (events_df['team'] == team) &
        (events_df['type'].isin(['Pressure', 'Duel', 'Interception']))
    ]
    
    # Filter defensive actions to only those in opposition half if opposition_half_only=True
    if opposition_half_only:
        defensive_actions = defensive_actions[
            defensive_actions['location'].apply(
                lambda x: x[0] >= 60 if isinstance(x, list) and len(x) >= 2 else False
            )
        ]
    
    # Calculate PPDA
    num_opposition_passes = len(opposition_passes)
    num_defensive_actions = len(defensive_actions)
    
    ppda = num_opposition_passes / num_defensive_actions if num_defensive_actions > 0 else float('inf')
    
    return ppda

def calculate_team_match_ppda(events_df: pd.DataFrame, opposition_half_only: bool = True) -> Dict[str, float]:
    """Calculate PPDA for both teams in a match.
    
    Args:
        events_df: DataFrame of match events
        opposition_half_only: If True, only consider defensive actions in the opposition half
        
    Returns:
        Dictionary mapping team names to PPDA values
    """
    teams = events_df['team'].unique()
    
    ppda_values = {}
    for team in teams:
        ppda_values[team] = calculate_ppda(events_df, team, opposition_half_only)
    
    return ppda_values
