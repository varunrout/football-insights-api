"""
PPDA (Passes Per Defensive Action) calculation module.

This module implements the calculation of PPDA, a metric that quantifies defensive pressure by
measuring the number of passes an opponent is allowed to make before a defensive action is taken.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

def calculate_ppda(events_df: pd.DataFrame,
                  opposition_thirds: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate PPDA (Passes Per Defensive Action) for both teams.

    PPDA = Opposition Passes / Defensive Actions
    where Defensive Actions = Tackles + Interceptions + Challenges + Fouls

    Args:
        events_df: DataFrame containing match events
        opposition_thirds: If True, only consider passes in opposition thirds (excluding defensive third)

    Returns:
        Dictionary with PPDA values for each team
    """
    # Ensure we have a dataframe to work with
    if events_df is None or events_df.empty:
        return {"home": {"ppda": 0.0}, "away": {"ppda": 0.0}}

    # Get unique teams
    teams = events_df['team'].dropna().unique()
    if len(teams) < 2:
        return {"home": {"ppda": 0.0}, "away": {"ppda": 0.0}}

    home_team = teams[0]
    away_team = teams[1]

    # Calculate PPDA for each team
    home_ppda = _calculate_team_ppda(events_df, home_team, away_team, opposition_thirds)
    away_ppda = _calculate_team_ppda(events_df, away_team, home_team, opposition_thirds)

    return {
        "home": home_ppda,
        "away": away_ppda
    }

def _calculate_team_ppda(events_df: pd.DataFrame,
                        team: str,
                        opposition: str,
                        opposition_thirds: bool) -> Dict[str, float]:
    """
    Calculate PPDA for a specific team.

    Args:
        events_df: DataFrame containing match events
        team: Team name to calculate PPDA for
        opposition: Opposition team name
        opposition_thirds: If True, only consider passes in opposition thirds

    Returns:
        Dictionary with PPDA values and components
    """
    # Filter for opposition passes
    opposition_passes = events_df[(events_df['team'] == opposition) &
                                 (events_df['type'] == 'Pass')]

    # If considering only opposition thirds, filter passes by location
    if opposition_thirds and 'location' in opposition_passes.columns:
        # Get passes in opposition middle and final third
        # Assuming pitch is 120x80 with (0,0) at bottom left
        opposition_passes = opposition_passes[
            opposition_passes['location'].apply(
                lambda x: x[0] > 40 if isinstance(x, (list, tuple)) and len(x) >= 2 else False
            )
        ]

    # Count opposition passes
    pass_count = len(opposition_passes)

    # Filter for team defensive actions
    defensive_actions = events_df[
        (events_df['team'] == team) &
        (events_df['type'].isin(['Interception', 'Tackle', 'Foul Committed', 'Challenge']))
    ]

    # If considering only opposition thirds, filter defensive actions by location
    if opposition_thirds and 'location' in defensive_actions.columns:
        defensive_actions = defensive_actions[
            defensive_actions['location'].apply(
                lambda x: x[0] > 40 if isinstance(x, (list, tuple)) and len(x) >= 2 else False
            )
        ]

    # Count defensive actions
    action_count = len(defensive_actions)

    # Calculate PPDA (avoid division by zero)
    ppda = pass_count / action_count if action_count > 0 else float('inf')

    return {
        "ppda": ppda,
        "opposition_passes": pass_count,
        "defensive_actions": action_count,
        "interceptions": len(defensive_actions[defensive_actions['type'] == 'Interception']),
        "tackles": len(defensive_actions[defensive_actions['type'] == 'Tackle']),
        "challenges": len(defensive_actions[defensive_actions['type'] == 'Challenge']),
        "fouls": len(defensive_actions[defensive_actions['type'] == 'Foul Committed'])
    }

def calculate_ppda_timeline(events_df: pd.DataFrame,
                           window_size: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Calculate PPDA over time using a rolling window of minutes.

    Args:
        events_df: DataFrame containing match events
        window_size: Size of rolling window in minutes

    Returns:
        Dictionary with PPDA timeline for each team
    """
    # Ensure we have a dataframe to work with
    if events_df is None or events_df.empty:
        return {"home": [], "away": []}

    # Get unique teams
    teams = events_df['team'].dropna().unique()
    if len(teams) < 2:
        return {"home": [], "away": []}

    home_team = teams[0]
    away_team = teams[1]

    # Get max minute in the match
    max_minute = events_df['minute'].max()

    # Calculate PPDA for each window
    home_timeline = []
    away_timeline = []

    for start_minute in range(0, int(max_minute) + 1 - window_size, window_size):
        end_minute = start_minute + window_size

        # Filter events for this window
        window_events = events_df[(events_df['minute'] >= start_minute) &
                                 (events_df['minute'] < end_minute)]

        # Calculate PPDA for this window
        window_ppda = calculate_ppda(window_events)

        # Add to timelines
        home_timeline.append({
            "start_minute": start_minute,
            "end_minute": end_minute,
            "ppda": window_ppda["home"]["ppda"],
            "opposition_passes": window_ppda["home"]["opposition_passes"],
            "defensive_actions": window_ppda["home"]["defensive_actions"]
        })

        away_timeline.append({
            "start_minute": start_minute,
            "end_minute": end_minute,
            "ppda": window_ppda["away"]["ppda"],
            "opposition_passes": window_ppda["away"]["opposition_passes"],
            "defensive_actions": window_ppda["away"]["defensive_actions"]
        })

    return {
        "home": home_timeline,
        "away": away_timeline
    }

def calculate_team_ppda_comparison(team_events: List[pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate and compare PPDA across multiple matches for a team.

    Args:
        team_events: List of event DataFrames for matches involving the team

    Returns:
        Dictionary with PPDA comparison data
    """
    if not team_events:
        return {"match_count": 0, "avg_ppda": 0.0, "matches": []}

    match_ppda_values = []
    total_ppda = 0.0

    for match_idx, events_df in enumerate(team_events):
        # Get teams in this match
        teams = events_df['team'].dropna().unique()
        if len(teams) < 2:
            continue

        # Calculate PPDA for this match
        match_ppda = calculate_ppda(events_df)

        # Determine if team is home or away based on first event
        team_events_only = events_df[events_df['team'] == teams[0]]
        is_home = len(team_events_only) > 0

        # Get the relevant PPDA value
        ppda_value = match_ppda["home"]["ppda"] if is_home else match_ppda["away"]["ppda"]
        total_ppda += ppda_value

        # Store match PPDA data
        match_ppda_values.append({
            "match_index": match_idx,
            "ppda": ppda_value,
            "is_home": is_home,
            "opposition": teams[1] if is_home else teams[0]
        })

    # Calculate average PPDA
    avg_ppda = total_ppda / len(match_ppda_values) if match_ppda_values else 0.0

    return {
        "match_count": len(match_ppda_values),
        "avg_ppda": avg_ppda,
        "matches": match_ppda_values
    }
