from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from app.util.metrics.expected_threat import ExpectedThreatModel  # Import the existing ExpectedThreatModel
from app.util.metrics.ppda import calculate_ppda as ppda_calculator  # Import the comprehensive PPDA calculator


def load_xt_model(model_path=None):
    """Load the trained xT model from disk."""
    if model_path is None:
        model_path = Path('data_cache/metrics/xt_model.pkl')
    
    if not model_path.exists():
        raise FileNotFoundError(f"xT model not found at {model_path}. Run the metric engineering notebook first.")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def calculate_xt_added(events_df, xt_model=None):
    """
    Calculate the xT added by each action in the events DataFrame.

    This is now a wrapper around the more comprehensive implementation from ExpectedThreatModel.

    Args:
        events_df: DataFrame of match events
        xt_model: Expected Threat model (will load from disk if None)

    Returns:
        DataFrame with xT values added
    """
    if xt_model is None:
        try:
            xt_model = load_xt_model()
        except FileNotFoundError:
            raise ValueError("No xT model provided and none found on disk. Please provide an xT model.")
    
    # Use the ExpectedThreatModel's comprehensive implementation
    events_with_xt = xt_model.calculate_xt_for_match(events_df)

    # Rename columns to match expected format for backward compatibility
    if 'xt_value' in events_with_xt.columns:
        events_with_xt = events_with_xt.rename(columns={'xt_value': 'xt_added'})

    return events_with_xt


def calculate_ppda(events_df, team, opposition_half_only=True):
    """Calculate PPDA (Passes Per Defensive Action) for a team.

    This is a wrapper around the more comprehensive implementation from ppda.py.

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
        return float('inf')  # Return infinity if we can't find the opposition team

    # Call the comprehensive PPDA calculator
    ppda_results = ppda_calculator(events_df, opposition_thirds=opposition_half_only)

    # Determine if the team is "home" or "away" in the PPDA results
    # Note: In the comprehensive implementation, the first team is considered "home"
    is_home = teams[0] == team if len(teams) > 0 else True
    team_key = "home" if is_home else "away"

    # Return just the PPDA value for the specified team
    return ppda_results[team_key]["ppda"]


def identify_progressive_passes(events_df, distance_threshold=10):
    """Identify progressive passes in the events DataFrame.
    
    Args:
        events_df: DataFrame of match events
        distance_threshold: Minimum distance (in meters) the pass must progress toward goal
        
    Returns:
        DataFrame of progressive passes
    """
    # Filter for passes only
    passes = events_df[events_df['type'] == 'Pass'].copy()
    
    # Calculate forward progression
    progressive_passes = []
    
    for idx, pass_event in passes.iterrows():
        if isinstance(pass_event['location'], list) and isinstance(pass_event['pass_end_location'], list):
            start_x = pass_event['location'][0]
            end_x = pass_event['pass_end_location'][0]

            # Calculate forward progression
            progression = end_x - start_x

            if progression >= distance_threshold:
                pass_event['progression'] = progression
                progressive_passes.append(pass_event)

    if progressive_passes:
        progressive_df = pd.DataFrame(progressive_passes)
        return progressive_df
    else:
        return pd.DataFrame()
