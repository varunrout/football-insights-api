from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from app.util.metrics.expected_threat import ExpectedThreatModel  # Import the existing ExpectedThreatModel
from app.util.metrics.ppda import calculate_ppda as ppda_calculator  # Import the comprehensive PPDA calculator


def load_xt_model(model_path=None):
    """Load the trained xT model from disk, or generate a default if missing."""
    if model_path is None:
        model_path = Path('data_cache/metrics/xt_model.pkl')

    if not model_path.exists():
        # Auto-generate a default xT model and save it
        model_path.parent.mkdir(parents=True, exist_ok=True)
        xt_model = ExpectedThreatModel()
        xt_model.initialize()  # Default theoretical grid
        xt_model.save(str(model_path))
        import logging
        logging.getLogger(__name__).warning(f"xT model not found, generated default and saved to {model_path}.")
        return xt_model

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        # If the loaded object is a dict (old format), convert to ExpectedThreatModel
        if isinstance(data, dict):
            xt_model = ExpectedThreatModel(grid_size=data.get('grid_size', (12, 8)))
            xt_model.pitch_length = data.get('pitch_length', 120)
            xt_model.pitch_width = data.get('pitch_width', 80)
            xt_model.cell_length = xt_model.pitch_length / xt_model.grid_size[0]
            xt_model.cell_width = xt_model.pitch_width / xt_model.grid_size[1]
            xt_model.xt_grid = data['xt_grid']
            # Overwrite with new format for future loads
            xt_model.save(str(model_path))
            return xt_model
        return data


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
    from app.util.metrics.expected_threat import ExpectedThreatModel
    import logging
    logger = logging.getLogger(__name__)
    if xt_model is None:
        try:
            xt_model = load_xt_model()
        except FileNotFoundError:
            raise ValueError("No xT model provided and none found on disk. Please provide an xT model.")
    # Type check: ensure xt_model is an ExpectedThreatModel
    if not isinstance(xt_model, ExpectedThreatModel):
        logger.warning(f"xt_model is not an ExpectedThreatModel (got {type(xt_model)}), reloading.")
        xt_model = load_xt_model()
        if not isinstance(xt_model, ExpectedThreatModel):
            raise ValueError(f"xt_model is not a valid ExpectedThreatModel after reload (got {type(xt_model)})")
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
    teams = events_df['team_name'].unique()
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
    passes = events_df[events_df['type_name'] == 'Pass'].copy()
    
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


def calculate_basic_match_metrics(events_df):
    """
    Calculate basic match metrics for each team from an events DataFrame.
    Returns a dict of metrics per team_id.
    """
    teams = events_df['team_id'].unique()
    metrics = {}
    for team in teams:
        team_events = events_df[events_df['team_id'] == team]
        opposition_events = events_df[events_df['team_id'] != team]

        # Possession
        team_possessions = len(events_df[events_df['possession_team_id'] == team]['possession'].unique())
        total_possessions = len(events_df['possession'].unique())
        possession_pct = team_possessions / total_possessions * 100 if total_possessions > 0 else 0

        # Passing
        passes = team_events[team_events['type_name'] == 'Pass']
        passes_attempted = len(passes)
        passes_completed = len(passes[passes['pass_outcome'].isna()]) if 'pass_outcome' in passes else 0
        pass_completion = passes_completed / passes_attempted * 100 if passes_attempted > 0 else 0

        # Shooting
        shots = team_events[team_events['type_name'] == 'Shot']
        goals = len(shots[shots['shot_outcome'] == 'Goal']) if 'shot_outcome' in shots else 0
        shots_on_target = len(shots[shots['shot_outcome'].isin(['On Target', 'Goal'])]) if 'shot_outcome' in shots else 0
        xg = shots['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in shots else 0

        # Defensive
        pressures = len(team_events[team_events['type_name'] == 'Pressure'])
        tackles = len(team_events[team_events['type_name'] == 'Tackle'])
        interceptions = len(team_events[team_events['type_name'] == 'Interception'])

        # PPDA (Passes allowed Per Defensive Action)
        try:
            ppda = calculate_ppda(events_df, team)
        except Exception:
            ppda = None

        metrics[team] = {
            'possession_pct': possession_pct,
            'passes_attempted': passes_attempted,
            'passes_completed': passes_completed,
            'pass_completion': pass_completion,
            'shots': len(shots),
            'goals': goals,
            'shots_on_target': shots_on_target,
            'xg': xg,
            'pressures': pressures,
            'tackles': tackles,
            'interceptions': interceptions,
            'ppda': ppda
        }
    return metrics


def get_player_xt_contributions(events_df, xt_model=None):
    """
    Aggregate xT contributions for each player from an events DataFrame.
    Returns a DataFrame with player_id, team_name, total_xt_added, positive_actions, avg_xt_per_action.
    """
    import pandas as pd
    from collections import defaultdict
    if xt_model is None:
        xt_model = load_xt_model()
    events_with_xt = calculate_xt_added(events_df, xt_model)
    player_xt_contributions = defaultdict(float)
    player_xt_count = defaultdict(int)
    player_teams = {}
    for _, event in events_with_xt.iterrows():
        if pd.notna(event.get('player_id')) and event.get('xt_added', 0) > 0:
            player = event['player_id']
            player_xt_contributions[player] += event['xt_added']
            player_xt_count[player] += 1
            player_teams[player] = event.get('team_name', 'Unknown')
    player_xt_data = []
    for player, xt_sum in player_xt_contributions.items():
        count = player_xt_count[player]
        team = player_teams.get(player, 'Unknown')
        player_xt_data.append({
            'player_id': player,
            'team_name': team,
            'total_xt_added': xt_sum,
            'positive_actions': count,
            'avg_xt_per_action': xt_sum / count if count > 0 else 0
        })
    return pd.DataFrame(player_xt_data)
