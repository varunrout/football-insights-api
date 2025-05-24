from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict


class ExpectedThreatModel:
    """A grid-based Expected Threat (xT) model."""
    
    def __init__(self, n_grid_cells_x=12, n_grid_cells_y=8):
        """Initialize the xT model with grid dimensions."""
        self.n_grid_cells_x = n_grid_cells_x
        self.n_grid_cells_y = n_grid_cells_y
        self.cell_length_x = 120 / n_grid_cells_x
        self.cell_length_y = 80 / n_grid_cells_y
        self.grid = np.zeros((n_grid_cells_x, n_grid_cells_y))
        self.is_trained = False
    
    def _get_cell_indices(self, x, y):
        """Convert pitch coordinates to grid cell indices."""
        # Ensure coordinates are within pitch boundaries
        x = max(0, min(119.9, x))
        y = max(0, min(79.9, y))
        
        # Calculate cell indices
        i = int(x / self.cell_length_x)
        j = int(y / self.cell_length_y)
        
        return i, j
    
    def train(self, events_df):
        """Train the xT model using event data."""
        # Initialize counters for shots and goals from each cell
        move_counts = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y, self.n_grid_cells_x, self.n_grid_cells_y))
        shot_counts = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y))
        goal_counts = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y))
        cell_count = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y))
        
        # Count shots and goals from each cell
        for _, event in events_df[events_df['type'] == 'Shot'].iterrows():
            if isinstance(event['location'], list) and len(event['location']) >= 2:
                x, y = event['location'][0], event['location'][1]
                i, j = self._get_cell_indices(x, y)
                
                shot_counts[i, j] += 1
                cell_count[i, j] += 1
                
                if event['shot_outcome'] == 'Goal':
                    goal_counts[i, j] += 1
        
        # Count moves between cells (passes, carries)
        for _, event in events_df[(events_df['type'] == 'Pass') | (events_df['type'] == 'Carry')].iterrows():
            if isinstance(event['location'], list) and len(event['location']) >= 2:
                start_x, start_y = event['location'][0], event['location'][1]
                
                # Get end location based on event type
                end_loc = None
                if event['type'] == 'Pass' and isinstance(event['pass_end_location'], list):
                    end_loc = event['pass_end_location']
                elif event['type'] == 'Carry' and isinstance(event['carry_end_location'], list):
                    end_loc = event['carry_end_location']
                
                if end_loc and len(end_loc) >= 2:
                    end_x, end_y = end_loc[0], end_loc[1]
                    
                    # Get cell indices
                    i_start, j_start = self._get_cell_indices(start_x, start_y)
                    i_end, j_end = self._get_cell_indices(end_x, end_y)
                    
                    # Increment move count
                    move_counts[i_start, j_start, i_end, j_end] += 1
                    cell_count[i_start, j_start] += 1
        
        # Calculate shot probability for each cell
        shot_probability = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y))
        for i in range(self.n_grid_cells_x):
            for j in range(self.n_grid_cells_y):
                if cell_count[i, j] > 0:
                    shot_probability[i, j] = shot_counts[i, j] / cell_count[i, j]
        
        # Calculate goal probability given a shot
        goal_given_shot = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y))
        for i in range(self.n_grid_cells_x):
            for j in range(self.n_grid_cells_y):
                if shot_counts[i, j] > 0:
                    goal_given_shot[i, j] = goal_counts[i, j] / shot_counts[i, j]
        
        # Calculate move probability matrix
        move_probability = np.zeros((self.n_grid_cells_x, self.n_grid_cells_y, self.n_grid_cells_x, self.n_grid_cells_y))
        for i_start in range(self.n_grid_cells_x):
            for j_start in range(self.n_grid_cells_y):
                if cell_count[i_start, j_start] > 0:
                    for i_end in range(self.n_grid_cells_x):
                        for j_end in range(self.n_grid_cells_y):
                            move_probability[i_start, j_start, i_end, j_end] = (
                                move_counts[i_start, j_start, i_end, j_end] / cell_count[i_start, j_start]
                            )
        
        # Initialize xT grid with direct shot value (probability of shot * probability of goal given shot)
        self.grid = shot_probability * goal_given_shot
        
        # Use dynamic programming to calculate expected threat for each cell
        for _ in range(5):  # Iterate a few times to converge
            new_grid = np.copy(self.grid)
            
            for i in range(self.n_grid_cells_x):
                for j in range(self.n_grid_cells_y):
                    # Direct value from shooting
                    direct_value = shot_probability[i, j] * goal_given_shot[i, j]
                    
                    # Value from moving to other cells
                    move_value = 0
                    for i_end in range(self.n_grid_cells_x):
                        for j_end in range(self.n_grid_cells_y):
                            move_value += move_probability[i, j, i_end, j_end] * self.grid[i_end, j_end]
                    
                    # Total value is max of direct value and move value
                    new_grid[i, j] = max(direct_value, move_value)
            
            self.grid = new_grid
        
        self.is_trained = True
        return self.grid
    
    def get_value(self, x, y):
        """Get the xT value for a location on the pitch."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting values")
        
        i, j = self._get_cell_indices(x, y)
        return self.grid[i, j]
    
    def calculate_move_value(self, start_x, start_y, end_x, end_y):
        """Calculate the value added by moving from one location to another."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating move values")
        
        i_start, j_start = self._get_cell_indices(start_x, start_y)
        i_end, j_end = self._get_cell_indices(end_x, end_y)
        
        # Value added is the difference in xT between the end and start locations
        return self.grid[i_end, j_end] - self.grid[i_start, j_start]


def load_xt_model(model_path=None):
    """Load the trained xT model from disk."""
    if model_path is None:
        model_path = Path('data_cache/metrics/xt_model.pkl')
    
    if not model_path.exists():
        raise FileNotFoundError(f"xT model not found at {model_path}. Run the metric engineering notebook first.")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def calculate_xt_added(events_df, xt_model=None):
    """Calculate the xT added by each action in the events DataFrame."""
    if xt_model is None:
        try:
            xt_model = load_xt_model()
        except FileNotFoundError:
            raise ValueError("No xT model provided and none found on disk. Please provide an xT model.")
    
    events_with_xt = events_df.copy()
    events_with_xt['xt_start'] = 0.0
    events_with_xt['xt_end'] = 0.0
    events_with_xt['xt_added'] = 0.0
    
    # Calculate xT values for passes and carries
    for idx, event in events_df.iterrows():
        if event['type'] in ['Pass', 'Carry']:
            # Get start location
            if isinstance(event['location'], list) and len(event['location']) >= 2:
                start_x, start_y = event['location'][0], event['location'][1]
                xt_start = xt_model.get_value(start_x, start_y)
                events_with_xt.loc[idx, 'xt_start'] = xt_start
                
                # Get end location based on event type
                end_loc = None
                if event['type'] == 'Pass' and isinstance(event['pass_end_location'], list):
                    end_loc = event['pass_end_location']
                elif event['type'] == 'Carry' and isinstance(event['carry_end_location'], list):
                    end_loc = event['carry_end_location']
                
                if end_loc and len(end_loc) >= 2:
                    end_x, end_y = end_loc[0], end_loc[1]
                    xt_end = xt_model.get_value(end_x, end_y)
                    events_with_xt.loc[idx, 'xt_end'] = xt_end
                    
                    # Calculate xT added
                    xt_added = xt_end - xt_start
                    events_with_xt.loc[idx, 'xt_added'] = xt_added
    
    return events_with_xt


def calculate_ppda(events_df, team, opposition_half_only=True):
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
        return None
    
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


def calculate_basic_match_metrics(events_df):
    """Calculate basic match metrics from events DataFrame."""
    teams = events_df['team'].unique()
    metrics = {}
    
    # Calculate metrics for each team
    for team in teams:
        team_events = events_df[events_df['team'] == team]
        opposition_events = events_df[events_df['team'] != team]
        
        # Possessions
        team_possessions = len(events_df[events_df['possession_team'] == team]['possession'].unique())
        total_possessions = len(events_df['possession'].unique())
        
        # Passes
        passes = team_events[team_events['type'] == 'Pass']
        successful_passes = passes[passes['pass_outcome'].isna()]
        
        # Shots
        shots = team_events[team_events['type'] == 'Shot']
        goals = shots[shots['shot_outcome'] == 'Goal']
        on_target = shots[shots['shot_outcome'].isin(['Goal', 'Saved'])]
        
        # Defensive actions
        pressures = team_events[team_events['type'] == 'Pressure']
        tackles = team_events[team_events['type'] == 'Duel']
        interceptions = team_events[team_events['type'] == 'Interception']
        
        # Calculate PPDA (Passes allowed per defensive action in opposition half)
        opp_passes_own_half = opposition_events[
            (opposition_events['type'] == 'Pass') & 
            (opposition_events['location'].apply(lambda x: x[0] < 60 if isinstance(x, list) and len(x) >= 2 else False))
        ]
        
        def_actions_opp_half = team_events[
            (team_events['type'].isin(['Pressure', 'Duel', 'Interception'])) & 
            (team_events['location'].apply(lambda x: x[0] >= 60 if isinstance(x, list) and len(x) >= 2 else False))
        ]
        
        ppda = len(opp_passes_own_half) / max(len(def_actions_opp_half), 1)  # Avoid division by zero
        
        # Store metrics
        metrics[team] = {
            'possession_pct': team_possessions / total_possessions * 100 if total_possessions > 0 else 0,
            'passes_attempted': len(passes),
            'passes_completed': len(successful_passes),
            'pass_completion': len(successful_passes) / len(passes) * 100 if len(passes) > 0 else 0,
            'shots': len(shots),
            'goals': len(goals),
            'shots_on_target': len(on_target),
            'xg': shots['shot_statsbomb_xg'].sum(),
            'pressures': len(pressures),
            'tackles': len(tackles),
            'interceptions': len(interceptions),
            'ppda': ppda
        }
    
    return metrics


def get_player_xt_contributions(events_df, xt_model=None, min_actions=5):
    """Calculate xT contribution by player across matches.
    
    Args:
        events_df: DataFrame of match events
        xt_model: Expected Threat model (will load from disk if None)
        min_actions: Minimum number of actions required to include a player
        
    Returns:
        DataFrame with player xT contributions
    """
    if xt_model is None:
        try:
            xt_model = load_xt_model()
        except FileNotFoundError:
            raise ValueError("No xT model provided and none found on disk. Please provide an xT model.")
    
    # Calculate xT for all events
    events_with_xt = calculate_xt_added(events_df, xt_model)
    
    # Aggregate by player
    player_xt_contributions = defaultdict(float)
    player_xt_count = defaultdict(int)
    player_teams = {}
    
    for _, event in events_with_xt.iterrows():
        if pd.notna(event['player']) and event['xt_added'] > 0:
            player_name = event['player']
            player_xt_contributions[player_name] += event['xt_added']
            player_xt_count[player_name] += 1
            player_teams[player_name] = event['team']
    
    # Convert to DataFrame
    player_xt_data = []
    for player, xt_sum in player_xt_contributions.items():
        count = player_xt_count[player]
        if count >= min_actions:
            team = player_teams.get(player, 'Unknown')
            player_xt_data.append({
                'player': player,
                'team': team,
                'total_xt_added': xt_sum,
                'positive_actions': count,
                'avg_xt_per_action': xt_sum / count
            })
    
    return pd.DataFrame(player_xt_data)
