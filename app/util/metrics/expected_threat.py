"""
Expected Threat (xT) model implementation.

This module implements a grid-based Expected Threat model for valuing pitch locations
and actions based on their probability of leading to goals.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    
    def _get_cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pitch coordinates to grid cell indices."""
        # Ensure coordinates are within pitch boundaries
        x = max(0, min(119.9, x))
        y = max(0, min(79.9, y))
        
        # Calculate cell indices
        i = int(x / self.cell_length_x)
        j = int(y / self.cell_length_y)
        
        return i, j
    
    def train(self, events_df: pd.DataFrame) -> np.ndarray:
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
                if event['type'] == 'Pass' and isinstance(event.get('pass_end_location'), list):
                    end_loc = event['pass_end_location']
                elif event['type'] == 'Carry' and isinstance(event.get('carry_end_location'), list):
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
    
    def get_value(self, x: float, y: float) -> float:
        """Get the xT value for a location on the pitch."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting values")
        
        i, j = self._get_cell_indices(x, y)
        return self.grid[i, j]
    
    def calculate_move_value(self, start_x: float, start_y: float, end_x: float, end_y: float) -> float:
        """Calculate the value added by moving from one location to another."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating move values")
        
        i_start, j_start = self._get_cell_indices(start_x, start_y)
        i_end, j_end = self._get_cell_indices(end_x, end_y)
        
        # Value added is the difference in xT between the end and start locations
        return self.grid[i_end, j_end] - self.grid[i_start, j_start]

def load_xt_model(model_path: Optional[str] = None) -> ExpectedThreatModel:
    """Load a trained xT model from disk."""
    if model_path is None:
        model_path = str(Path("data_cache/metrics/xt_model.pkl"))
    
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, IOError):
        logger.warning(f"xT model not found at {model_path}. Returning default model.")
        # Return a default model with pre-filled values
        model = ExpectedThreatModel()
        model.grid = np.array([
            [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.01, 0.008],
            [0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.015, 0.01],
            [0.003, 0.005, 0.008, 0.01, 0.02, 0.025, 0.025, 0.02],
            [0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03],
            [0.008, 0.01, 0.02, 0.03, 0.05, 0.06, 0.06, 0.05],
            [0.01, 0.015, 0.025, 0.04, 0.06, 0.08, 0.08, 0.06],
            [0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.1, 0.08],
            [0.02, 0.03, 0.04, 0.07, 0.1, 0.15, 0.15, 0.1],
            [0.03, 0.04, 0.06, 0.1, 0.15, 0.2, 0.2, 0.15],
            [0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.3, 0.2],
            [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.4, 0.3],
            [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.6, 0.4]
        ])
        model.is_trained = True
        return model

def calculate_xt_added(events_df: pd.DataFrame, xt_model: Optional[ExpectedThreatModel] = None) -> pd.DataFrame:
    """Calculate the xT added by each action in the events DataFrame."""
    if xt_model is None:
        xt_model = load_xt_model()
    
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
