"""
Expected Threat (xT) model implementation.

This module implements the Expected Threat (xT) model, which quantifies the value
of actions on the pitch based on their potential to lead to goals.
Based on Karun Singh's xT model: https://karun.in/blog/expected-threat.html
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExpectedThreatModel:
    """
    Implementation of the Expected Threat (xT) model.

    xT assigns values to pitch locations based on the probability of scoring from actions
    originating at those locations. It values actions (like passes and carries) based on
    the change in scoring probability they create.
    """

    def __init__(self, grid_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the Expected Threat model.

        Args:
            grid_size: Tuple of (x, y) dimensions for the pitch grid
        """
        self.grid_size = grid_size
        self.pitch_length = 120  # Standard pitch length (in yards)
        self.pitch_width = 80   # Standard pitch width (in yards)
        self.cell_length = self.pitch_length / grid_size[0]
        self.cell_width = self.pitch_width / grid_size[1]

        # The xT grid holds the value of each cell
        self.xt_grid = None

    def initialize(self, precomputed_grid: Optional[np.ndarray] = None):
        """
        Initialize the xT grid with values.

        Args:
            precomputed_grid: Optional precomputed xT grid values
        """
        if precomputed_grid is not None:
            if precomputed_grid.shape == self.grid_size:
                self.xt_grid = precomputed_grid
                logger.info("Initialized xT model with provided grid")
            else:
                logger.warning(f"Provided grid has incorrect shape {precomputed_grid.shape}, expected {self.grid_size}")
                self._initialize_default_grid()
        else:
            self._initialize_default_grid()

    def _initialize_default_grid(self):
        """Initialize the xT grid with default values based on a theoretical model"""
        # Create an empty grid
        grid = np.zeros(self.grid_size)

        # Fill with increasing values as we approach the goal
        # This is a simplified model; in practice, this would be trained on real data
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Distance-based value increasing toward the opponent's goal
                # Value increases exponentially as we get closer to goal
                x_factor = (x + 1) / self.grid_size[0]  # Normalized x position (0-1)

                # Y-factor gives higher value to central positions
                y_center_distance = abs(y - (self.grid_size[1] - 1) / 2) / (self.grid_size[1] / 2)
                y_factor = 1 - y_center_distance * 0.5  # Central positions worth more

                # Exponential increase toward goal
                grid[x, y] = 0.001 * (np.exp(3 * x_factor) - 1) * y_factor

        # Normalize grid to have max value of around 0.3 (roughly the xG of a good chance)
        grid = grid / np.max(grid) * 0.3

        self.xt_grid = grid
        logger.info("Initialized default xT model grid")

    def save(self, file_path: str):
        """
        Save the xT model to a file.

        Args:
            file_path: Path to save the model
        """
        if self.xt_grid is None:
            logger.warning("Cannot save uninitialized xT model")
            return

        with open(file_path, 'wb') as f:
            pickle.dump({
                'grid_size': self.grid_size,
                'pitch_length': self.pitch_length,
                'pitch_width': self.pitch_width,
                'xt_grid': self.xt_grid
            }, f)

        logger.info(f"Saved xT model to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'ExpectedThreatModel':
        """
        Load an xT model from a file.

        Args:
            file_path: Path to load the model from

        Returns:
            Loaded ExpectedThreatModel instance
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        model = cls(grid_size=data['grid_size'])
        model.pitch_length = data.get('pitch_length', 120)
        model.pitch_width = data.get('pitch_width', 80)
        model.cell_length = model.pitch_length / model.grid_size[0]
        model.cell_width = model.pitch_width / model.grid_size[1]
        model.xt_grid = data['xt_grid']

        logger.info(f"Loaded xT model from {file_path}")
        return model

    def get_cell_for_position(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert a pitch position to grid cell coordinates.

        Args:
            position: (x, y) position on the pitch in yards

        Returns:
            (x, y) grid cell coordinates
        """
        x = min(max(0, int(position[0] / self.cell_length)), self.grid_size[0] - 1)
        y = min(max(0, int(position[1] / self.cell_width)), self.grid_size[1] - 1)
        return (x, y)

    def get_xt_value(self, position: Tuple[float, float]) -> float:
        """
        Get the xT value for a position on the pitch.

        Args:
            position: (x, y) position on the pitch in yards

        Returns:
            xT value for that position
        """
        if self.xt_grid is None:
            logger.warning("xT model not initialized")
            return 0.0

        cell = self.get_cell_for_position(position)
        return self.xt_grid[cell]

    def calculate_xt_delta(self, event: Dict[str, Any]) -> float:
        """
        Calculate the xT delta (value added) for an event.

        Args:
            event: Event dictionary or DataFrame row

        Returns:
            xT delta value
        """
        if self.xt_grid is None:
            logger.warning("xT model not initialized")
            return 0.0

        # Convert event dictionary or row to standard format
        if isinstance(event, pd.Series):
            event = event.to_dict()

        # Get start and end locations
        start_location = None
        end_location = None

        # Extract start location
        if 'location' in event and event['location'] is not None:
            if isinstance(event['location'], (list, tuple)) and len(event['location']) >= 2:
                start_location = (event['location'][0], event['location'][1])

        # Extract end location based on event type
        if event.get('type') == 'Pass':
            if 'pass_end_location' in event and event['pass_end_location'] is not None:
                if isinstance(event['pass_end_location'], (list, tuple)) and len(event['pass_end_location']) >= 2:
                    end_location = (event['pass_end_location'][0], event['pass_end_location'][1])
        elif event.get('type') == 'Carry':
            if 'carry_end_location' in event and event['carry_end_location'] is not None:
                if isinstance(event['carry_end_location'], (list, tuple)) and len(event['carry_end_location']) >= 2:
                    end_location = (event['carry_end_location'][0], event['carry_end_location'][1])

        # If we couldn't extract valid locations, return 0
        if start_location is None or end_location is None:
            return 0.0

        # Calculate xT values for start and end positions
        start_xt = self.get_xt_value(start_location)
        end_xt = self.get_xt_value(end_location)

        # Calculate delta (the value added by the action)
        xt_delta = end_xt - start_xt

        return max(0, xt_delta)  # Only consider positive contributions

    def calculate_xt_for_match(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate xT values for all applicable events in a match.

        Args:
            events_df: DataFrame of match events

        Returns:
            DataFrame with xT values added
        """
        if self.xt_grid is None:
            logger.warning("xT model not initialized")
            return events_df

        # Make a copy to avoid modifying the original
        df = events_df.copy()

        # Add xT column
        df['xt_value'] = 0.0

        # Add xt_start and xt_end columns
        df['xt_start'] = 0.0
        df['xt_end'] = 0.0

        # Calculate xT for all events with location and end_location
        valid_events = df[df['type'].isin(['Pass', 'Carry'])]

        for idx, event in valid_events.iterrows():
            # Get start location
            start_location = None
            if 'location' in event and event['location'] is not None:
                if isinstance(event['location'], (list, tuple)) and len(event['location']) >= 2:
                    start_location = (event['location'][0], event['location'][1])

            # Get end location
            end_location = None
            if event['type'] == 'Pass' and 'pass_end_location' in event and event['pass_end_location'] is not None:
                if isinstance(event['pass_end_location'], (list, tuple)) and len(event['pass_end_location']) >= 2:
                    end_location = (event['pass_end_location'][0], event['pass_end_location'][1])
            elif event['type'] == 'Carry' and 'carry_end_location' in event and event['carry_end_location'] is not None:
                if isinstance(event['carry_end_location'], (list, tuple)) and len(event['carry_end_location']) >= 2:
                    end_location = (event['carry_end_location'][0], event['carry_end_location'][1])

            # Calculate xT if we have valid locations
            if start_location is not None and end_location is not None:
                start_xt = self.get_xt_value(start_location)
                end_xt = self.get_xt_value(end_location)
                xt_delta = max(0, end_xt - start_xt)

                # Update the DataFrame
                df.at[idx, 'xt_start'] = start_xt
                df.at[idx, 'xt_end'] = end_xt
                df.at[idx, 'xt_value'] = xt_delta

        return df

    def get_xt_grid(self) -> Dict[str, Any]:
        """
        Get the xT grid in a format suitable for visualization.

        Returns:
            Dictionary with grid information
        """
        if self.xt_grid is None:
            logger.warning("xT model not initialized")
            return {
                "grid_size": {"x": 0, "y": 0},
                "pitch_dimensions": {"x": 0, "y": 0},
                "grid_values": []
            }

        return {
            "grid_size": {"x": self.grid_size[0], "y": self.grid_size[1]},
            "pitch_dimensions": {"x": self.pitch_length, "y": self.pitch_width},
            "grid_values": self.xt_grid.tolist()
        }

    def calculate_player_xt_contribution(self, events_df: pd.DataFrame,
                                        player_id: Union[int, str]) -> Dict[str, Any]:
        """
        Calculate total xT contribution for a specific player.

        Args:
            events_df: DataFrame of match events
            player_id: ID of the player to analyze

        Returns:
            Dictionary with player xT contribution metrics
        """
        if self.xt_grid is None:
            logger.warning("xT model not initialized")
            return {"player_id": player_id, "total_xt": 0.0, "actions": []}

        # Calculate xT for all events
        xt_df = self.calculate_xt_for_match(events_df)

        # Filter for the player
        player_events = xt_df[xt_df['player'] == player_id]

        # Sum xT contributions
        total_xt = player_events['xt_value'].sum()

        # Get counts by action type
        pass_xt = player_events[player_events['type'] == 'Pass']['xt_value'].sum()
        carry_xt = player_events[player_events['type'] == 'Carry']['xt_value'].sum()

        # Get top 5 contributions
        top_actions = player_events.sort_values('xt_value', ascending=False).head(5)
        top_actions_list = []

        for _, action in top_actions.iterrows():
            top_actions_list.append({
                "type": action['type'],
                "minute": action['minute'],
                "second": action['second'],
                "xt_value": action['xt_value'],
                "start_location": action['location'],
                "end_location": action['pass_end_location'] if action['type'] == 'Pass' else action['carry_end_location']
            })

        return {
            "player_id": player_id,
            "total_xt": total_xt,
            "pass_xt": pass_xt,
            "carry_xt": carry_xt,
            "action_count": len(player_events),
            "top_actions": top_actions_list
        }
