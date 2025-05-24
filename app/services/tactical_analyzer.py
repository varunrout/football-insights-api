from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

from app.services.metric_calculator import (
    calculate_ppda,
    identify_progressive_passes,
    calculate_basic_match_metrics
)


def draw_pitch(ax, pitch_color='#22312b', line_color='white'):
    """Draw a football pitch on the given axes."""
    # Pitch dimensions in StatsBomb data: 120x80
    pitch_length = 120
    pitch_width = 80
    
    # Set pitch appearance
    ax.set_xlim([-5, pitch_length + 5])
    ax.set_ylim([-5, pitch_width + 5])
    ax.set_facecolor(pitch_color)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Main pitch outline
    ax.plot([0, 0], [0, pitch_width], line_color, zorder=1)
    ax.plot([0, pitch_length], [pitch_width, pitch_width], line_color, zorder=1)
    ax.plot([pitch_length, pitch_length], [pitch_width, 0], line_color, zorder=1)
    ax.plot([pitch_length, 0], [0, 0], line_color, zorder=1)
    
    # Middle line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], line_color, zorder=1)
    
    # Middle circle
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15, fill=False, color=line_color, zorder=1)
    ax.add_patch(center_circle)
    
    # Left penalty area
    ax.plot([18, 18], [pitch_width/2 - 22, pitch_width/2 + 22], line_color, zorder=1)
    ax.plot([0, 18], [pitch_width/2 - 22, pitch_width/2 - 22], line_color, zorder=1)
    ax.plot([18, 0], [pitch_width/2 + 22, pitch_width/2 + 22], line_color, zorder=1)
    
    # Right penalty area
    ax.plot([pitch_length - 18, pitch_length - 18], [pitch_width/2 - 22, pitch_width/2 + 22], line_color, zorder=1)
    ax.plot([pitch_length, pitch_length - 18], [pitch_width/2 - 22, pitch_width/2 - 22], line_color, zorder=1)
    ax.plot([pitch_length - 18, pitch_length], [pitch_width/2 + 22, pitch_width/2 + 22], line_color, zorder=1)
    
    return ax


class PassingNetworkAnalyzer:
    """Analyze team passing networks and structures."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
    
    def create_passing_network(self, team_name, min_passes=2, include_subs=False):
        """Create a passing network for a specific team.
        
        Args:
            team_name: Name of the team to analyze
            min_passes: Minimum number of passes between players to show in the network
            include_subs: Whether to include substitutes in the analysis
            
        Returns:
            G: NetworkX graph object representing the passing network
            avg_positions: Dictionary of player average positions
            pass_count: Dictionary of pass counts between players
        """
        # Filter for the team's passes
        team_passes = self.events_df[(self.events_df['team'] == team_name) & 
                                    (self.events_df['type'] == 'Pass')]
        
        # Get unique players who played for the team
        team_players = self.events_df[self.events_df['team'] == team_name]['player'].unique()
        
        if not include_subs:
            # Try to identify starting players - this is simplified and may need refinement
            # Count events in first 15 minutes to identify likely starters
            early_events = self.events_df[(self.events_df['team'] == team_name) & 
                                         (self.events_df['minute'] < 15)]
            starter_counts = early_events['player'].value_counts()
            starters = starter_counts[starter_counts >= 3].index.tolist()
            
            # Use only passes between starters
            team_passes = team_passes[
                (team_passes['player'].isin(starters)) &
                (team_passes['pass_recipient'].isin(starters))
            ]
        
        # Calculate average positions
        avg_positions = {}
        for player in team_players:
            player_events = self.events_df[(self.events_df['team'] == team_name) & 
                                          (self.events_df['player'] == player)]
            if len(player_events) > 0:
                # Extract locations and calculate average
                locations = []
                for _, event in player_events.iterrows():
                    if isinstance(event['location'], list) and len(event['location']) >= 2:
                        locations.append(event['location'])
                
                if locations:
                    avg_x = sum(loc[0] for loc in locations) / len(locations)
                    avg_y = sum(loc[1] for loc in locations) / len(locations)
                    avg_positions[player] = (avg_x, avg_y)
        
        # Count passes between players
        pass_count = defaultdict(int)
        for _, pass_event in team_passes.iterrows():
            if pd.notna(pass_event['player']) and pd.notna(pass_event['pass_recipient']):
                passer = pass_event['player']
                receiver = pass_event['pass_recipient']
                pass_count[(passer, receiver)] += 1
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes (players)
        for player in avg_positions.keys():
            G.add_node(player, pos=avg_positions[player])
        
        # Add edges (passes)
        for (passer, receiver), count in pass_count.items():
            if count >= min_passes and passer in avg_positions and receiver in avg_positions:
                G.add_edge(passer, receiver, weight=count)
        
        return G, avg_positions, pass_count
    
    def get_network_metrics(self, G):
        """Calculate network analysis metrics for a team's passing network.
        
        Args:
            G: NetworkX graph representing the passing network
            
        Returns:
            Dictionary of network metrics
        """
        metrics = {}
        
        # Basic network properties
        metrics['density'] = nx.density(G)
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        # Centrality metrics
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000) if G.number_of_edges() > 0 else {}
        
        # Key players
        if metrics['betweenness_centrality']:
            metrics['key_connector'] = max(metrics['betweenness_centrality'].items(), key=lambda x: x[1])[0]
            
        if metrics['eigenvector_centrality']:
            metrics['key_hub'] = max(metrics['eigenvector_centrality'].items(), key=lambda x: x[1])[0]
            
        # Community detection (tactical units)
        if G.number_of_edges() > 0:
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())
                metrics['communities'] = [list(c) for c in communities]
            except:
                metrics['communities'] = []
        else:
            metrics['communities'] = []
            
        return metrics


class PressAnalyzer:
    """Analyze team pressing patterns and defensive organization."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
    
    def analyze_pressing(self, team, opposition_half_only=True):
        """Analyze pressing patterns for a specific team.
        
        Args:
            team: Name of the team to analyze
            opposition_half_only: If True, only consider defensive actions in the opposition half
            
        Returns:
            Analysis results dictionary
        """
        # Filter for pressure events by the team
        pressure_events = self.events_df[(self.events_df['team'] == team) & 
                                       (self.events_df['type'] == 'Pressure')]
        
        if pressure_events.empty:
            return {"error": "No pressure events found for this team."}
        
        # Calculate PPDA
        ppda = calculate_ppda(self.events_df, team)
        
        # Extract pressure locations
        pressure_locations = []
        for _, event in pressure_events.iterrows():
            if isinstance(event['location'], list) and len(event['location']) >= 2:
                pressure_locations.append((event['location'][0], event['location'][1]))
        
        # Calculate pressing intensity by zone
        zones = {
            'Defensive Third': (0, 40),
            'Middle Third': (40, 80),
            'Attacking Third': (80, 120)
        }
        
        zone_counts = {zone: 0 for zone in zones}
        for x, _ in pressure_locations:
            for zone, (min_x, max_x) in zones.items():
                if min_x <= x < max_x:
                    zone_counts[zone] += 1
                    break
        
        total_pressures = sum(zone_counts.values())
        zone_percentages = {zone: (count / total_pressures * 100) if total_pressures > 0 else 0 
                            for zone, count in zone_counts.items()}
        
        # Calculate pressing duration and success
        pressing_sequences = self._identify_pressing_sequences(team)
        
        # Return analysis results
        results = {
            'ppda': ppda,
            'total_pressures': len(pressure_events),
            'pressure_locations': pressure_locations,
            'zone_counts': zone_counts,
            'zone_percentages': zone_percentages,
            'pressing_sequences': pressing_sequences
        }
        
        return results
    
    def _identify_pressing_sequences(self, team):
        """Identify sequences of coordinated pressing."""
        sequences = []
        
        # Group events by minute to identify potential pressing sequences
        for minute in sorted(self.events_df['minute'].unique()):
            minute_events = self.events_df[(self.events_df['minute'] == minute) & 
                                          (self.events_df['team'] == team)]
            
            # If we have multiple pressure events in the same minute, could be a pressing sequence
            pressures = minute_events[minute_events['type'] == 'Pressure']
            if len(pressures) >= 3:  # At least 3 pressure actions in a minute indicates coordinated press
                # Check if the pressing led to a turnover
                next_events = self.events_df[(self.events_df['minute'] > minute) & 
                                            (self.events_df['minute'] <= minute + 1)]
                
                ball_recovery = next_events[(next_events['team'] == team) & 
                                           (next_events['type'].isin(['Ball Recovery', 'Interception']))]
                
                sequences.append({
                    'minute': minute,
                    'pressure_count': len(pressures),
                    'led_to_turnover': len(ball_recovery) > 0
                })
                
        return sequences


class BuildupAnalyzer:
    """Analyze team buildup patterns and progression from defense to attack."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
    
    def analyze_buildup(self, team):
        """Analyze buildup patterns for a specific team.
        
        Args:
            team: Name of the team to analyze
            
        Returns:
            Analysis results dictionary
        """
        # Filter for team's events
        team_events = self.events_df[self.events_df['team'] == team]
        
        # Find all possessions for the team
        team_possessions = self.events_df[self.events_df['possession_team'] == team]['possession'].unique()
        
        # Analyze progressive passes
        progressive_passes = identify_progressive_passes(team_events)
        
        # Count progressive passes by position
        prog_by_position = defaultdict(int)
        if not progressive_passes.empty:
            position_counts = progressive_passes.groupby('position').size()
            prog_by_position = position_counts.to_dict()
        
        # Find buildup possessions that start from the back
        buildup_possessions = []
        
        for possession in team_possessions:
            possession_events = self.events_df[self.events_df['possession'] == possession]
            if len(possession_events) > 3:  # Ignore very short possessions
                # Check if possession starts in defensive third
                first_event = possession_events.iloc[0]
                if isinstance(first_event['location'], list) and len(first_event['location']) >= 2:
                    if first_event['location'][0] < 40:  # Defensive third
                        # Check if possession reaches attacking third
                        reached_attack = False
                        for _, event in possession_events.iterrows():
                            if isinstance(event['location'], list) and len(event['location']) >= 2:
                                if event['location'][0] >= 80:  # Attacking third
                                    reached_attack = True
                                    break
                        
                        if reached_attack:
                            buildup_possessions.append(possession)
        
        # Calculate buildup success rate
        buildup_success_rate = len(buildup_possessions) / len(team_possessions) * 100 if team_possessions.size > 0 else 0
        
        # Extract the first three players involved in successful buildup possessions
        key_buildup_players = defaultdict(int)
        for possession in buildup_possessions:
            possession_events = self.events_df[(self.events_df['possession'] == possession) & 
                                             (self.events_df['team'] == team)].sort_values('index')
            buildup_sequence = []
            for _, event in possession_events.head(5).iterrows():  # Look at first 5 events
                if pd.notna(event['player']):
                    buildup_sequence.append(event['player'])
                if len(buildup_sequence) >= 3:  # First 3 players
                    break
            
            # Count player appearances in buildup sequences
            for player in buildup_sequence:
                key_buildup_players[player] += 1
        
        # Return analysis results
        results = {
            'total_possessions': len(team_possessions),
            'successful_buildups': len(buildup_possessions),
            'buildup_success_rate': buildup_success_rate,
            'progressive_passes': len(progressive_passes),
            'prog_by_position': dict(prog_by_position),
            'key_buildup_players': dict(key_buildup_players)
        }
        
        return results


class TransitionAnalyzer:
    """Analyze team transition patterns between defense and attack."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
    
    def analyze_transitions(self, team):
        """Analyze transition patterns for a specific team.
        
        Args:
            team: Name of the team to analyze
            
        Returns:
            Analysis results dictionary
        """
        # Filter for team's events
        team_events = self.events_df[self.events_df['team'] == team]
        
        # Identify counter-attacks
        counter_attacks = []
        possessions = self.events_df['possession'].unique()
        
        for possession in possessions:
            possession_events = self.events_df[self.events_df['possession'] == possession].sort_values('index')
            if possession_events.empty:
                continue
                
            # Check if possession belongs to the team
            if possession_events.iloc[0]['possession_team'] == team:
                # Find possessions that start with a recovery/interception/tackle
                first_event_type = possession_events.iloc[0]['type']
                if first_event_type in ['Ball Recovery', 'Interception', 'Duel']:
                    # Check if a shot occurred within 15 seconds
                    start_second = possession_events.iloc[0]['second']
                    shot_events = possession_events[possession_events['type'] == 'Shot']
                    
                    if not shot_events.empty:
                        shot_seconds = shot_events['second'].values
                        if any(s - start_second <= 15 for s in shot_seconds):
                            counter_attacks.append(possession)
        
        # Calculate counter-attack effectiveness
        counter_goals = 0
        counter_shots = 0
        counter_xg = 0
        
        for possession in counter_attacks:
            possession_events = self.events_df[(self.events_df['possession'] == possession) & 
                                             (self.events_df['type'] == 'Shot')]
            counter_shots += len(possession_events)
            counter_goals += sum(possession_events['shot_outcome'] == 'Goal')
            counter_xg += possession_events['shot_statsbomb_xg'].sum()
        
        # Analyze ball recoveries
        recoveries = team_events[team_events['type'] == 'Ball Recovery']
        recovery_locations = []
        
        for _, event in recoveries.iterrows():
            if isinstance(event['location'], list) and len(event['location']) >= 2:
                recovery_locations.append((event['location'][0], event['location'][1]))
        
        # Calculate recovery zones
        zones = {
            'Defensive Third': (0, 40),
            'Middle Third': (40, 80),
            'Attacking Third': (80, 120)
        }
        
        zone_counts = {zone: 0 for zone in zones}
        for x, _ in recovery_locations:
            for zone, (min_x, max_x) in zones.items():
                if min_x <= x < max_x:
                    zone_counts[zone] += 1
                    break
        
        total_recoveries = sum(zone_counts.values())
        zone_percentages = {zone: (count / total_recoveries * 100) if total_recoveries > 0 else 0 
                            for zone, count in zone_counts.items()}
        
        # Return analysis results
        results = {
            'counter_attacks': len(counter_attacks),
            'counter_shots': counter_shots,
            'counter_goals': counter_goals,
            'counter_conversion': (counter_goals / max(counter_shots, 1)) * 100,
            'counter_xg': counter_xg,
            'total_recoveries': total_recoveries,
            'recovery_locations': recovery_locations,
            'zone_percentages': zone_percentages
        }
        
        return results


class SetPieceAnalyzer:
    """Analyze team set piece patterns and effectiveness."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
    
    def analyze_set_pieces(self, team):
        """Analyze set piece patterns for a specific team.
        
        Args:
            team: Name of the team to analyze
            
        Returns:
            Analysis results dictionary
        """
        # Filter for team's events
        team_events = self.events_df[self.events_df['team'] == team]
        
        # Identify corners
        corners = team_events[
            (team_events['pass_type'] == 'Corner') |
            (team_events['play_pattern'] == 'From Corner')
        ]
        
        # Identify free kicks
        free_kicks = team_events[
            ((team_events['pass_type'] == 'Free Kick') |
            (team_events['play_pattern'] == 'From Free Kick')) &
            (team_events['type'] == 'Pass')
        ]
        
        # Calculate set piece effectiveness
        # For corners
        corner_shots = 0
        corner_goals = 0
        corner_xg = 0
        
        for _, corner in corners.iterrows():
            # Find shots within 10 seconds of corner
            if pd.notna(corner['pass_assisted_shot_id']):
                corner_shots += 1
                # Check if it's a goal
                shot = self.events_df[self.events_df.index == corner['pass_assisted_shot_id']]
                if not shot.empty:
                    if shot.iloc[0]['shot_outcome'] == 'Goal':
                        corner_goals += 1
                    corner_xg += shot.iloc[0]['shot_statsbomb_xg'] if pd.notna(shot.iloc[0]['shot_statsbomb_xg']) else 0
        
        # For free kicks
        freekick_shots = 0
        freekick_goals = 0
        freekick_xg = 0
        
        for _, freekick in free_kicks.iterrows():
            # Find shots within 10 seconds of free kick
            if pd.notna(freekick['pass_assisted_shot_id']):
                freekick_shots += 1
                # Check if it's a goal
                shot = self.events_df[self.events_df.index == freekick['pass_assisted_shot_id']]
                if not shot.empty:
                    if shot.iloc[0]['shot_outcome'] == 'Goal':
                        freekick_goals += 1
                    freekick_xg += shot.iloc[0]['shot_statsbomb_xg'] if pd.notna(shot.iloc[0]['shot_statsbomb_xg']) else 0
        
        # Identify corner takers
        corner_takers = corners['player'].value_counts().to_dict() if not corners.empty else {}
        
        # Identify free kick takers
        freekick_takers = free_kicks['player'].value_counts().to_dict() if not free_kicks.empty else {}
        
        # Extract end locations of corners
        corner_end_locations = []
        for _, corner in corners.iterrows():
            if isinstance(corner['pass_end_location'], list) and len(corner['pass_end_location']) >= 2:
                corner_end_locations.append((corner['pass_end_location'][0], corner['pass_end_location'][1]))
        
        # Return analysis results
        results = {
            'total_corners': len(corners),
            'corner_shots': corner_shots,
            'corner_goals': corner_goals,
            'corner_conversion': (corner_goals / max(len(corners), 1)) * 100,
            'corner_xg': corner_xg,
            'corner_end_locations': corner_end_locations,
            'total_freekicks': len(free_kicks),
            'freekick_shots': freekick_shots,
            'freekick_goals': freekick_goals,
            'freekick_conversion': (freekick_goals / max(len(free_kicks), 1)) * 100,
            'freekick_xg': freekick_xg,
            'corner_takers': corner_takers,
            'freekick_takers': freekick_takers
        }
        
        return results


class TacticalAnalyzer:
    """Main class for tactical analysis combining all specialized analyzers."""
    
    def __init__(self, events_df):
        """Initialize with match events dataframe."""
        self.events_df = events_df
        self.teams = events_df['team'].unique()
        
        # Initialize specialized analyzers
        self.passing_network = PassingNetworkAnalyzer(events_df)
        self.press_analyzer = PressAnalyzer(events_df)
        self.buildup_analyzer = BuildupAnalyzer(events_df)
        self.transition_analyzer = TransitionAnalyzer(events_df)
        self.set_piece_analyzer = SetPieceAnalyzer(events_df)
    
    def get_team_tactical_profile(self, team):
        """Generate a comprehensive tactical profile for a team.
        
        Args:
            team: Name of the team to analyze
            
        Returns:
            Dictionary with tactical profile data
        """
        # Basic match metrics
        basic_metrics = calculate_basic_match_metrics(self.events_df)
        team_metrics = basic_metrics.get(team, {})
        
        # Get tactical analyses
        passing_network, avg_positions, pass_count = self.passing_network.create_passing_network(team)
        network_metrics = self.passing_network.get_network_metrics(passing_network)
        
        pressing_analysis = self.press_analyzer.analyze_pressing(team)
        buildup_analysis = self.buildup_analyzer.analyze_buildup(team)
        transition_analysis = self.transition_analyzer.analyze_transitions(team)
        set_piece_analysis = self.set_piece_analyzer.analyze_set_pieces(team)
        
        # Combine into a tactical profile
        tactical_profile = {
            'team': team,
            'basic_metrics': team_metrics,
            'passing_network': {
                'graph': passing_network,
                'avg_positions': avg_positions,
                'pass_count': pass_count,
                'metrics': network_metrics
            },
            'pressing': pressing_analysis,
            'buildup': buildup_analysis,
            'transitions': transition_analysis,
            'set_pieces': set_piece_analysis
        }
        
        return tactical_profile
    
    def compare_teams(self, team1, team2):
        """Compare tactical profiles of two teams.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            
        Returns:
            Dictionary with comparative analysis
        """
        profile1 = self.get_team_tactical_profile(team1)
        profile2 = self.get_team_tactical_profile(team2)
        
        comparison = {
            'teams': [team1, team2],
            'possession': [profile1['basic_metrics'].get('possession_pct', 0), 
                          profile2['basic_metrics'].get('possession_pct', 0)],
            'pass_completion': [profile1['basic_metrics'].get('pass_completion', 0), 
                               profile2['basic_metrics'].get('pass_completion', 0)],
            'ppda': [profile1['basic_metrics'].get('ppda', 0), 
                    profile2['basic_metrics'].get('ppda', 0)],
            'buildup_success': [profile1['buildup'].get('buildup_success_rate', 0),
                               profile2['buildup'].get('buildup_success_rate', 0)],
            'counter_attack_goals': [profile1['transitions'].get('counter_goals', 0),
                                    profile2['transitions'].get('counter_goals', 0)],
            'network_density': [profile1['passing_network']['metrics'].get('density', 0),
                              profile2['passing_network']['metrics'].get('density', 0)]
        }
        
        return comparison
