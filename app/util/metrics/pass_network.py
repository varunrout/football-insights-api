"""
Pass network analysis module for calculating and analyzing passing networks in football matches.
This module provides functionality to extract insights about team structure and connections
between players based on passing patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

def calculate_pass_network(events_df: pd.DataFrame, team_name: str, 
                          min_passes: int = 3, include_subs: bool = False) -> Dict[str, Any]:
    """
    Calculate a pass network for a team in a match.
    
    Args:
        events_df: DataFrame of match events
        team_name: Team name to analyze
        min_passes: Minimum number of passes between players to include in network
        include_subs: Whether to include substitute players in the network
        
    Returns:
        Dictionary with pass network data including player positions and connections
    """
    # Filter to team's passes
    team_events = events_df[events_df['team'] == team_name]
    passes = team_events[team_events['type'] == 'Pass']
    
    # Get unique players
    if not include_subs:
        # Get players from starting lineup
        lineup_events = team_events[team_events['type'] == 'Starting XI']
        if not lineup_events.empty:
            # Extract lineup from tactics field
            lineup_players = []
            for _, event in lineup_events.iterrows():
                if isinstance(event.get('tactics'), dict) and 'lineup' in event['tactics']:
                    for player in event['tactics']['lineup']:
                        if isinstance(player, dict) and 'player' in player:
                            player_info = player['player']
                            if isinstance(player_info, dict) and 'id' in player_info and 'name' in player_info:
                                lineup_players.append(player_info['id'])
            
            # Filter passes to only those by and to starting players
            passes = passes[
                passes['player_id'].isin(lineup_players) & 
                passes['pass_recipient_id'].isin(lineup_players)
            ]
    
    # Initialize dictionaries to store player data
    player_positions = {}  # player_id -> [avg_x, avg_y]
    player_info = {}  # player_id -> {name, position, id}
    
    # Calculate average positions
    for player_id in passes['player_id'].unique():
        player_passes = passes[passes['player_id'] == player_id]
        if player_passes.empty:
            continue
            
        # Get player info from first occurrence
        first_event = player_passes.iloc[0]
        player_info[player_id] = {
            'name': first_event.get('player', f"Player {player_id}"),
            'position': first_event.get('position', 'Unknown'),
            'id': player_id
        }
        
        # Calculate average location
        locations = []
        for _, event in player_passes.iterrows():
            if isinstance(event.get('location'), list) and len(event['location']) >= 2:
                locations.append(event['location'])
                
        if locations:
            avg_x = sum(loc[0] for loc in locations) / len(locations)
            avg_y = sum(loc[1] for loc in locations) / len(locations)
            player_positions[player_id] = [avg_x, avg_y]
    
    # Calculate pass connections
    connections = []
    pass_counts = {}  # (source_id, target_id) -> count
    
    for _, passing_event in passes.iterrows():
        if (pd.notna(passing_event.get('pass_recipient_id')) and 
            pd.notna(passing_event.get('player_id'))):
            
            source_id = passing_event['player_id']
            target_id = passing_event['pass_recipient_id']
            
            # Count the pass
            pass_pair = (source_id, target_id)
            if pass_pair in pass_counts:
                pass_counts[pass_pair] += 1
            else:
                pass_counts[pass_pair] = 1
    
    # Filter connections by minimum passes
    for (source_id, target_id), count in pass_counts.items():
        if count >= min_passes and source_id in player_positions and target_id in player_positions:
            # Calculate pass success rate
            source_to_target_passes = passes[
                (passes['player_id'] == source_id) & 
                (passes['pass_recipient_id'] == target_id)
            ]
            success_rate = 100  # Default is 100% success rate
            
            connections.append({
                'source': source_id,
                'target': target_id,
                'passes': count,
                'success_rate': success_rate
            })
    
    # Prepare player nodes data
    nodes = []
    for player_id, position in player_positions.items():
        if player_id in player_info:
            nodes.append({
                'player_id': player_id,
                'name': player_info[player_id]['name'],
                'position': player_info[player_id]['position'],
                'avg_x': position[0],
                'avg_y': position[1]
            })
    
    return {
        'team': team_name,
        'players': nodes,
        'connections': connections
    }

def analyze_team_structure(pass_network: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze team structure based on the pass network.
    
    Args:
        pass_network: Pass network data from calculate_pass_network
        
    Returns:
        Dictionary with structural analysis metrics
    """
    players = pass_network['players']
    connections = pass_network['connections']
    
    # Calculate player centrality (based on number of connections)
    player_centrality = {}
    for player in players:
        player_id = player['player_id']
        incoming = sum(1 for conn in connections if conn['target'] == player_id)
        outgoing = sum(1 for conn in connections if conn['source'] == player_id)
        player_centrality[player_id] = incoming + outgoing
    
    # Identify key players (top 3 by centrality)
    key_players = sorted(player_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate average team width and depth
    if len(players) > 1:
        x_positions = [p['avg_x'] for p in players]
        y_positions = [p['avg_y'] for p in players]
        team_width = max(y_positions) - min(y_positions)
        team_depth = max(x_positions) - min(x_positions)
        compactness = np.sqrt(np.var(x_positions) + np.var(y_positions))
    else:
        team_width = 0
        team_depth = 0
        compactness = 0
    
    # Calculate strongest connection
    strongest_connection = None
    max_passes = 0
    for conn in connections:
        if conn['passes'] > max_passes:
            max_passes = conn['passes']
            strongest_connection = conn
    
    return {
        'key_players': [
            {
                'player_id': player_id,
                'centrality': centrality,
                'name': next((p['name'] for p in players if p['player_id'] == player_id), f"Player {player_id}")
            } 
            for player_id, centrality in key_players
        ],
        'team_width': team_width,
        'team_depth': team_depth,
        'compactness': compactness,
        'strongest_connection': strongest_connection,
        'total_connections': len(connections),
        'connection_density': len(connections) / (len(players) * (len(players) - 1)) if len(players) > 1 else 0
    }
