"""
Pass network analysis module.

This module implements functions to generate and analyze pass networks
for teams, providing insights into team playing styles and structures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def generate_pass_network(events_df: pd.DataFrame, team_id: Optional[int] = None,
                         min_passes: int = 3, include_direction: bool = True) -> Dict[str, Any]:
    """
    Generate a pass network from event data for a specific team.

    Args:
        events_df: DataFrame containing match events
        team_id: Team ID to filter for (if None, uses the team with most passes)
        min_passes: Minimum number of passes between players to include in network
        include_direction: Whether to include pass direction information

    Returns:
        Dictionary with pass network data
    """
    if events_df is None or events_df.empty:
        logger.warning("Empty events DataFrame provided")
        return {"nodes": [], "edges": [], "team_id": team_id}

    # Filter for passes
    pass_events = events_df[events_df['type_name'] == 'Pass'].copy()

    if pass_events.empty:
        logger.warning("No pass events found in the provided data")
        return {"nodes": [], "edges": [], "team_id": team_id}

    # Determine team if not provided
    if team_id is None:
        # Use the team with the most passes
        team_counts = pass_events['team_id'].value_counts()
        if not team_counts.empty:
            team_id = team_counts.index[0]
            logger.info(f"Team ID not provided, using team with most passes: {team_id}")
        else:
            logger.warning("Could not determine team ID from pass events")
            return {"nodes": [], "edges": [], "team_id": None}

    # Filter for the specified team
    team_passes = pass_events[pass_events['team_id'] == team_id]

    if team_passes.empty:
        logger.warning(f"No passes found for team {team_id}")
        return {"nodes": [], "edges": [], "team_id": team_id}

    # Create a directed graph
    G = nx.DiGraph()

    # Track player positions and pass counts
    player_positions = {}
    player_pass_counts = defaultdict(int)
    player_pass_success = defaultdict(int)
    player_x_positions = defaultdict(list)
    player_y_positions = defaultdict(list)

    # Process passes to build the network
    for _, pass_event in team_passes.iterrows():
        passer = pass_event.get('player_name') or pass_event.get('player_id')

        # Skip passes without player info
        if pd.isna(passer):
            continue

        # Get pass recipient (if available)
        recipient = pass_event.get('pass_recipient_name') or pass_event.get('pass_recipient_id')

        # Skip passes without recipient info
        if pd.isna(recipient):
            continue

        # Track player positions based on event locations
        if 'location' in pass_event and pass_event['location'] is not None:
            if isinstance(pass_event['location'], (list, tuple)) and len(pass_event['location']) >= 2:
                x, y = pass_event['location'][0], pass_event['location'][1]
                player_x_positions[passer].append(x)
                player_y_positions[passer].append(y)

                # Also add position info for the recipient based on pass end location
                if 'pass_end_location' in pass_event and pass_event['pass_end_location'] is not None:
                    if isinstance(pass_event['pass_end_location'], (list, tuple)) and len(pass_event['pass_end_location']) >= 2:
                        rx, ry = pass_event['pass_end_location'][0], pass_event['pass_end_location'][1]
                        player_x_positions[recipient].append(rx)
                        player_y_positions[recipient].append(ry)

        # Track player pass counts
        player_pass_counts[passer] += 1

        # Track successful passes
        if 'pass_outcome' not in pass_event or pd.isna(pass_event['pass_outcome']):
            # Assume pass was successful if no outcome is specified
            player_pass_success[passer] += 1

        # Add an edge to the graph (or increment weight if exists)
        if G.has_edge(passer, recipient):
            G[passer][recipient]['weight'] += 1
        else:
            G.add_edge(passer, recipient, weight=1)

    # Calculate average positions for each player
    for player in player_x_positions:
        if player_x_positions[player] and player_y_positions[player]:
            avg_x = sum(player_x_positions[player]) / len(player_x_positions[player])
            avg_y = sum(player_y_positions[player]) / len(player_y_positions[player])
            player_positions[player] = (avg_x, avg_y)

    # Remove edges below the minimum pass threshold
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_passes]
    G.remove_edges_from(edges_to_remove)

    # Prepare node data (players)
    nodes = []
    for player in G.nodes():
        position = player_positions.get(player, (50, 40))  # Default to middle of pitch
        pass_count = player_pass_counts.get(player, 0)
        success_count = player_pass_success.get(player, 0)
        pass_accuracy = (success_count / pass_count * 100) if pass_count > 0 else 0

        nodes.append({
            "id": player,
            "name": player,  # In a real implementation, would map ID to name
            "position": "Unknown",  # In a real implementation, would get from player data
            "x": position[0],
            "y": position[1],
            "passes": pass_count,
            "pass_accuracy": pass_accuracy
        })

    # Prepare edge data (passes between players)
    edges = []
    for u, v, d in G.edges(data=True):
        if include_direction:
            # For directed graph (source → target)
            edges.append({
                "source": u,
                "target": v,
                "weight": d['weight'],
                "normalized_weight": d['weight'] / max(player_pass_counts.values()) if player_pass_counts else 0
            })
        else:
            # For undirected representation, ensure we don't duplicate
            if not any(e for e in edges if
                      (e["source"] == v and e["target"] == u) or
                      (e["source"] == u and e["target"] == v)):
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": d['weight'],
                    "normalized_weight": d['weight'] / max(player_pass_counts.values()) if player_pass_counts else 0
                })

    # Calculate network metrics
    network_metrics = calculate_network_metrics(G, nodes, edges)

    return {
        "team_id": team_id,
        "nodes": nodes,
        "edges": edges,
        "metrics": network_metrics
    }

def calculate_network_metrics(G: nx.DiGraph, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """
    Calculate metrics for the pass network.

    Args:
        G: NetworkX graph object
        nodes: List of node dictionaries
        edges: List of edge dictionaries

    Returns:
        Dictionary of network metrics
    """
    metrics = {}

    # Skip metrics if no nodes or edges
    if not G.nodes() or not G.edges():
        return {
            "betweenness_centrality": {},
            "eigenvector_centrality": {},
            "density": 0,
            "avg_clustering": 0,
            "avg_shortest_path": 0
        }

    # Calculate betweenness centrality (importance of nodes as bridges)
    try:
        metrics["betweenness_centrality"] = nx.betweenness_centrality(G, weight='weight')
    except:
        metrics["betweenness_centrality"] = {}

    # Calculate eigenvector centrality (importance based on connections)
    try:
        metrics["eigenvector_centrality"] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        metrics["eigenvector_centrality"] = {}

    # Calculate network density (how complete the network is)
    metrics["density"] = nx.density(G)

    # Calculate average clustering coefficient (how tightly clustered)
    try:
        metrics["avg_clustering"] = nx.average_clustering(G, weight='weight')
    except:
        metrics["avg_clustering"] = 0

    # Calculate average shortest path length (if the graph is connected)
    try:
        if nx.is_strongly_connected(G):
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(G, weight='weight')
        else:
            # For disconnected graphs, calculate for the largest component
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) > 1:
                metrics["avg_shortest_path"] = nx.average_shortest_path_length(subgraph, weight='weight')
            else:
                metrics["avg_shortest_path"] = 0
    except:
        metrics["avg_shortest_path"] = 0

    return metrics

def analyze_passing_style(pass_network: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the passing style represented by a pass network.

    Args:
        pass_network: Pass network dictionary from generate_pass_network

    Returns:
        Dictionary with style analysis
    """
    # Extract components from pass network
    nodes = pass_network.get("nodes", [])
    edges = pass_network.get("edges", [])
    metrics = pass_network.get("metrics", {})

    if not nodes or not edges:
        return {"style": "Unknown", "confidence": 0, "metrics": {}}

    # Calculate degree centralization
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"])

    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    # Calculate out-degree for each node
    out_degrees = dict(G.out_degree(weight='weight'))

    # Find maximum out-degree
    max_out_degree = max(out_degrees.values()) if out_degrees else 0

    # Calculate total nodes
    n = G.number_of_nodes()

    # Calculate degree centralization (how centered the network is on key players)
    if n > 1 and max_out_degree > 0:
        sum_of_diffs = sum(max_out_degree - deg for deg in out_degrees.values())
        max_possible_sum = (n - 1) * (n - 1)
        degree_centralization = sum_of_diffs / max_possible_sum
    else:
        degree_centralization = 0

    # Calculate average edge weight (passing intensity)
    avg_edge_weight = sum(edge["weight"] for edge in edges) / len(edges) if edges else 0

    # Network density from metrics
    network_density = metrics.get("density", 0)

    # Define style characteristics
    style_metrics = {
        "degree_centralization": degree_centralization,
        "network_density": network_density,
        "avg_edge_weight": avg_edge_weight,
        "betweenness_variation": np.std(list(metrics.get("betweenness_centrality", {}).values())) if metrics.get("betweenness_centrality") else 0
    }

    # Determine team style based on network properties
    # High centralization + low density = Star (focused on key players)
    # Low centralization + high density = Tiki-taka (distributed passing)
    # Medium centralization + low density = Direct (fewer passes, more direct)
    # Low centralization + low density = Chaotic (unstructured play)

    if degree_centralization > 0.5 and network_density < 0.3:
        style = "Star"
        confidence = min(1.0, degree_centralization * 0.8 + (1 - network_density) * 0.2)
    elif degree_centralization < 0.3 and network_density > 0.4:
        style = "Tiki-taka"
        confidence = min(1.0, network_density * 0.7 + (1 - degree_centralization) * 0.3)
    elif 0.3 <= degree_centralization <= 0.5 and network_density < 0.3:
        style = "Direct"
        confidence = min(1.0, (1 - network_density) * 0.6 + avg_edge_weight / 10)
    elif degree_centralization < 0.3 and network_density < 0.3:
        style = "Chaotic"
        confidence = min(1.0, (1 - network_density) * 0.5 + (1 - degree_centralization) * 0.5)
    else:
        style = "Balanced"
        confidence = 0.5

    return {
        "style": style,
        "confidence": confidence,
        "metrics": style_metrics
    }

def compare_pass_networks(network1: Dict[str, Any], network2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two pass networks to identify differences.

    Args:
        network1: First pass network dictionary
        network2: Second pass network dictionary

    Returns:
        Dictionary with comparison results
    """
    # Extract metrics
    metrics1 = network1.get("metrics", {})
    metrics2 = network2.get("metrics", {})

    # Calculate style analyses
    style1 = analyze_passing_style(network1)
    style2 = analyze_passing_style(network2)

    # Calculate metric differences
    metric_diffs = {
        "density_diff": metrics2.get("density", 0) - metrics1.get("density", 0),
        "clustering_diff": metrics2.get("avg_clustering", 0) - metrics1.get("avg_clustering", 0),
        "path_length_diff": metrics2.get("avg_shortest_path", 0) - metrics1.get("avg_shortest_path", 0)
    }

    # Compare betweenness centrality to find key player differences
    betweenness1 = metrics1.get("betweenness_centrality", {})
    betweenness2 = metrics2.get("betweenness_centrality", {})

    player_importance_shift = {}
    all_players = set(betweenness1.keys()) | set(betweenness2.keys())

    for player in all_players:
        importance1 = betweenness1.get(player, 0)
        importance2 = betweenness2.get(player, 0)

        if abs(importance2 - importance1) > 0.05:  # Only track significant changes
            player_importance_shift[player] = importance2 - importance1

    # Sort by absolute value of shift
    key_player_shifts = sorted(
        player_importance_shift.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]  # Top 5 changes

    return {
        "network1_style": style1["style"],
        "network2_style": style2["style"],
        "style_difference": "Same" if style1["style"] == style2["style"] else "Different",
        "metric_differences": metric_diffs,
        "key_player_shifts": key_player_shifts,
        "overall_similarity": calculate_network_similarity(network1, network2)
    }

def calculate_network_similarity(network1: Dict[str, Any], network2: Dict[str, Any]) -> float:
    """
    Calculate a similarity score between two pass networks.

    Args:
        network1: First pass network dictionary
        network2: Second pass network dictionary

    Returns:
        Similarity score between 0 and 1
    """
    # Extract components
    metrics1 = network1.get("metrics", {})
    metrics2 = network2.get("metrics", {})
    edges1 = network1.get("edges", [])
    edges2 = network2.get("edges", [])

    # Calculate metric similarity component (30% weight)
    density_sim = 1 - min(1, abs(metrics1.get("density", 0) - metrics2.get("density", 0)))
    clustering_sim = 1 - min(1, abs(metrics1.get("avg_clustering", 0) - metrics2.get("avg_clustering", 0)))
    path_sim = 1 - min(1, abs(metrics1.get("avg_shortest_path", 0) - metrics2.get("avg_shortest_path", 0)) / 5)

    metric_similarity = (density_sim + clustering_sim + path_sim) / 3

    # Calculate edge pattern similarity component (70% weight)
    # Convert edges to sets of (source, target) tuples for comparison
    edge_tuples1 = {(e["source"], e["target"]) for e in edges1}
    edge_tuples2 = {(e["source"], e["target"]) for e in edges2}

    # Calculate Jaccard similarity of edge sets
    if edge_tuples1 and edge_tuples2:
        intersection = len(edge_tuples1.intersection(edge_tuples2))
        union = len(edge_tuples1.union(edge_tuples2))
        edge_similarity = intersection / union if union > 0 else 0
    else:
        edge_similarity = 0

    # Combine similarities with weights
    similarity = 0.3 * metric_similarity + 0.7 * edge_similarity

    return similarity
