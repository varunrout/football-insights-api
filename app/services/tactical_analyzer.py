"""
Tactical analyzer for football match data.

This service provides tactical analysis of football matches, including
defensive metrics, offensive patterns, and playing style classification.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
from pathlib import Path

from app.util.football_data_manager import FootballDataManager
from app.services.metrics_engine import MetricsEngine
from app.util.metrics.pass_network import generate_pass_network, analyze_passing_style, compare_pass_networks
from app.util.metrics.ppda import calculate_ppda, calculate_ppda_timeline
from app.config.environment import settings

logger = logging.getLogger(__name__)

class TacticalAnalyzer:
    """
    Analyzes tactical aspects of football matches.

    This service uses event data to calculate various tactical metrics and
    provide insights into team playing styles and tactical approaches.
    """

    def __init__(self, metrics_engine: Optional[MetricsEngine] = None):
        """
        Initialize the TacticalAnalyzer.

        Args:
            metrics_engine: MetricsEngine instance for metrics calculation
        """
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.cache_dir = Path(settings.DATA_CACHE_DIR) / "tactical_analysis"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._cached_analysis = {}

    def get_defensive_metrics(self, team_id: int,
                            match_id: Optional[int] = None,
                            competition_id: Optional[int] = None,
                            season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive defensive metrics for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary of defensive metrics
        """
        cache_key = f"defensive_metrics_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    metrics = pickle.load(f)
                    self._cached_analysis[cache_key] = metrics
                    return metrics
            except Exception as e:
                logger.warning(f"Failed to load cached defensive metrics: {e}")

        # If analyzing a single match
        if match_id:
            # Get match metrics
            match_metrics = self.metrics_engine.calculate_match_metrics(match_id)

            # Extract team-specific defensive metrics
            if match_metrics:
                # Determine if team is home or away
                is_home = match_metrics.get("home_team_id") == team_id
                team_key = "home" if is_home else "away"
                opp_key = "away" if is_home else "home"

                # Extract defensive metrics
                ppda = match_metrics.get("ppda", {}).get(team_key, {}).get("ppda", 0)
                defensive_actions = match_metrics.get("defensive", {}).get(f"{team_key}_defensive_actions", 0)
                tackles = match_metrics.get("defensive", {}).get(f"{team_key}_tackles", 0)
                interceptions = match_metrics.get("defensive", {}).get(f"{team_key}_interceptions", 0)
                blocks = match_metrics.get("defensive", {}).get(f"{team_key}_blocks", 0)
                clearances = match_metrics.get("defensive", {}).get(f"{team_key}_clearances", 0)

                # Extract pressure metrics if available
                pressure_events = match_metrics.get("pressure", {}).get(team_key, {})

                metrics = {
                    "team_id": team_id,
                    "match_id": match_id,
                    "ppda": ppda,
                    "defensive_actions": {
                        "total": defensive_actions,
                        "tackles": tackles,
                        "interceptions": interceptions,
                        "blocks": blocks,
                        "clearances": clearances
                    },
                    "pressure": pressure_events,
                    "opposition_passes": match_metrics.get("ppda", {}).get(team_key, {}).get("opposition_passes", 0),
                    "challenges": match_metrics.get("ppda", {}).get(team_key, {}).get("challenges", 0),
                    "opponent_xg": match_metrics.get("shots", {}).get(f"{opp_key}_xg", 0),
                }

                # Get PPDA timeline
                if "events" in match_metrics:
                    metrics["ppda_timeline"] = calculate_ppda_timeline(match_metrics["events"]).get(team_key, [])

                # Cache the results
                self._cached_analysis[cache_key] = metrics

                # Save to disk
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(metrics, f)
                except Exception as e:
                    logger.warning(f"Failed to cache defensive metrics: {e}")

                return metrics

        # For multi-match analysis
        # Get team metrics across matches
        team_metrics = self.metrics_engine.calculate_team_metrics(
            team_id, competition_id, season_id)

        if team_metrics and "matches_found" in team_metrics and team_metrics["matches_found"] > 0:
            # Extract and aggregate defensive metrics across matches
            aggregated = team_metrics.get("metrics", {})

            metrics = {
                "team_id": team_id,
                "competition_id": competition_id,
                "season_id": season_id,
                "matches_analyzed": team_metrics["matches_found"],
                "avg_ppda": aggregated.get("avg_ppda", 0),
                "defensive_actions_per_match": {
                    "tackles": aggregated.get("tackles_per_match", 0),
                    "interceptions": aggregated.get("interceptions_per_match", 0),
                    "blocks": aggregated.get("blocks_per_match", 0),
                    "clearances": aggregated.get("clearances_per_match", 0),
                },
                "total_defensive_actions": aggregated.get("total_defensive_actions", 0),
                "avg_opposition_passes": aggregated.get("avg_opposition_passes", 0),
                "avg_challenges": aggregated.get("avg_challenges", 0),
                "avg_opponent_xg": aggregated.get("avg_xg_against", 0),
            }

            # Add match-by-match defensive metrics
            metrics["match_metrics"] = [
                {
                    "match_id": m.get("match_id"),
                    "ppda": m.get("ppda", 0),
                    "defensive_actions": m.get("defensive_actions", 0),
                    "opponent_xg": m.get("xg_against", 0)
                }
                for m in team_metrics.get("match_metrics", [])
            ]

            # Cache the results
            self._cached_analysis[cache_key] = metrics

            # Save to disk
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(metrics, f)
            except Exception as e:
                logger.warning(f"Failed to cache defensive metrics: {e}")

            return metrics

        # Fallback to empty response
        return {
            "team_id": team_id,
            "error": "No defensive metrics found for the specified filters",
            "matches_analyzed": 0
        }

    def get_offensive_metrics(self, team_id: int,
                           match_id: Optional[int] = None,
                           competition_id: Optional[int] = None,
                           season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive offensive metrics for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary of offensive metrics
        """
        cache_key = f"offensive_metrics_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    metrics = pickle.load(f)
                    self._cached_analysis[cache_key] = metrics
                    return metrics
            except Exception as e:
                logger.warning(f"Failed to load cached offensive metrics: {e}")

        # If analyzing a single match
        if match_id:
            # Get match metrics
            match_metrics = self.metrics_engine.calculate_match_metrics(match_id)

            # Extract team-specific offensive metrics
            if match_metrics:
                # Determine if team is home or away
                is_home = match_metrics.get("home_team_id") == team_id
                team_key = "home" if is_home else "away"

                # Extract offensive metrics
                possession = match_metrics.get("possession", {}).get(f"{team_key}_possession", 0)
                passes = match_metrics.get("passes", {}).get(f"{team_key}_passes", 0)
                pass_completion = match_metrics.get("passes", {}).get(f"{team_key}_pass_completion", 0)
                progressive_passes = match_metrics.get("passes", {}).get(f"{team_key}_progressive_passes", 0)
                shots = match_metrics.get("shots", {}).get(f"{team_key}_shots", 0)
                shots_on_target = match_metrics.get("shots", {}).get(f"{team_key}_shots_on_target", 0)
                xg = match_metrics.get("shots", {}).get(f"{team_key}_xg", 0)

                metrics = {
                    "team_id": team_id,
                    "match_id": match_id,
                    "possession": possession,
                    "passing": {
                        "total": passes,
                        "completion_rate": pass_completion,
                        "progressive_passes": progressive_passes
                    },
                    "shooting": {
                        "shots": shots,
                        "shots_on_target": shots_on_target,
                        "xg": xg
                    },
                    "xt": match_metrics.get("xt", {}).get(f"{team_key}_xt", 0)
                }

                # Add possession chain metrics if available
                if "possession_chains" in match_metrics:
                    team_chains = match_metrics["possession_chains"].get(team_key, [])
                    if team_chains:
                        metrics["possession_chains"] = {
                            "total": len(team_chains),
                            "avg_length": sum(chain.get("length", 0) for chain in team_chains) / len(team_chains),
                            "avg_duration": sum(chain.get("duration", 0) for chain in team_chains) / len(team_chains),
                            "top_5_chains": sorted(team_chains, key=lambda c: c.get("xg", 0), reverse=True)[:5]
                        }

                # Cache the results
                self._cached_analysis[cache_key] = metrics

                # Save to disk
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(metrics, f)
                except Exception as e:
                    logger.warning(f"Failed to cache offensive metrics: {e}")

                return metrics

        # For multi-match analysis
        # Get team metrics across matches
        team_metrics = self.metrics_engine.calculate_team_metrics(
            team_id, competition_id, season_id)

        if team_metrics and "matches_found" in team_metrics and team_metrics["matches_found"] > 0:
            # Extract and aggregate offensive metrics across matches
            aggregated = team_metrics.get("metrics", {})

            metrics = {
                "team_id": team_id,
                "competition_id": competition_id,
                "season_id": season_id,
                "matches_analyzed": team_metrics["matches_found"],
                "avg_possession": aggregated.get("avg_possession", 0),
                "passing": {
                    "avg_passes_per_match": aggregated.get("avg_passes_per_match", 0),
                    "avg_pass_completion": aggregated.get("avg_pass_completion", 0),
                    "avg_progressive_passes": aggregated.get("avg_progressive_passes", 0)
                },
                "shooting": {
                    "total_shots": aggregated.get("shots", 0),
                    "shots_per_match": aggregated.get("shots", 0) / team_metrics["matches_found"] if team_metrics["matches_found"] > 0 else 0,
                    "shots_on_target": aggregated.get("shots_on_target", 0),
                    "shot_accuracy": aggregated.get("shot_accuracy", 0),
                    "total_xg": aggregated.get("xg_for", 0),
                    "xg_per_match": aggregated.get("xg_for", 0) / team_metrics["matches_found"] if team_metrics["matches_found"] > 0 else 0,
                },
                "goals": {
                    "total": aggregated.get("goals_for", 0),
                    "per_match": aggregated.get("goals_for", 0) / team_metrics["matches_found"] if team_metrics["matches_found"] > 0 else 0,
                    "xg_difference": aggregated.get("goals_for", 0) - aggregated.get("xg_for", 0)
                }
            }

            # Add match-by-match offensive metrics
            metrics["match_metrics"] = [
                {
                    "match_id": m.get("match_id"),
                    "possession": m.get("possession", 0),
                    "passes": m.get("passes", 0),
                    "shots": m.get("shots", 0),
                    "xg": m.get("xg_for", 0),
                    "goals": m.get("goals_for", 0)
                }
                for m in team_metrics.get("match_metrics", [])
            ]

            # Cache the results
            self._cached_analysis[cache_key] = metrics

            # Save to disk
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(metrics, f)
            except Exception as e:
                logger.warning(f"Failed to cache offensive metrics: {e}")

            return metrics

        # Fallback to empty response
        return {
            "team_id": team_id,
            "error": "No offensive metrics found for the specified filters",
            "matches_analyzed": 0
        }

    def get_pass_network(self, team_id: int, match_id: int,
                       min_passes: int = 3) -> Dict[str, Any]:
        """
        Get pass network for a team in a specific match.

        Args:
            team_id: Team ID
            match_id: Match ID
            min_passes: Minimum number of passes between players to include

        Returns:
            Dictionary with pass network data
        """
        cache_key = f"pass_network_{team_id}_match_{match_id}_min_{min_passes}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    network = pickle.load(f)
                    self._cached_analysis[cache_key] = network
                    return network
            except Exception as e:
                logger.warning(f"Failed to load cached pass network: {e}")

        # Get events for the match
        data_manager = self.metrics_engine.data_manager
        events = data_manager.get_events(match_id)

        if events is not None and not events.empty:
            # Generate pass network
            network = generate_pass_network(events, team_id, min_passes)

            # Add style analysis
            style_analysis = analyze_passing_style(network)
            network["style_analysis"] = style_analysis

            # Add match info
            match_info = self._get_match_info(match_id)
            network["match_info"] = match_info

            # Cache the results
            self._cached_analysis[cache_key] = network

            # Save to disk
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(network, f)
            except Exception as e:
                logger.warning(f"Failed to cache pass network: {e}")

            return network

        # Fallback to empty response
        return {
            "team_id": team_id,
            "match_id": match_id,
            "error": "No pass events found for the specified match",
            "nodes": [],
            "edges": []
        }

    def get_build_up_analysis(self, team_id: int,
                           match_id: Optional[int] = None,
                           competition_id: Optional[int] = None,
                           season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get build-up play analysis for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with build-up analysis
        """
        cache_key = f"build_up_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached build-up analysis: {e}")

        # Placeholder implementation - would analyze possession chains
        # and passes from defensive third to attacking third

        # Start with offensive metrics as a base
        metrics = self.get_offensive_metrics(team_id, match_id, competition_id, season_id)

        # Add build-up specific metrics
        build_up_metrics = {
            "team_id": team_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "possessions_from_goalkeeper": 25,  # Placeholder values
            "short_build_up_pct": 65,
            "long_build_up_pct": 35,
            "avg_build_up_duration": 12.5,  # seconds
            "progressive_passes_from_defensive_third": 15,
            "defensive_third_pass_completion": 92.5,
            "middle_third_pass_completion": 85.3,
            "build_up_xg_created": 1.2,
            "key_build_up_players": [
                {"player_id": 1001, "name": "Player A", "involvement_pct": 25.3},
                {"player_id": 1002, "name": "Player B", "involvement_pct": 22.1},
                {"player_id": 1003, "name": "Player C", "involvement_pct": 18.7}
            ]
        }

        # Combine with base offensive metrics
        analysis = {**metrics, **build_up_metrics}

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache build-up analysis: {e}")

        return analysis

    def get_pressing_analysis(self, team_id: int,
                           match_id: Optional[int] = None,
                           competition_id: Optional[int] = None,
                           season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get pressing analysis for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with pressing analysis
        """
        cache_key = f"pressing_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached pressing analysis: {e}")

        # Start with defensive metrics as a base
        metrics = self.get_defensive_metrics(team_id, match_id, competition_id, season_id)

        # Add pressing specific metrics
        pressing_metrics = {
            "team_id": team_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "pressing_intensity": {
                "score": 8.5,  # Out of 10
                "percentile": 85  # League percentile
            },
            "pressing_zones": {
                "attacking_third": 35.0,  # Percentage of pressing in each third
                "middle_third": 45.0,
                "defensive_third": 20.0
            },
            "pressing_success_rate": 62.5,  # Percentage of successful pressures
            "ball_recoveries_after_pressure": 18,
            "avg_time_to_pressure_opposition": 3.2,  # seconds
            "pressing_triggers": {
                "opponent_receive_with_back_to_goal": 25.0,  # Percentage of total pressures
                "long_pass": 15.0,
                "pass_to_fullback": 30.0,
                "pass_to_center_back": 20.0,
                "other": 10.0
            },
            "key_pressing_players": [
                {"player_id": 1001, "name": "Player A", "pressures": 35, "success_rate": 68.5},
                {"player_id": 1002, "name": "Player B", "pressures": 32, "success_rate": 65.2},
                {"player_id": 1003, "name": "Player C", "pressures": 28, "success_rate": 71.4}
            ]
        }

        # Combine with base defensive metrics
        analysis = {**metrics, **pressing_metrics}

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache pressing analysis: {e}")

        return analysis

    def get_transition_analysis(self, team_id: int,
                             match_id: Optional[int] = None,
                             competition_id: Optional[int] = None,
                             season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get transition analysis for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with transition analysis
        """
        cache_key = f"transition_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached transition analysis: {e}")

        # Placeholder implementation
        analysis = {
            "team_id": team_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "offensive_transitions": {
                "count": 24,
                "success_rate": 45.8,  # Percentage
                "avg_duration": 6.2,  # seconds
                "xg_created": 0.85,
                "goals": 1,
                "key_pass_types": {
                    "forward": 65.0,  # Percentage
                    "lateral": 25.0,
                    "backward": 10.0
                },
                "recovery_to_shot_time": 8.5  # seconds
            },
            "defensive_transitions": {
                "count": 22,
                "success_rate": 68.2,  # Percentage of transitions stopped
                "avg_time_to_defensive_shape": 4.8,  # seconds
                "counterpressing_duration": 5.2,  # seconds
                "xg_conceded": 0.45,
                "goals_conceded": 0
            },
            "transition_zones": {
                "defensive_third": 35.0,  # Percentage of transitions starting in each third
                "middle_third": 50.0,
                "attacking_third": 15.0
            },
            "key_transition_players": [
                {"player_id": 1001, "name": "Player A", "transition_involvements": 12, "success_rate": 75.0},
                {"player_id": 1002, "name": "Player B", "transition_involvements": 10, "success_rate": 70.0},
                {"player_id": 1003, "name": "Player C", "transition_involvements": 8, "success_rate": 62.5}
            ]
        }

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache transition analysis: {e}")

        return analysis

    def get_set_piece_analysis(self, team_id: int,
                            match_id: Optional[int] = None,
                            competition_id: Optional[int] = None,
                            season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get set piece analysis for a team.

        Args:
            team_id: Team ID
            match_id: Optional match ID (if None, aggregates across matches)
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with set piece analysis
        """
        cache_key = f"set_piece_{team_id}"
        if match_id:
            cache_key += f"_match_{match_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached set piece analysis: {e}")

        # Placeholder implementation
        analysis = {
            "team_id": team_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "corners": {
                "total": 32,
                "shots_created": 8,
                "goals": 2,
                "xg": 1.85,
                "patterns": {
                    "short": 25.0,  # Percentage
                    "near_post": 30.0,
                    "far_post": 25.0,
                    "center": 20.0
                }
            },
            "free_kicks": {
                "total": 28,
                "direct_shots": 6,
                "goals": 1,
                "xg": 0.95,
                "patterns": {
                    "direct": 20.0,  # Percentage
                    "crossed": 50.0,
                    "short": 30.0
                }
            },
            "throw_ins": {
                "total": 45,
                "attacking_third": 15,
                "possession_retained_pct": 75.0,
                "patterns": {
                    "short": 65.0,  # Percentage
                    "long": 35.0
                }
            },
            "defensive_set_pieces": {
                "corners_faced": 30,
                "goals_conceded": 1,
                "xg_conceded": 1.25,
                "free_kicks_faced": 25,
                "free_kick_goals_conceded": 0,
                "free_kick_xg_conceded": 0.65
            },
            "key_set_piece_takers": [
                {"player_id": 1001, "name": "Player A", "set_pieces_taken": 18, "assists": 2, "xg_created": 1.2},
                {"player_id": 1002, "name": "Player B", "set_pieces_taken": 15, "assists": 1, "xg_created": 0.95}
            ],
            "key_set_piece_targets": [
                {"player_id": 1003, "name": "Player C", "set_piece_shots": 5, "goals": 1, "xg": 0.85},
                {"player_id": 1004, "name": "Player D", "set_piece_shots": 4, "goals": 1, "xg": 0.75}
            ]
        }

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache set piece analysis: {e}")

        return analysis

    def get_formation_analysis(self, team_id: int, match_id: int) -> Dict[str, Any]:
        """
        Get formation analysis for a team in a specific match.

        Args:
            team_id: Team ID
            match_id: Match ID

        Returns:
            Dictionary with formation analysis
        """
        cache_key = f"formation_{team_id}_match_{match_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached formation analysis: {e}")

        # Placeholder implementation
        analysis = {
            "team_id": team_id,
            "match_id": match_id,
            "base_formation": "4-3-3",
            "formation_changes": [
                {"minute": 0, "formation": "4-3-3", "reason": "Starting formation"},
                {"minute": 65, "formation": "4-2-3-1", "reason": "Tactical change"},
                {"minute": 80, "formation": "5-3-2", "reason": "Defending lead"}
            ],
            "avg_positions": {
                "first_half": [
                    {"player_id": 1001, "name": "Player A", "position": "GK", "x": 5, "y": 40},
                    {"player_id": 1002, "name": "Player B", "position": "RB", "x": 20, "y": 15},
                    {"player_id": 1003, "name": "Player C", "position": "CB", "x": 15, "y": 30},
                    {"player_id": 1004, "name": "Player D", "position": "CB", "x": 15, "y": 50},
                    {"player_id": 1005, "name": "Player E", "position": "LB", "x": 20, "y": 65},
                    {"player_id": 1006, "name": "Player F", "position": "CM", "x": 40, "y": 30},
                    {"player_id": 1007, "name": "Player G", "position": "CM", "x": 40, "y": 50},
                    {"player_id": 1008, "name": "Player H", "position": "CM", "x": 50, "y": 40},
                    {"player_id": 1009, "name": "Player I", "position": "RW", "x": 70, "y": 20},
                    {"player_id": 1010, "name": "Player J", "position": "ST", "x": 75, "y": 40},
                    {"player_id": 1011, "name": "Player K", "position": "LW", "x": 70, "y": 60}
                ],
                "second_half": [
                    # Similar structure to first_half
                ]
            },
            "positional_flexibility": {
                "most_flexible_players": [
                    {"player_id": 1006, "name": "Player F", "position_changes": 3},
                    {"player_id": 1008, "name": "Player H", "position_changes": 2}
                ],
                "formation_fluidity": 7.5  # Scale of 1-10
            },
            "defensive_shape": {
                "avg_defensive_block_width": 45.0,  # yards
                "avg_defensive_block_depth": 35.0,  # yards
                "defensive_line_height": 35.0  # yards from own goal
            },
            "attacking_shape": {
                "avg_attacking_width": 55.0,  # yards
                "avg_attacking_depth": 40.0,  # yards
                "forward_line_depth": 85.0  # yards from own goal
            }
        }

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache formation analysis: {e}")

        return analysis

    def get_team_style(self, team_id: int,
                    competition_id: Optional[int] = None,
                    season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get team playing style analysis.

        Args:
            team_id: Team ID
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with team style analysis
        """
        cache_key = f"team_style_{team_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis = pickle.load(f)
                    self._cached_analysis[cache_key] = analysis
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cached team style analysis: {e}")

        # Get offensive and defensive metrics
        offensive_metrics = self.get_offensive_metrics(team_id, None, competition_id, season_id)
        defensive_metrics = self.get_defensive_metrics(team_id, None, competition_id, season_id)

        # Placeholder implementation with real metrics integration
        build_up_score = offensive_metrics.get("passing", {}).get("avg_pass_completion", 0) / 100 * 10
        possession_score = offensive_metrics.get("avg_possession", 0) / 10
        pressing_score = 10 - (defensive_metrics.get("avg_ppda", 10) / 2)  # Lower PPDA = higher pressing score
        directness_score = 5.0  # Placeholder

        # Overall style categorization
        style_categories = {}
        style_categories["build_up"] = "Possession" if build_up_score > 7.5 else "Mixed" if build_up_score > 5 else "Direct"
        style_categories["pressing"] = "High Press" if pressing_score > 7.5 else "Mid Block" if pressing_score > 5 else "Low Block"
        style_categories["attacking"] = "Positional" if possession_score > 6.5 else "Counter-attacking" if directness_score > 7.5 else "Mixed"

        # Primary style definition
        if build_up_score > 7.5 and possession_score > 6.5:
            primary_style = "Possession-based"
        elif pressing_score > 7.5 and directness_score > 6:
            primary_style = "High-pressing Transition"
        elif directness_score > 7.5 and possession_score < 5:
            primary_style = "Direct Counter-attacking"
        elif build_up_score > 6 and pressing_score > 6:
            primary_style = "Balanced"
        elif pressing_score < 4 and possession_score < 5:
            primary_style = "Defensive Low Block"
        else:
            primary_style = "Mixed"

        # Combine into final analysis
        analysis = {
            "team_id": team_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "matches_analyzed": offensive_metrics.get("matches_analyzed", 0),
            "primary_style": primary_style,
            "style_categories": style_categories,
            "style_scores": {
                "build_up": build_up_score,
                "possession": possession_score,
                "pressing": pressing_score,
                "directness": directness_score
            },
            "key_metrics": {
                "possession": offensive_metrics.get("avg_possession", 0),
                "ppda": defensive_metrics.get("avg_ppda", 0),
                "pass_completion": offensive_metrics.get("passing", {}).get("avg_pass_completion", 0),
                "xg_per_match": offensive_metrics.get("shooting", {}).get("xg_per_match", 0),
                "xg_against_per_match": defensive_metrics.get("avg_opponent_xg", 0),
                "shots_per_match": offensive_metrics.get("shooting", {}).get("shots_per_match", 0)
            }
        }

        # Cache the results
        self._cached_analysis[cache_key] = analysis

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to cache team style analysis: {e}")

        return analysis

    def compare_team_styles(self, team_id1: int, team_id2: int,
                         competition_id: Optional[int] = None,
                         season_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare playing styles between two teams.

        Args:
            team_id1: First team ID
            team_id2: Second team ID
            competition_id: Optional competition ID filter
            season_id: Optional season ID filter

        Returns:
            Dictionary with style comparison
        """
        cache_key = f"style_comparison_{team_id1}_{team_id2}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    comparison = pickle.load(f)
                    self._cached_analysis[cache_key] = comparison
                    return comparison
            except Exception as e:
                logger.warning(f"Failed to load cached style comparison: {e}")

        # Get team styles
        team1_style = self.get_team_style(team_id1, competition_id, season_id)
        team2_style = self.get_team_style(team_id2, competition_id, season_id)

        # Calculate style differences
        style_diffs = {}
        for key in team1_style.get("style_scores", {}):
            score1 = team1_style["style_scores"].get(key, 0)
            score2 = team2_style["style_scores"].get(key, 0)
            style_diffs[key] = score2 - score1

        # Calculate metric differences
        metric_diffs = {}
        for key in team1_style.get("key_metrics", {}):
            metric1 = team1_style["key_metrics"].get(key, 0)
            metric2 = team2_style["key_metrics"].get(key, 0)
            metric_diffs[key] = metric2 - metric1

        # Calculate style similarity
        similarity_score = 0
        for key in style_diffs:
            # Convert difference to similarity (10 = identical, 0 = completely different)
            similarity_score += 10 - min(10, abs(style_diffs[key]))

        # Average similarity
        similarity_score = similarity_score / len(style_diffs) if style_diffs else 0

        # Comparative advantage assessment
        advantages = {}
        if team1_style["style_categories"]["pressing"] == "High Press" and team2_style["style_categories"]["build_up"] == "Direct":
            advantages["team1"] = "High press may disrupt direct build-up"
        if team2_style["style_categories"]["pressing"] == "High Press" and team1_style["style_categories"]["build_up"] == "Direct":
            advantages["team2"] = "High press may disrupt direct build-up"
        if team1_style["style_categories"]["build_up"] == "Possession" and team2_style["style_categories"]["pressing"] == "Low Block":
            advantages["team1"] = "Possession build-up effective against low block"
        if team2_style["style_categories"]["build_up"] == "Possession" and team1_style["style_categories"]["pressing"] == "Low Block":
            advantages["team2"] = "Possession build-up effective against low block"

        # Combine into comparison
        comparison = {
            "team1": {
                "team_id": team_id1,
                "style": team1_style
            },
            "team2": {
                "team_id": team_id2,
                "style": team2_style
            },
            "style_differences": style_diffs,
            "metric_differences": metric_diffs,
            "style_similarity_score": similarity_score,
            "comparative_advantages": advantages
        }

        # Cache the results
        self._cached_analysis[cache_key] = comparison

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(comparison, f)
        except Exception as e:
            logger.warning(f"Failed to cache style comparison: {e}")

        return comparison

    def _get_match_info(self, match_id: int) -> Dict[str, Any]:
        """Get basic match information"""
        data_manager = self.metrics_engine.data_manager

        # This would normally query the match from the data manager
        # For now, return a placeholder that would be replaced with actual implementation
        return {
            "match_id": match_id,
            "home_team": f"Home Team for {match_id}",
            "away_team": f"Away Team for {match_id}",
            "score": "0-0",
            "competition": f"Competition for {match_id}",
            "season": f"Season for {match_id}",
            "date": "2023-01-01"
        }
