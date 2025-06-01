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
from app.util.metrics.ppda import calculate_ppda
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
            # Get match metrics from the metrics engine instead of recalculating
            match_metrics = self.metrics_engine.calculate_match_metrics(match_id)

            # Extract team-specific defensive metrics
            if match_metrics:
                # Determine if team is home or away
                is_home = False
                if match_metrics.get("home_team_id") is not None:
                    is_home = match_metrics.get("home_team_id") == team_id
                team_key = "home" if is_home else "away"
                opp_key = "away" if is_home else "home"

                # Extract defensive metrics directly from match metrics
                defensive_metrics = match_metrics.get("defensive", {})
                ppda_metrics = match_metrics.get("ppda", {}).get(team_key, {})

                metrics = {
                    "team_id": team_id,
                    "match_id": match_id,
                    "ppda": ppda_metrics.get("ppda", 0),
                    "defensive_actions": {
                        "total": defensive_metrics.get(f"{team_key}_defensive_actions", 0),
                        "tackles": defensive_metrics.get(f"{team_key}_tackles", 0),
                        "interceptions": defensive_metrics.get(f"{team_key}_interceptions", 0),
                        "blocks": defensive_metrics.get(f"{team_key}_blocks", 0),
                        "clearances": defensive_metrics.get(f"{team_key}_clearances", 0)
                    },
                    "pressure": match_metrics.get("pressure", {}).get(team_key, {}),
                    "opposition_passes": ppda_metrics.get("opposition_passes", 0),
                    "challenges": ppda_metrics.get("challenges", 0),
                    "opponent_xg": match_metrics.get("shots", {}).get(f"{opp_key}_xg", 0)
                }

                # Get PPDA timeline if events are available
                if "events" in match_metrics:
                    from app.util.metrics.ppda import calculate_ppda_timeline
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

        # For multi-match analysis, use the team metrics from metrics_engine
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
            # Get match metrics from the metrics engine instead of recalculating
            match_metrics = self.metrics_engine.calculate_match_metrics(match_id)

            # Extract team-specific offensive metrics
            if match_metrics:
                # Determine if team is home or away
                is_home = False
                if match_metrics.get("home_team_id") is not None:
                    is_home = match_metrics.get("home_team_id") == team_id
                team_key = "home" if is_home else "away"

                # Extract offensive metrics directly from match metrics
                pass_metrics = match_metrics.get("passes", {})
                shot_metrics = match_metrics.get("shots", {})
                possession_metrics = match_metrics.get("possession", {})
                xt_metrics = match_metrics.get("xt", {})

                metrics = {
                    "team_id": team_id,
                    "match_id": match_id,
                    "possession": possession_metrics.get(f"{team_key}_possession", 0),
                    "passing": {
                        "total": pass_metrics.get(f"{team_key}_passes", 0),
                        "completion_rate": pass_metrics.get(f"{team_key}_pass_completion", 0),
                        "progressive_passes": pass_metrics.get(f"{team_key}_progressive_passes", 0),
                        "passes_final_third": pass_metrics.get(f"{team_key}_passes_final_third", 0)
                    },
                    "shooting": {
                        "shots": shot_metrics.get(f"{team_key}_shots", 0),
                        "shots_on_target": shot_metrics.get(f"{team_key}_shots_on_target", 0),
                        "goals": shot_metrics.get(f"{team_key}_goals", 0),
                        "xg": shot_metrics.get(f"{team_key}_xg", 0)
                    },
                    "xt": {
                        "total": xt_metrics.get(f"{team_key}_xt", 0),
                        "passes": xt_metrics.get(f"{team_key}_xt_passes", 0),
                        "carries": xt_metrics.get(f"{team_key}_xt_carries", 0)
                    }
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

        # For multi-match analysis, use the team metrics from metrics_engine
        team_metrics = self.metrics_engine.calculate_team_metrics(
            team_id, competition_id, season_id)

        if team_metrics and "matches_found" in team_metrics and team_metrics["matches_found"] > 0:
            # Extract and aggregate offensive metrics across matches
            aggregated = team_metrics.get("metrics", {})
            match_count = team_metrics["matches_found"]

            # Use .get with default 0 to avoid KeyError
            total_passes = aggregated.get("passes", 0)
            total_progressive_passes = aggregated.get("progressive_passes", 0)
            total_shots = aggregated.get("shots", 0)
            shots_on_target = aggregated.get("shots_on_target", 0)
            total_xg = aggregated.get("xg_for", 0)
            total_goals = aggregated.get("goals_for", 0)
            avg_pass_completion = aggregated.get("pass_completion", 0)
            avg_possession = aggregated.get("avg_possession", 0)

            metrics = {
                "team_id": team_id,
                "competition_id": competition_id,
                "season_id": season_id,
                "matches_analyzed": match_count,
                "avg_possession": avg_possession,
                "passing": {
                    "avg_passes_per_match": total_passes / match_count if match_count > 0 else 0,
                    "avg_pass_completion": avg_pass_completion,
                    "avg_progressive_passes": total_progressive_passes / match_count if match_count > 0 else 0
                },
                "shooting": {
                    "total_shots": total_shots,
                    "shots_per_match": total_shots / match_count if match_count > 0 else 0,
                    "shots_on_target": shots_on_target,
                    "shot_accuracy": (shots_on_target / total_shots * 100) if total_shots > 0 else 0,
                    "total_xg": total_xg,
                    "xg_per_match": total_xg / match_count if match_count > 0 else 0,
                },
                "goals": {
                    "total": total_goals,
                    "per_match": total_goals / match_count if match_count > 0 else 0,
                    "xg_difference": total_goals - total_xg
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
        avg_pass_completion = offensive_metrics.get("passing", {}).get("avg_pass_completion", 0)
        avg_possession = offensive_metrics.get("avg_possession", 0)
        avg_ppda = defensive_metrics.get("avg_ppda", 10)

        build_up_score = (avg_pass_completion / 100 * 10) if avg_pass_completion else 0
        possession_score = (avg_possession / 10) if avg_possession else 0
        pressing_score = 10 - (avg_ppda / 2) if avg_ppda else 0
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
