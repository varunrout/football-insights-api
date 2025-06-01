# app/services/metrics_engine.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
from pathlib import Path

from app.util.football_data_manager import FootballDataManager
from app.util.metrics.expected_threat import ExpectedThreatModel
from app.util.metrics.ppda import calculate_ppda
from app.util.metrics.pass_network import generate_pass_network
from app.config.environment import settings

logger = logging.getLogger(__name__)

class MetricsEngine:
    """
    Orchestrates the calculation of various football metrics across matches and players.
    Works in conjunction with FootballDataManager to access and process data.
    """

    def __init__(self, data_manager: Optional[FootballDataManager] = None):
        """
        Initialize the MetricsEngine.

        Args:
            data_manager: FootballDataManager instance for data access
        """
        self.data_manager = data_manager or FootballDataManager()
        self.metrics_cache_dir = Path(settings.DATA_CACHE_DIR) / "metrics"
        self.metrics_cache_dir.mkdir(exist_ok=True, parents=True)

        # Metric model instances
        self._xt_model = None  # Lazy-loaded
        self._cached_match_metrics = {}
        self._cached_player_metrics = {}

    @property
    def xt_model(self) -> ExpectedThreatModel:
        """Get or initialize the Expected Threat model"""
        if self._xt_model is None:
            # Try to load from cache first
            model_path = self.metrics_cache_dir / "xt_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self._xt_model = pickle.load(f)
                        logger.info("Loaded xT model from cache")
                except Exception as e:
                    logger.warning(f"Failed to load xT model from cache: {e}")
                    self._xt_model = ExpectedThreatModel()
                    self._xt_model.initialize()
            else:
                logger.info("Initializing new xT model")
                self._xt_model = ExpectedThreatModel()
                self._xt_model.initialize()

                # Save model to cache
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(self._xt_model, f)
                        logger.info("Saved xT model to cache")
                except Exception as e:
                    logger.warning(f"Failed to save xT model to cache: {e}")

        return self._xt_model

    def calculate_match_metrics(self, match_id: int, force_recalculation: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a specific match.

        Args:
            match_id: The match ID
            force_recalculation: If True, recalculates even if cached

        Returns:
            Dictionary of match metrics
        """
        # Check cache first
        if match_id in self._cached_match_metrics and not force_recalculation:
            return self._cached_match_metrics[match_id]

        # Load cached metrics from disk if available
        cache_path = self.metrics_cache_dir / f"match_{match_id}_metrics.pkl"
        if cache_path.exists() and not force_recalculation:
            try:
                with open(cache_path, 'rb') as f:
                    metrics = pickle.load(f)
                    self._cached_match_metrics[match_id] = metrics
                    return metrics
            except Exception as e:
                logger.warning(f"Failed to load cached metrics for match {match_id}: {e}")

        # Get match events
        events = self.data_manager.get_events(match_id)

        # Get match information (home and away teams)
        match_info = self._get_match_info(match_id)

        # Initialize metrics dictionary
        metrics = {"match_id": match_id, "home_team_name": match_info.get("home_team_name", "Unknown"),
                   "away_team_name": match_info.get("away_team_name", "Unknown"), "score": match_info.get("score", "0-0"),
                   "possession": self._calculate_possession(events), "shots": self._calculate_shot_metrics(events),
                   "passes": self._calculate_pass_metrics(events),
                   "defensive": self._calculate_defensive_metrics(events), "xt": self._calculate_xt_metrics(events),
                   "ppda": calculate_ppda(events)}

        # Add possession adjusted metrics

        # Calculate pass networks
        home_team_id = match_info.get("home_team_id")
        away_team_id = match_info.get("away_team_id")

        if home_team_id:
            metrics["home_pass_network"] = generate_pass_network(
                events, team_id=home_team_id)

        if away_team_id:
            metrics["away_pass_network"] = generate_pass_network(
                events, team_id=away_team_id)

        # Calculate player-specific metrics
        metrics["player_metrics"] = self._calculate_player_match_metrics(events, match_id)

        # Cache the results
        self._cached_match_metrics[match_id] = metrics

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(metrics, f)
        except Exception as e:
            logger.warning(f"Failed to cache metrics for match {match_id}: {e}")

        return metrics

    def calculate_player_metrics(self, player_id: int,
                              competition_id: Optional[int] = None,
                              team_id: Optional[int] = None,
                              season_id: Optional[int] = None,
                              force_recalculation: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a specific player.

        Args:
            player_id: The player ID
            competition_id: Optional filter by competition
            team_id: Optional filter by team
            season_id: Optional filter by season
            force_recalculation: If True, recalculates even if cached

        Returns:
            Dictionary of player metrics
        """
        # Generate cache key
        cache_key = f"player_{player_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if team_id:
            cache_key += f"_team_{team_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check memory cache
        if cache_key in self._cached_player_metrics and not force_recalculation:
            return self._cached_player_metrics[cache_key]

        # Check disk cache
        cache_path = self.metrics_cache_dir / f"{cache_key}.pkl"
        if cache_path.exists() and not force_recalculation:
            try:
                with open(cache_path, 'rb') as f:
                    metrics = pickle.load(f)
                    self._cached_player_metrics[cache_key] = metrics
                    return metrics
            except Exception as e:
                logger.warning(f"Failed to load cached metrics for {cache_key}: {e}")

        # Fetch the matches the player participated in
        matches = self._get_player_matches(player_id, competition_id, team_id, season_id)

        if not matches:
            logger.warning(f"No matches found for player {player_id} with the specified filters")
            return {
                "player_id": player_id,
                "matches_found": 0,
                "metrics": {}
            }

        # Collect metrics across all matches
        all_match_metrics = []
        total_minutes = 0

        for match_id in matches:
            match_events = self.data_manager.get_events(match_id)
            player_match_metrics = self._extract_player_match_metrics(
                match_events, player_id, match_id)

            if player_match_metrics:
                all_match_metrics.append(player_match_metrics)
                total_minutes += player_match_metrics.get("minutes_played", 0)

        # Aggregate metrics across matches
        aggregated_metrics = self._aggregate_player_metrics(all_match_metrics)

        # Calculate per 90 metrics
        per_90_metrics = {}
        if total_minutes > 0:
            minutes_per_90 = total_minutes / 90
            for key, value in aggregated_metrics.items():
                # Only calculate per 90 for count metrics, not percentages or rates
                if (isinstance(value, (int, float)) and
                    not key.endswith(("_pct", "_rate", "_accuracy"))):
                    per_90_metrics[f"{key}_per_90"] = value / minutes_per_90

        # Compile final metrics
        metrics = {
            "player_id": player_id,
            "matches_found": len(matches),
            "total_minutes": total_minutes,
            "metrics": aggregated_metrics,
            "per_90_metrics": per_90_metrics,
            "match_metrics": all_match_metrics
        }

        # Cache the results
        self._cached_player_metrics[cache_key] = metrics

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(metrics, f)
        except Exception as e:
            logger.warning(f"Failed to cache metrics for {cache_key}: {e}")

        return metrics

    def calculate_team_metrics(self, team_id: int,
                            competition_id: Optional[int] = None,
                            season_id: Optional[int] = None,
                            force_recalculation: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a specific team.

        Args:
            team_id: The team ID
            competition_id: Optional filter by competition
            season_id: Optional filter by season
            force_recalculation: If True, recalculates even if cached

        Returns:
            Dictionary of team metrics
        """
        # Generate cache key
        cache_key = f"team_{team_id}"
        if competition_id:
            cache_key += f"_comp_{competition_id}"
        if season_id:
            cache_key += f"_season_{season_id}"

        # Check disk cache
        cache_path = self.metrics_cache_dir / f"{cache_key}.pkl"
        if cache_path.exists() and not force_recalculation:
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached metrics for {cache_key}: {e}")

        # Get team matches
        matches = self._get_team_matches(team_id, competition_id, season_id)

        if not matches:
            logger.warning(f"No matches found for team {team_id} with the specified filters")
            return {
                "team_id": team_id,
                "matches_found": 0,
                "metrics": {}
            }

        # Calculate metrics for each match
        match_metrics = []
        for match_id in matches:
            metrics = self.calculate_match_metrics(match_id)
            # Extract team-specific metrics from the match
            team_match_metrics = self._extract_team_match_metrics(metrics, team_id)
            match_metrics.append(team_match_metrics)

        # Aggregate metrics across matches
        aggregated_metrics = self._aggregate_team_metrics(match_metrics)

        # Compile final metrics
        metrics = {
            "team_id": team_id,
            "matches_found": len(matches),
            "metrics": aggregated_metrics,
            "match_metrics": match_metrics
        }

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(metrics, f)
        except Exception as e:
            logger.warning(f"Failed to cache metrics for {cache_key}: {e}")

        return metrics

    def calculate_xt_contribution(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Expected Threat (xT) contribution for events.

        Args:
            events_df: DataFrame of events

        Returns:
            DataFrame with xT contribution values
        """
        from app.services.metric_calculator import calculate_xt_added

        # Use the consolidated function from metric_calculator
        events_with_xt = calculate_xt_added(events_df, self.xt_model)

        # Filter for pass events only
        pass_events = events_with_xt[events_with_xt['type_name'] == 'Pass'].copy()

        if pass_events.empty:
            return pd.DataFrame()

        # Keep only relevant columns
        result = pass_events[['id', 'player_id', 'team_id', 'minute', 'second',
                             'location', 'pass_end_location', 'xt_added']]

        # Rename for consistent API
        result = result.rename(columns={'xt_added': 'xt_contribution'})

        return result

    def _get_match_info(self, match_id: int) -> Dict[str, Any]:
        """Get basic match information"""
        # This would normally query the match from the data manager
        # For now, return a placeholder that would be replaced with actual implementation
        return {
            "match_id": match_id,
            "home_team_name": f"Home Team for {match_id}",
            "away_team_name": f"Away Team for {match_id}",
            "score": "0-0",
            "home_team_id": None,  # Would be a real ID in actual implementation
            "away_team_id": None   # Would be a real ID in actual implementation
        }

    def _calculate_possession(self, events_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate possession metrics from events"""
        # This would calculate possession percentages, durations, etc.
        # Placeholder implementation
        return {
            "home_possession": 50.0,
            "away_possession": 50.0,
            "home_possession_final_third": 30.0,
            "away_possession_final_third": 30.0
        }

    def _calculate_shot_metrics(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate shot-related metrics from events"""
        # This would calculate shot metrics like xG, shot locations, etc.
        # Placeholder implementation
        return {
            "home_shots": 10,
            "away_shots": 8,
            "home_shots_on_target": 4,
            "away_shots_on_target": 3,
            "home_xg": 1.2,
            "away_xg": 0.8,
            "home_goals": 1,
            "away_goals": 0
        }

    def _calculate_pass_metrics(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate pass-related metrics from events"""
        # This would calculate pass metrics like completion rate, progressive passes, etc.
        # Placeholder implementation
        return {
            "home_passes": 450,
            "away_passes": 380,
            "home_pass_completion": 85.0,
            "away_pass_completion": 80.0,
            "home_progressive_passes": 45,
            "away_progressive_passes": 35,
            "home_passes_final_third": 120,
            "away_passes_final_third": 90
        }

    def _calculate_defensive_metrics(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate defensive metrics from events"""
        # This would calculate defensive metrics like tackles, interceptions, etc.
        # Placeholder implementation
        return {
            "home_defensive_actions": 35,
            "away_defensive_actions": 42,
            "home_tackles": 15,
            "away_tackles": 18,
            "home_interceptions": 8,
            "away_interceptions": 12,
            "home_blocks": 5,
            "away_blocks": 6,
            "home_clearances": 7,
            "away_clearances": 10
        }

    def _calculate_xt_metrics(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Expected Threat metrics from events"""
        from app.services.metric_calculator import calculate_xt_added

        # Use the consolidated method from metric_calculator for calculating xT
        events_with_xt = calculate_xt_added(events_df, self.xt_model)

        # Extract team-specific metrics (use new schema: team_id, type_name)
        teams = events_df['team_id'].dropna().unique()
        if len(teams) < 2:
            return {
                "home_xt": 0.0,
                "away_xt": 0.0,
                "home_xt_passes": 0.0,
                "away_xt_passes": 0.0,
                "home_xt_carries": 0.0,
                "away_xt_carries": 0.0
            }

        home_team = teams[0]
        away_team = teams[1]

        # Calculate xT for each team
        home_xt_events = events_with_xt[events_with_xt['team_id'] == home_team]
        away_xt_events = events_with_xt[events_with_xt['team_id'] == away_team]

        # Aggregate by event type (use type_name)
        home_xt_total = home_xt_events['xt_added'].sum()
        away_xt_total = away_xt_events['xt_added'].sum()

        home_xt_passes = home_xt_events[home_xt_events['type_name'] == 'Pass']['xt_added'].sum()
        away_xt_passes = away_xt_events[away_xt_events['type_name'] == 'Pass']['xt_added'].sum()

        home_xt_carries = home_xt_events[home_xt_events['type_name'] == 'Carry']['xt_added'].sum()
        away_xt_carries = away_xt_events[away_xt_events['type_name'] == 'Carry']['xt_added'].sum()

        return {
            "home_xt": home_xt_total,
            "away_xt": away_xt_total,
            "home_xt_passes": home_xt_passes,
            "away_xt_passes": away_xt_passes,
            "home_xt_carries": home_xt_carries,
            "away_xt_carries": away_xt_carries
        }

    def _calculate_player_match_metrics(self, events_df: pd.DataFrame, match_id: int) -> Dict[str, Dict[str, Any]]:
        """Calculate player-specific metrics for a match"""
        # This would calculate player-level metrics
        # Placeholder implementation
        player_metrics = {}

        # Extract unique players (use player_id)
        players = events_df['player_id'].dropna().unique()

        for player in players:
            # Get player's team
            player_events = events_df[events_df['player_id'] == player]
            team = player_events['team_id'].iloc[0] if len(player_events) > 0 else "Unknown"

            # Calculate minutes played (simplified)
            max_minute = player_events['minute'].max() if len(player_events) > 0 else 0
            min_minute = player_events['minute'].min() if len(player_events) > 0 else 0
            minutes_played = max_minute - min_minute

            # Generate placeholder metrics (use type_name)
            player_metrics[player] = {
                "player_id": player,
                "team_id": team,
                "match_id": match_id,
                "minutes_played": minutes_played,
                "passes": len(player_events[player_events['type_name'] == 'Pass']),
                "shots": len(player_events[player_events['type_name'] == 'Shot']),
                "xg": sum(player_events.get('shot_statsbomb_xg', 0)),
                "goals": len(player_events[(player_events['type_name'] == 'Shot') &
                                      (player_events['shot_outcome'] == 'Goal')]),
                "assists": 0,  # Would need shot outcome + related pass info
                "defensive_actions": len(player_events[player_events['type_name'].isin(['Tackle', 'Interception', 'Block'])])
            }

        return player_metrics

    def _get_player_matches(self, player_id: int, competition_id: Optional[int] = None,
                         team_id: Optional[int] = None, season_id: Optional[int] = None) -> List[int]:
        """Get matches that a player participated in using FootballDataManager."""
        # Use FootballDataManager to get matches for the player
        if competition_id is not None and season_id is not None:
            matches_df = self.data_manager.get_matches(competition_id, season_id)
            if team_id is not None:
                matches_df = matches_df[(matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)]
            # Filter matches where player is in lineup (if available)
            # For now, return all match_ids (could be improved with lineup check)
            return matches_df['match_id'].tolist()
        return []

    def _get_team_matches(self, team_id: int, competition_id: Optional[int] = None,
                       season_id: Optional[int] = None) -> List[int]:
        """Get matches that a team participated in using FootballDataManager."""
        if competition_id is not None and season_id is not None:
            matches_df = self.data_manager.get_matches(competition_id, season_id)
            matches_df = matches_df[(matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)]
            return matches_df['match_id'].tolist()
        return []

    def _extract_player_match_metrics(self, events_df: pd.DataFrame, player_id: int,
                                  match_id: int) -> Optional[Dict[str, Any]]:
        """Extract metrics for a specific player in a match"""
        # This would extract player metrics from match events
        # Placeholder implementation
        return {
            "player_id": player_id,
            "match_id": match_id,
            "minutes_played": 90,
            "passes": 45,
            "pass_accuracy": 85.0,
            "shots": 2,
            "xg": 0.3,
            "goals": 0,
            "assists": 0,
            "defensive_actions": 5
        }

    def _extract_team_match_metrics(self, match_metrics: Dict[str, Any], team_id: int) -> Dict[str, Any]:
        """Extract team-specific metrics from match metrics using new schema"""
        # Determine if the team is home or away
        home_team_id = match_metrics.get("home_team_id")
        away_team_id = match_metrics.get("away_team_id")
        if home_team_id is not None and team_id == home_team_id:
            team_key = "home"
            opp_key = "away"
        elif away_team_id is not None and team_id == away_team_id:
            team_key = "away"
            opp_key = "home"
        else:
            # Fallback: unknown team role
            team_key = "home"
            opp_key = "away"

        return {
            "team_id": team_id,
            "match_id": match_metrics.get("match_id"),
            "possession": match_metrics.get("possession", {}).get(f"{team_key}_possession", 0),
            "passes": match_metrics.get("passes", {}).get(f"{team_key}_passes", 0),
            "pass_completion": match_metrics.get("passes", {}).get(f"{team_key}_pass_completion", 0),
            "shots": match_metrics.get("shots", {}).get(f"{team_key}_shots", 0),
            "xg_for": match_metrics.get("shots", {}).get(f"{team_key}_xg", 0),
            "goals_for": match_metrics.get("shots", {}).get(f"{team_key}_goals", 0),
            "goals_against": match_metrics.get("shots", {}).get(f"{opp_key}_goals", 0),
            "xg_against": match_metrics.get("shots", {}).get(f"{opp_key}_xg", 0),
            "defensive_actions": match_metrics.get("defensive", {}).get(f"{team_key}_defensive_actions", 0),
            "ppda": match_metrics.get("ppda", {}).get(team_key, {}).get("ppda", 0)
        }

    def _aggregate_player_metrics(self, match_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate player metrics across multiple matches"""
        # This would aggregate metrics across matches
        # Placeholder implementation
        if not match_metrics:
            return {}

        aggregated = {
            "matches_played": len(match_metrics),
            "minutes_played": sum(m.get("minutes_played", 0) for m in match_metrics),
            "goals": sum(m.get("goals", 0) for m in match_metrics),
            "assists": sum(m.get("assists", 0) for m in match_metrics),
            "xg": sum(m.get("xg", 0) for m in match_metrics),
            "shots": sum(m.get("shots", 0) for m in match_metrics),
            "passes": sum(m.get("passes", 0) for m in match_metrics),
            "defensive_actions": sum(m.get("defensive_actions", 0) for m in match_metrics)
        }

        # Calculate averages
        if aggregated["matches_played"] > 0:
            aggregated["goals_per_match"] = aggregated["goals"] / aggregated["matches_played"]
            aggregated["shots_per_match"] = aggregated["shots"] / aggregated["matches_played"]

        if aggregated["minutes_played"] > 0:
            mins_per_90 = aggregated["minutes_played"] / 90
            aggregated["goals_per_90"] = aggregated["goals"] / mins_per_90
            aggregated["shots_per_90"] = aggregated["shots"] / mins_per_90

        return aggregated

    def _aggregate_team_metrics(self, match_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate team metrics across multiple matches"""
        # This would aggregate metrics across matches
        # Placeholder implementation
        if not match_metrics:
            return {}

        aggregated = {
            "matches_played": len(match_metrics),
            "wins": sum(1 for m in match_metrics if m.get("goals_for", 0) > m.get("goals_against", 0)),
            "draws": sum(1 for m in match_metrics if m.get("goals_for", 0) == m.get("goals_against", 0)),
            "losses": sum(1 for m in match_metrics if m.get("goals_for", 0) < m.get("goals_against", 0)),
            "goals_for": sum(m.get("goals_for", 0) for m in match_metrics),
            "goals_against": sum(m.get("goals_against", 0) for m in match_metrics),
            "xg_for": sum(m.get("xg_for", 0) for m in match_metrics),
            "xg_against": sum(m.get("xg_against", 0) for m in match_metrics),
            "shots": sum(m.get("shots", 0) for m in match_metrics),
            "passes": sum(m.get("passes", 0) for m in match_metrics),
            "possession": sum(m.get("possession", 0) for m in match_metrics),
            "defensive_actions": sum(m.get("defensive_actions", 0) for m in match_metrics)
        }

        # Calculate averages
        if aggregated["matches_played"] > 0:
            aggregated["points"] = aggregated["wins"] * 3 + aggregated["draws"]
            aggregated["points_per_match"] = aggregated["points"] / aggregated["matches_played"]
            aggregated["goals_for_per_match"] = aggregated["goals_for"] / aggregated["matches_played"]
            aggregated["goals_against_per_match"] = aggregated["goals_against"] / aggregated["matches_played"]
            aggregated["avg_possession"] = aggregated["possession"] / aggregated["matches_played"]

        return aggregated
