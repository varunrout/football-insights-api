from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/head-to-head")
async def get_head_to_head(
    team1_id: int = Query(..., description="First team ID"),
    team2_id: int = Query(..., description="Second team ID"),
    competition_id: Optional[int] = Query(None, description="Filter by competition ID"),
    season_id: Optional[int] = Query(None, description="Filter by season ID"),
    last_n: Optional[int] = Query(5, description="Number of most recent matches to include"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get head-to-head comparison between two teams
    Returns historical matchup data and performance metrics
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        matches_df = fdm.get_matches_for_team(competition_id, season_id, team1_id)
        if matches_df is None or matches_df.empty:
            return {"error": "No matches found for team1 in this competition/season."}
        # Find matches where both teams played
        h2h_matches = matches_df[((matches_df['home_team_id'] == team1_id) & (matches_df['away_team_id'] == team2_id)) |
                                 ((matches_df['home_team_id'] == team2_id) & (matches_df['away_team_id'] == team1_id))]
        h2h_matches = h2h_matches.sort_values("match_date", ascending=False).head(last_n)
        historical_matches = []
        team1_wins = team2_wins = draws = team1_goals = team2_goals = team1_xg = team2_xg = 0
        for _, match in h2h_matches.iterrows():
            match_id = match['match_id']
            events = fdm.get_events(match_id)
            # Get goals and xG
            home_goals = len(events[(events['team_id'] == match['home_team_id']) & (events['type_name'] == 'Shot') & (events['shot_outcome'] == 'Goal')])
            away_goals = len(events[(events['team_id'] == match['away_team_id']) & (events['type_name'] == 'Shot') & (events['shot_outcome'] == 'Goal')])
            home_xg = events[events['team_id'] == match['home_team_id']]['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in events else 0
            away_xg = events[events['team_id'] == match['away_team_id']]['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in events else 0
            # Determine winner
            if home_goals > away_goals:
                winner = 'team1' if match['home_team_id'] == team1_id else 'team2'
            elif away_goals > home_goals:
                winner = 'team2' if match['away_team_id'] == team2_id else 'team1'
            else:
                winner = 'draw'
            # Update summary
            if winner == 'team1':
                team1_wins += 1
            elif winner == 'team2':
                team2_wins += 1
            else:
                draws += 1
            if match['home_team_id'] == team1_id:
                team1_goals += home_goals
                team2_goals += away_goals
                team1_xg += home_xg
                team2_xg += away_xg
            else:
                team1_goals += away_goals
                team2_goals += home_goals
                team1_xg += away_xg
                team2_xg += home_xg
            historical_matches.append({
                "match_id": match_id,
                "date": match.get('match_date', None),
                "competition": match.get('competition_name', None),
                "home_team": match.get('home_team', None),
                "away_team": match.get('away_team', None),
                "score": f"{home_goals}-{away_goals}",
                "winner": winner,
                "team1_xg": team1_xg,
                "team2_xg": team2_xg
            })
        return {
            "team1": {
                "id": team1_id,
                "name": None
            },
            "team2": {
                "id": team2_id,
                "name": None
            },
            "summary": {
                "matches_played": len(historical_matches),
                "team1_wins": team1_wins,
                "team2_wins": team2_wins,
                "draws": draws,
                "team1_goals": team1_goals,
                "team2_goals": team2_goals,
                "team1_xg": team1_xg,
                "team2_xg": team2_xg
            },
            "historical_matches": historical_matches,
            "competition_id": competition_id,
            "season_id": season_id
        }
    except Exception as e:
        logger.error(f"Error getting head-to-head analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/team-style")
async def get_team_style_comparison(
    team_id: int = Query(..., description="Team ID to analyze"),
    competition_id: int = Query(..., description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    compare_to: Optional[List[int]] = Query(None, description="Team IDs to compare against (None = league average)"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get team style comparison
    Returns team playing style metrics compared to other teams or league average
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        matches_df = fdm.get_matches_for_team(competition_id, season_id, team_id)
        if matches_df is None or matches_df.empty:
            return {"error": "No matches found for team in this competition/season."}
        all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
        if not all_events:
            return {"error": "No data found for competition/season/team."}
        events = pd.concat(all_events)
        # Filter to team and league
        team_events = events[events['team_id'] == team_id]
        league_teams = events['team_id'].unique()
        style_metrics = [
            "possession", "build_up_speed", "width", "directness", "pressing_intensity",
            "defensive_line_height", "counter_attack_frequency", "set_piece_reliance"
        ]
        def calc_style(ev):
            total_passes = len(ev[ev['type_name'] == 'Pass'])
            forward_passes = len(ev[(ev['type_name'] == 'Pass') & (ev.get('pass_angle', 0) < 45)])
            possession = 100 * len(ev[ev['possession_team_id'] == team_id]) / max(1, len(ev))
            pressing = len(ev[ev['type_name'] == 'Pressure']) / max(1, len(ev)) * 100
            return {
                "possession": possession,
                "build_up_speed": 0,  # Placeholder
                "width": 0,  # Placeholder
                "directness": forward_passes / max(1, total_passes) * 100 if total_passes > 0 else 0,
                "pressing_intensity": pressing,
                "defensive_line_height": 0,  # Placeholder
                "counter_attack_frequency": 0,  # Placeholder
                "set_piece_reliance": 0  # Placeholder
            }
        team_style = calc_style(team_events)
        # League average
        league_styles = [calc_style(events[events['team_id'] == tid]) for tid in league_teams]
        league_avg = {k: float(np.mean([d[k] for d in league_styles])) for k in style_metrics}
        comparisons = [{"team_id": None, "team_name": "League Average", "style": league_avg}]
        # Add specific team comparisons if requested
        if compare_to:
            for comp_team_id in compare_to:
                comp_style = calc_style(events[events['team_id'] == comp_team_id])
                comparisons.append({
                    "team_id": comp_team_id,
                    "team_name": None,
                    "style": comp_style
                })
        return {
            "team": {
                "id": team_id,
                "name": None
            },
            "competition_id": competition_id,
            "season_id": season_id,
            "style_metrics": style_metrics,
            "team_style": team_style,
            "comparisons": comparisons,
            "total_teams_in_competition": len(league_teams)
        }
    except Exception as e:
        logger.error(f"Error getting team style comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/matchup-prediction")
async def get_matchup_prediction(
    team1_id: int = Query(..., description="First team ID"),
    team2_id: int = Query(..., description="Second team ID"),
    competition_id: Optional[int] = Query(None, description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get matchup prediction
    Returns probability distributions for match outcomes and key metrics
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        matches_df = fdm.get_matches_for_team(competition_id, season_id, team1_id)
        if matches_df is None or matches_df.empty:
            return {"error": "No matches found for team1 in this competition/season."}
        all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
        if not all_events:
            return {"error": "No data found for competition/season/team."}
        events = pd.concat(all_events)
        # Get team names from matches_df
        def get_team_name(team_id):
            row = matches_df[(matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)]
            if not row.empty:
                if row.iloc[0]['home_team_id'] == team_id:
                    return row.iloc[0]['home_team_name']
                else:
                    return row.iloc[0]['away_team_name']
            return None
        team1_name = get_team_name(team1_id)
        team2_name = get_team_name(team2_id)
        # Calculate averages and stddevs for both teams
        def team_stats(tid):
            te = events[events['team_id'] == tid]
            matches = te['match_id'].nunique()
            possession = 100 * len(te[te['possession_team'] == tid]) / max(1, len(te))
            shots = len(te[te['type_name'] == 'Shot']) / max(1, matches)
            xg = te['shot_statsbomb_xg'].sum() / max(1, matches) if 'shot_statsbomb_xg' in te else 0
            return possession, shots, xg
        team1_poss, team1_shots, team1_xg = team_stats(team1_id)
        team2_poss, team2_shots, team2_xg = team_stats(team2_id)
        # Use means and stddevs for prediction
        metric_distributions = {
            "possession": {
                "team1_mean": team1_poss,
                "team1_stddev": 5.0,
                "team2_mean": team2_poss,
                "team2_stddev": 5.0
            },
            "shots": {
                "team1_mean": team1_shots,
                "team1_stddev": 2.0,
                "team2_mean": team2_shots,
                "team2_stddev": 2.0
            },
            "xg": {
                "team1_mean": team1_xg,
                "team1_stddev": 0.5,
                "team2_mean": team2_xg,
                "team2_stddev": 0.5
            }
        }
        # Simple win probability based on xG
        total_xg = team1_xg + team2_xg
        win_probability = {
            "team1": team1_xg / total_xg if total_xg > 0 else 0.5,
            "team2": team2_xg / total_xg if total_xg > 0 else 0.5,
            "draw": 0.2
        }
        # Normalize
        total = sum(win_probability.values())
        for k in win_probability:
            win_probability[k] /= total
        # Score probabilities (simple Poisson model)
        from scipy.stats import poisson
        score_probabilities = []
        for team1_score in range(6):
            for team2_score in range(6):
                p1 = poisson.pmf(team1_score, team1_xg)
                p2 = poisson.pmf(team2_score, team2_xg)
                prob = p1 * p2
                score_probabilities.append({
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "probability": prob
                })
        total_score_probability = sum(sp["probability"] for sp in score_probabilities)
        for sp in score_probabilities:
            sp["probability"] /= total_score_probability
        score_probabilities.sort(key=lambda x: x["probability"], reverse=True)
        score_probabilities = score_probabilities[:10]
        # Key matchup factors (placeholder)
        key_matchup_factors = [
            {"factor": "xG difference", "advantage": "team1" if team1_xg > team2_xg else "team2", "importance": 8, "description": "Team with higher xG per match has an edge."}
        ]
        return {
            "team1": {"id": team1_id, "name": team1_name},
            "team2": {"id": team2_id, "name": team2_name},
            "win_probability": win_probability,
            "score_probabilities": score_probabilities,
            "metric_distributions": metric_distributions,
            "key_matchup_factors": key_matchup_factors,
            "competition_id": competition_id
        }
    except Exception as e:
        logger.error(f"Error getting matchup prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/league-benchmarks")
async def get_league_benchmarks(
    team_id: int = Query(..., description="Team ID to analyze"),
    competition_id: int = Query(..., description="Competition ID"),
    season_id: Optional[int] = Query(None, description="Season ID"),
    metrics: Optional[List[str]] = Query(None, description="Specific metrics to include"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get league benchmarks for a team
    
    Returns team metrics compared to league benchmarks (min, max, avg, percentiles)
    """
    try:
        if not competition_id or not season_id:
            return {"error": "competition_id and season_id are required."}
        matches_df = fdm.get_matches_for_team(competition_id, season_id, team_id)
        if matches_df is None or matches_df.empty:
            return {"error": "No matches found for team in this competition/season."}
        all_events = [fdm.get_events(match['match_id']) for _, match in matches_df.iterrows()]
        if not all_events:
            return {"error": "No data found for competition/season/team."}
        events = pd.concat(all_events)
        # Get team name
        team_row = matches_df[(matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)]
        if not team_row.empty:
            if team_row.iloc[0]['home_team_id'] == team_id:
                team_name = team_row.iloc[0]['home_team_name']
            else:
                team_name = team_row.iloc[0]['away_team_name']
        else:
            team_name = f"Team {team_id}"
        # Default metrics if none specified
        if not metrics:
            metrics = [
                "goals_per_game", "xg_per_game", "shots_per_game", "possession", 
                "pass_accuracy", "ppda", "challenge_success_rate", "xg_against_per_game"
            ]
        # Prepare per-team stats
        team_ids = events['team_id'].unique()
        team_stats = {}
        for tid in team_ids:
            te = events[events['team_id'] == tid]
            matches_played = te['match_id'].nunique()
            goals = len(te[(te['type_name'] == 'Shot') & (te['shot_outcome'] == 'Goal')]) / max(1, matches_played)
            xg = te['shot_statsbomb_xg'].sum() / max(1, matches_played) if 'shot_statsbomb_xg' in te else 0
            shots = len(te[te['type_name'] == 'Shot']) / max(1, matches_played)
            passes = te[te['type_name'] == 'Pass']
            completed_passes = passes[passes['pass_outcome'].isna()]
            pass_accuracy = len(completed_passes) / max(1, len(passes)) * 100 if len(passes) > 0 else 0
            possessions = len(events[events['possession_team_id'] == tid]['possession'].unique())
            total_possessions = len(events['possession'].unique())
            possession = possessions / max(1, total_possessions) * 100 if total_possessions > 0 else 0
            # PPDA: passes allowed per defensive action in opposition half
            opp_passes_own_half = events[(events['team_id'] != tid) & (events['type_name'] == 'Pass') & (events['location'].apply(lambda x: isinstance(x, list) and len(x) >= 2 and x[0] < 60))]
            def_actions_opp_half = te[(te['type_name'].isin(['Pressure', 'Duel', 'Interception'])) & (te['location'].apply(lambda x: isinstance(x, list) and len(x) >= 2 and x[0] >= 60))]
            ppda = len(opp_passes_own_half) / max(1, len(def_actions_opp_half))
            # Challenge success rate: duels won / duels
            duels = te[te['type_name'] == 'Duel']
            duels_won = duels[duels.get('duel_outcome', None) == 'Won'] if 'duel_outcome' in duels else duels[[]]
            challenge_success_rate = len(duels_won) / max(1, len(duels)) * 100 if len(duels) > 0 else 0
            # xG against per game
            opp_shots = events[(events['team_id'] != tid) & (events['type_name'] == 'Shot')]
            xg_against = opp_shots['shot_statsbomb_xg'].sum() / max(1, matches_played) if 'shot_statsbomb_xg' in opp_shots else 0
            team_stats[tid] = {
                "goals_per_game": goals,
                "xg_per_game": xg,
                "shots_per_game": shots,
                "possession": possession,
                "pass_accuracy": pass_accuracy,
                "ppda": ppda,
                "challenge_success_rate": challenge_success_rate,
                "xg_against_per_game": xg_against
            }
        # Aggregate league values for each metric
        benchmark_data = {}
        for metric in metrics:
            values = [team_stats[tid][metric] for tid in team_stats]
            team_val = team_stats.get(team_id, {}).get(metric, None)
            if team_val is None:
                continue
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            avg_val = float(np.mean(values))
            median_val = float(np.median(values))
            sorted_values = sorted(values)
            rank = sorted_values.index(team_val) + 1
            percentile = float(np.sum(np.array(values) < team_val)) / len(values) * 100 if len(values) > 0 else 0
            benchmark_data[metric] = {
                "team_value": team_val,
                "league_min": min_val,
                "league_max": max_val,
                "league_avg": avg_val,
                "league_median": median_val,
                "percentile": percentile,
                "rank": rank,
                "total_teams": len(values)
            }
        return {
            "team": {
                "id": team_id,
                "name": team_name
            },
            "competition_id": competition_id,
            "season_id": season_id,
            "metrics": metrics,
            "benchmarks": benchmark_data
        }
    except Exception as e:
        logger.error(f"Error getting league benchmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
