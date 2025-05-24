from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging
from app.util.football_data_manager import FootballDataManager

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
        # Generate team names (in real implementation would get from DB)
        team1_name = f"Team {team1_id}"
        team2_name = f"Team {team2_id}"
        
        # Generate historical match data
        historical_matches = []
        for i in range(1, last_n + 1):
            match_id = 1000 + i
            
            # Alternate winners for sample data
            if i % 3 == 0:
                winner = "draw"
                score = f"{i % 2 + 1}-{i % 2 + 1}"
            elif i % 2 == 0:
                winner = "team1"
                score = f"{i % 3 + 2}-{i % 2}"
            else:
                winner = "team2"
                score = f"{i % 2}-{i % 3 + 2}"
            
            # Determine who was home vs away
            home_team = team1_name if i % 2 == 0 else team2_name
            away_team = team2_name if i % 2 == 0 else team1_name
            
            # Create match data
            historical_matches.append({
                "match_id": match_id,
                "date": f"2023-{12 - i}-{15 - i}",  # Fabricated dates
                "competition": "League A" if i % 2 == 0 else "Cup B",
                "home_team": home_team,
                "away_team": away_team,
                "score": score,
                "winner": winner,
                "team1_xg": float(score.split("-")[0]) - 0.3 + (i * 0.1),
                "team2_xg": float(score.split("-")[1]) - 0.2 + (i * 0.15)
            })
        
        # Generate summary stats
        team1_wins = sum(1 for m in historical_matches if m["winner"] == "team1")
        team2_wins = sum(1 for m in historical_matches if m["winner"] == "team2")
        draws = sum(1 for m in historical_matches if m["winner"] == "draw")
        
        team1_goals = sum(float(m["score"].split("-")[0]) for m in historical_matches)
        team2_goals = sum(float(m["score"].split("-")[1]) for m in historical_matches)
        
        team1_xg = sum(m["team1_xg"] for m in historical_matches)
        team2_xg = sum(m["team2_xg"] for m in historical_matches)
        
        # Generate comparative metrics
        comparative_metrics = {
            "possession": {
                "team1": 52.3,
                "team2": 47.7
            },
            "shots": {
                "team1": 12.5,
                "team2": 10.8
            },
            "shots_on_target": {
                "team1": 4.8,
                "team2": 3.9
            },
            "passes": {
                "team1": 450,
                "team2": 410
            },
            "pass_accuracy": {
                "team1": 84.2,
                "team2": 81.5
            },
            "ppda": {
                "team1": 9.3,
                "team2": 10.5
            },
            "chances_created": {
                "team1": 2.3,
                "team2": 1.8
            }
        }
        
        return {
            "team1": {
                "id": team1_id,
                "name": team1_name
            },
            "team2": {
                "id": team2_id,
                "name": team2_name
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
            "comparative_metrics": comparative_metrics,
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
        # Generate team name (in real implementation would get from DB)
        team_name = f"Team {team_id}"
        
        # Define style metrics to compare
        style_metrics = [
            "Possession (%)",
            "Build-up Speed",
            "Width",
            "Directness",
            "Pressing Intensity",
            "Defensive Line Height",
            "Counter-Attack Frequency",
            "Set Piece Reliance"
        ]
        
        # Generate team style values
        import random
        random.seed(team_id)
        team_style = {}
        for metric in style_metrics:
            team_style[metric] = random.uniform(1, 10)
        
        # Generate comparison data
        comparisons = []
        
        # League average (always included)
        league_avg = {}
        random.seed(competition_id)
        for metric in style_metrics:
            league_avg[metric] = random.uniform(4, 7)  # League averages tend to be middle values
        
        comparisons.append({
            "team_id": None,
            "team_name": "League Average",
            "style": league_avg
        })
        
        # Add specific team comparisons if requested
        if compare_to:
            for comp_team_id in compare_to:
                comp_team_name = f"Team {comp_team_id}"
                comp_style = {}
                random.seed(comp_team_id)
                for metric in style_metrics:
                    comp_style[metric] = random.uniform(1, 10)
                
                comparisons.append({
                    "team_id": comp_team_id,
                    "team_name": comp_team_name,
                    "style": comp_style
                })
        
        # Generate supporting statistics
        supporting_stats = {
            "Possession (%)": {
                "value": 54.3,
                "rank": 5,
                "percentile": 75
            },
            "Build-up Speed": {
                "direct_attacks_per_game": 2.3,
                "avg_pass_sequence_duration": 12.5,
                "rank": 12,
                "percentile": 40
            },
            "Width": {
                "width_of_attacks": 42.1,
                "crosses_per_game": 18.7,
                "rank": 3,
                "percentile": 85
            },
            "Directness": {
                "forward_passes_pct": 38.2,
                "progressive_passes_per_game": 65.3,
                "rank": 8,
                "percentile": 60
            },
            "Pressing Intensity": {
                "ppda": 8.7,
                "pressures_per_game": 145.2,
                "rank": 4,
                "percentile": 80
            },
            "Defensive Line Height": {
                "avg_defensive_line_distance": 42.1,
                "offsides_won_per_game": 2.8,
                "rank": 6,
                "percentile": 70
            },
            "Counter-Attack Frequency": {
                "counter_attacks_per_game": 4.2,
                "counter_attack_shots": 1.7,
                "rank": 10,
                "percentile": 50
            },
            "Set Piece Reliance": {
                "set_piece_goals_pct": 28.5,
                "set_piece_xg_per_game": 0.45,
                "rank": 15,
                "percentile": 25
            }
        }
        
        return {
            "team": {
                "id": team_id,
                "name": team_name
            },
            "competition_id": competition_id,
            "season_id": season_id,
            "style_metrics": style_metrics,
            "team_style": team_style,
            "comparisons": comparisons,
            "supporting_stats": supporting_stats,
            "total_teams_in_competition": 20
        }
    except Exception as e:
        logger.error(f"Error getting team style comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/matchup-prediction")
async def get_matchup_prediction(
    team1_id: int = Query(..., description="First team ID"),
    team2_id: int = Query(..., description="Second team ID"),
    competition_id: Optional[int] = Query(None, description="Competition ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    Get matchup prediction
    
    Returns probability distributions for match outcomes and key metrics
    """
    try:
        # Generate team names (in real implementation would get from DB)
        team1_name = f"Team {team1_id}"
        team2_name = f"Team {team2_id}"
        
        # Generate win probability distribution
        win_probability = {
            "team1": 0.4 + (team1_id % 10) * 0.01,
            "team2": 0.3 + (team2_id % 10) * 0.01,
            "draw": 0.3 - ((team1_id + team2_id) % 10) * 0.01
        }
        
        # Normalize to ensure probabilities sum to 1
        total = sum(win_probability.values())
        for k in win_probability:
            win_probability[k] /= total
        
        # Generate score probability distribution
        score_probabilities = []
        for team1_score in range(6):
            for team2_score in range(6):
                # Generate probability (more likely for lower scores)
                probability = 0.1 / ((team1_score + 1) * (team2_score + 1))
                
                # Adjust based on win probability
                if team1_score > team2_score:
                    probability *= win_probability["team1"] * 2
                elif team2_score > team1_score:
                    probability *= win_probability["team2"] * 2
                else:
                    probability *= win_probability["draw"] * 3
                
                score_probabilities.append({
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "probability": probability
                })
        
        # Normalize score probabilities
        total_score_probability = sum(sp["probability"] for sp in score_probabilities)
        for sp in score_probabilities:
            sp["probability"] /= total_score_probability
        
        # Sort by probability descending
        score_probabilities.sort(key=lambda x: x["probability"], reverse=True)
        
        # Keep only top 10 most likely scores
        score_probabilities = score_probabilities[:10]
        
        # Generate expected metric distributions
        metric_distributions = {
            "possession": {
                "team1_mean": 52.4,
                "team1_stddev": 4.3,
                "team2_mean": 47.6,
                "team2_stddev": 4.3
            },
            "shots": {
                "team1_mean": 13.2,
                "team1_stddev": 3.1,
                "team2_mean": 11.5,
                "team2_stddev": 2.8
            },
            "xg": {
                "team1_mean": 1.45,
                "team1_stddev": 0.6,
                "team2_mean": 1.25,
                "team2_stddev": 0.5
            },
            "big_chances": {
                "team1_mean": 2.3,
                "team1_stddev": 1.2,
                "team2_mean": 1.9,
                "team2_stddev": 1.1
            }
        }
        
        # Generate key matchup factors
        key_matchup_factors = [
            {
                "factor": "Team 1's high press vs Team 2's build-up",
                "advantage": "team1",
                "importance": 8,
                "description": "Team 1's pressing system is likely to disrupt Team 2's build-up play"
            },
            {
                "factor": "Team 2's counter-attacking vs Team 1's high line",
                "advantage": "team2",
                "importance": 7,
                "description": "Team 2's speed in transition could exploit Team 1's high defensive line"
            },
            {
                "factor": "Set pieces",
                "advantage": "team1",
                "importance": 6,
                "description": "Team 1 has a significant height advantage and better set-piece record"
            },
            {
                "factor": "Midfield control",
                "advantage": "team2",
                "importance": 9,
                "description": "Team 2's midfield trio should dominate possession in central areas"
            }
        ]
        
        return {
            "team1": {
                "id": team1_id,
                "name": team1_name
            },
            "team2": {
                "id": team2_id,
                "name": team2_name
            },
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
        # Generate team name (in real implementation would get from DB)
        team_name = f"Team {team_id}"
        
        # Default metrics if none specified
        if not metrics:
            metrics = [
                "goals_per_game", "xg_per_game", "shots_per_game", "possession", 
                "pass_accuracy", "ppda", "challenge_success_rate", "xg_against_per_game"
            ]
        
        # Generate benchmark data for each metric
        benchmark_data = {}
        import random
        random.seed(team_id + competition_id)
        
        for metric in metrics:
            # Generate reasonable ranges based on the metric
            if metric == "goals_per_game":
                min_val, max_val = 0.5, 3.0
                team_val = 1.0 + random.random() * 1.5
            elif metric == "xg_per_game":
                min_val, max_val = 0.7, 2.5
                team_val = 1.1 + random.random() * 1.2
            elif metric == "shots_per_game":
                min_val, max_val = 8.0, 20.0
                team_val = 10.0 + random.random() * 8.0
            elif metric == "possession":
                min_val, max_val = 35.0, 65.0
                team_val = 45.0 + random.random() * 15.0
            elif metric == "pass_accuracy":
                min_val, max_val = 70.0, 90.0
                team_val = 75.0 + random.random() * 12.0
            elif metric == "ppda":
                min_val, max_val = 6.0, 16.0
                team_val = 8.0 + random.random() * 6.0
            elif metric == "challenge_success_rate":
                min_val, max_val = 40.0, 65.0
                team_val = 45.0 + random.random() * 15.0
            elif metric == "xg_against_per_game":
                min_val, max_val = 0.7, 2.2
                team_val = 0.9 + random.random() * 1.0
            else:
                min_val, max_val = 0.0, 10.0
                team_val = random.random() * 10.0
            
            # Calculate mock percentiles based on team value
            perc_position = (team_val - min_val) / (max_val - min_val)
            percentile = int(perc_position * 100)
            
            # Generate random values for other teams to create a distribution
            other_team_values = [min_val + random.random() * (max_val - min_val) for _ in range(19)]
            all_values = other_team_values + [team_val]
            all_values.sort()
            
            avg_val = sum(all_values) / len(all_values)
            median_val = all_values[len(all_values) // 2]
            
            benchmark_data[metric] = {
                "team_value": team_val,
                "league_min": min_val,
                "league_max": max_val,
                "league_avg": avg_val,
                "league_median": median_val,
                "percentile": percentile,
                "rank": all_values.index(team_val) + 1,
                "total_teams": len(all_values)
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
