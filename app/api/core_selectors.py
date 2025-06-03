from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from app.util.football_data_manager import FootballDataManager
import pandas as pd

router = APIRouter()

def get_data_manager():
    """Dependency to get FootballDataManager instance"""
    return FootballDataManager()

@router.get("/competitions")
async def list_competitions(
    only_with_360: bool = Query(True, description="Only competitions with 360 data"),
    exclude_women: bool = Query(True, description="Exclude women's competitions"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List available competitions
    """
    try:
        df = fdm.get_competitions(only_with_360=only_with_360, exclude_women=exclude_women)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing competitions: {str(e)}")

@router.get("/seasons")
async def list_seasons(
    competition_id: int = Query(..., description="Competition ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List seasons for a competition
    """
    try:
        df = fdm.get_competitions()
        comp = df[df['competition_id'] == competition_id]
        if comp.empty:
            return []
        # Return all unique seasons for this competition
        return comp[['season_id', 'season_name']].drop_duplicates().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing seasons: {str(e)}")

@router.get("/teams")
async def list_teams(
    competition_id: int = Query(None, description="Competition ID"),
    season_id: int = Query(None, description="Season ID"),
    team_id: int = Query(None, description="Team ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List teams based on provided parameters:
    - If team_id is provided, return the specific team
    - If only competition_id is provided, filter teams by competition
    - If competition_id and season_id are provided, filter teams by both

    At least one of the parameters (competition_id, season_id, or team_id) is required.
    """
    try:
        # Check that at least one parameter is provided
        if competition_id is None and season_id is None and team_id is None:
            raise HTTPException(
                status_code=400,
                detail="At least one parameter (competition_id, season_id, or team_id) is required"
            )

        # Case 1: If team_id is provided, return that specific team
        if team_id is not None:
            # We still need competition_id to get data context
            if competition_id is None:
                raise HTTPException(status_code=400, detail="competition_id is required when team_id is provided")

            # If season_id is provided, use it to get more specific data
            if season_id is not None:
                matches = fdm.get_matches(competition_id, season_id)
                filtered = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
                if filtered.empty:
                    return []

                row = filtered.iloc[0]
                name = row['home_team_name'] if row['home_team_id'] == team_id else row['away_team_name']
                return [{"team_id": team_id, "team_name": name}]
            else:
                # Without season_id, we need to fetch from competitions
                competitions = fdm.get_competitions()
                seasons = competitions[competitions['competition_id'] == competition_id]['season_id'].unique()

                if len(seasons) == 0:
                    return []

                # Use the most recent season to find the team
                season_id = seasons[0]  # Assuming seasons are sorted newest first
                matches = fdm.get_matches(competition_id, season_id)
                filtered = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]

                if filtered.empty:
                    return []

                row = filtered.iloc[0]
                name = row['home_team_name'] if row['home_team_id'] == team_id else row['away_team_name']
                return [{"team_id": team_id, "team_name": name}]

        # Case 2: If only competition_id is provided
        elif competition_id is not None and season_id is None:
            competitions = fdm.get_competitions()
            seasons = competitions[competitions['competition_id'] == competition_id]['season_id'].unique()

            if len(seasons) == 0:
                return []

            # Use the most recent season
            season_id = seasons[0]  # Assuming seasons are sorted newest first
            matches = fdm.get_matches(competition_id, season_id)

            teams = set(matches['home_team_id']).union(set(matches['away_team_id']))
            team_info = []
            for tid in teams:
                row = matches[(matches['home_team_id'] == tid) | (matches['away_team_id'] == tid)].iloc[0]
                name = row['home_team_name'] if row['home_team_id'] == tid else row['away_team_name']
                team_info.append({"team_id": tid, "team_name": name})

            return team_info

        # Case 3: If both competition_id and season_id are provided (original implementation)
        elif competition_id is not None and season_id is not None:
            matches = fdm.get_matches(competition_id, season_id)
            teams = set(matches['home_team_id']).union(set(matches['away_team_id']))
            team_info = []
            for tid in teams:
                row = matches[(matches['home_team_id'] == tid) | (matches['away_team_id'] == tid)].iloc[0]
                name = row['home_team_name'] if row['home_team_id'] == tid else row['away_team_name']
                team_info.append({"team_id": tid, "team_name": name})
            return team_info

        # Case 4: If only season_id is provided
        elif season_id is not None and competition_id is None:
            competitions = fdm.get_competitions()
            comps_with_season = competitions[competitions['season_id'] == season_id]

            if comps_with_season.empty:
                return []

            # Get first competition with this season
            competition_id = comps_with_season.iloc[0]['competition_id']
            matches = fdm.get_matches(competition_id, season_id)

            teams = set(matches['home_team_id']).union(set(matches['away_team_id']))
            team_info = []
            for tid in teams:
                row = matches[(matches['home_team_id'] == tid) | (matches['away_team_id'] == tid)].iloc[0]
                name = row['home_team_name'] if row['home_team_id'] == tid else row['away_team_name']
                team_info.append({"team_id": tid, "team_name": name})

            return team_info
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing teams: {str(e)}")

@router.get("/matches")
async def list_matches(
    competition_id: int = Query(None, description="Competition ID"),
    season_id: int = Query(None, description="Season ID"),
    team_id: int = Query(None, description="Filter by team ID"),
    match_id: int = Query(None, description="Specific match ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List matches based on provided parameters:
    - If match_id is provided, return that specific match
    - If team_id is provided with competition_id/season_id, filter matches by team
    - If only competition_id and season_id are provided, return all matches for that season

    At least one parameter is required.
    """
    try:
        # Check that at least one parameter is provided
        if competition_id is None and season_id is None and team_id is None and match_id is None:
            raise HTTPException(
                status_code=400,
                detail="At least one parameter (competition_id, season_id, team_id, or match_id) is required"
            )

        # Case 1: If match_id is provided, return that specific match
        if match_id is not None:
            # Check if we have competition_id and season_id to locate the match
            if competition_id is not None and season_id is not None:
                matches = fdm.get_matches(competition_id, season_id)
                match = matches[matches['match_id'] == match_id]
                if not match.empty:
                    return match.to_dict(orient="records")

            # If we don't have competition_id and season_id, search across all competitions
            competitions = fdm.get_competitions()
            for _, comp in competitions.iterrows():
                try:
                    matches = fdm.get_matches(comp['competition_id'], comp['season_id'])
                    match = matches[matches['match_id'] == match_id]
                    if not match.empty:
                        return match.to_dict(orient="records")
                except:
                    continue

            # If we reach here, match not found
            return []

        # Case 2: If team_id is provided without match_id
        elif team_id is not None:
            # If we don't have competition_id or season_id, search across all competitions
            if competition_id is None and season_id is None:
                competitions = fdm.get_competitions()
                all_team_matches = []

                # Search for the team in all competitions
                for _, comp in competitions.iterrows():
                    try:
                        comp_id = comp['competition_id']
                        sea_id = comp['season_id']
                        matches = fdm.get_matches(comp_id, sea_id)
                        team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]

                        if not team_matches.empty:
                            all_team_matches.append(team_matches)
                    except Exception as e:
                        # Log or handle specific exceptions if needed
                        continue

                # Combine results from all competitions where the team was found
                if all_team_matches:
                    combined_matches = pd.concat(all_team_matches)
                    return combined_matches.to_dict(orient="records")
                return []

            # If we have both competition_id and season_id
            if competition_id is not None and season_id is not None:
                matches = fdm.get_matches(competition_id, season_id)
                team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
                return team_matches.to_dict(orient="records")

            # If we only have competition_id
            elif competition_id is not None:
                competitions = fdm.get_competitions()
                comp_seasons = competitions[competitions['competition_id'] == competition_id]

                if comp_seasons.empty:
                    return []

                # Get most recent season
                season_id = comp_seasons.iloc[0]['season_id']
                matches = fdm.get_matches(competition_id, season_id)
                team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
                return team_matches.to_dict(orient="records")

            # If we only have season_id
            elif season_id is not None:
                competitions = fdm.get_competitions()
                season_comps = competitions[competitions['season_id'] == season_id]

                if season_comps.empty:
                    return []

                # Get first competition with this season
                competition_id = season_comps.iloc[0]['competition_id']
                matches = fdm.get_matches(competition_id, season_id)
                team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
                return team_matches.to_dict(orient="records")
            return None

        # Case 3: If we have competition_id and season_id but no team_id or match_id
        elif competition_id is not None and season_id is not None:
            matches = fdm.get_matches(competition_id, season_id)
            return matches.to_dict(orient="records")

        # Case 4: If we only have competition_id
        elif competition_id is not None:
            competitions = fdm.get_competitions()
            comp_seasons = competitions[competitions['competition_id'] == competition_id]

            if comp_seasons.empty:
                return []

            # Get most recent season
            season_id = comp_seasons.iloc[0]['season_id']
            matches = fdm.get_matches(competition_id, season_id)
            return matches.to_dict(orient="records")

        # Case 5: If we only have season_id
        elif season_id is not None:
            competitions = fdm.get_competitions()
            season_comps = competitions[competitions['season_id'] == season_id]

            if season_comps.empty:
                return []

            # Get first competition with this season
            competition_id = season_comps.iloc[0]['competition_id']
            matches = fdm.get_matches(competition_id, season_id)
            return matches.to_dict(orient="records")
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing matches: {str(e)}")

@router.get("/players")
async def list_players(
    competition_id: int = Query(None, description="Competition ID"),
    season_id: int = Query(None, description="Season ID"),
    team_id: int = Query(None, description="Filter by team ID"),
    player_id: int = Query(None, description="Specific player ID"),
    match_id: int = Query(None, description="Filter by match ID"),
    fdm: FootballDataManager = Depends(get_data_manager)
):
    """
    List players based on provided parameters:
    - If player_id is provided, return that specific player
    - If match_id is provided, return players from that match
    - If team_id is provided, return players from that team
    - If competition_id and season_id are provided, return all players

    At least one parameter is required.
    """
    try:
        # Check that at least one parameter is provided
        if competition_id is None and season_id is None and team_id is None and player_id is None and match_id is None:
            raise HTTPException(
                status_code=400,
                detail="At least one parameter (competition_id, season_id, team_id, player_id, or match_id) is required"
            )

        # Case 1: If player_id is provided, find that specific player
        if player_id is not None:
            # We need context to find player data - require at least one other parameter
            if competition_id is None and season_id is None and team_id is None and match_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="At least one additional parameter is required when filtering by player_id"
                )

            # If we have match_id, that's the most direct way to get player data
            if match_id is not None:
                events = fdm.get_events(match_id)
                if events is None or events.empty:
                    return []

            # If we have team_id and context (competition_id and season_id)
            elif team_id is not None and competition_id is not None and season_id is not None:
                events = fdm.get_events_for_team(competition_id, season_id, team_id)
                if events is None or events.empty:
                    return []

            # If we have competition_id and season_id but no team_id
            elif competition_id is not None and season_id is not None:
                matches = fdm.get_matches(competition_id, season_id)
                if matches.empty:
                    return []
                # Get first match to check for player
                match_id = matches.iloc[0]['match_id']
                events = fdm.get_events(match_id)
                if events is None or events.empty:
                    return []

            # Extract player data from events
            if 'tactics_lineup_json' in events.columns and events['tactics_lineup_json'].notnull().any():
                import json
                lineups = events['tactics_lineup_json'].dropna().unique()
                for lineup_json in lineups:
                    lineup = json.loads(lineup_json)
                    for entry in lineup:
                        player = entry.get('player', {})
                        if player.get('id') == player_id:
                            position = entry.get('position', {})
                            return [{
                                'player_id': player.get('id'),
                                'player_name': player.get('name'),
                                'position_id': position.get('id'),
                                'position_name': position.get('name'),
                                'jersey_number': entry.get('jersey_number')
                            }]
            return []

        # Case 2: If match_id is provided, return players from that match
        elif match_id is not None:
            events = fdm.get_events(match_id)
            if events is None or events.empty:
                return []

            if 'tactics_lineup_json' in events.columns and events['tactics_lineup_json'].notnull().any():
                import json
                lineups = events['tactics_lineup_json'].dropna().unique()
                players = []
                for lineup_json in lineups:
                    lineup = json.loads(lineup_json)
                    for entry in lineup:
                        player = entry.get('player', {})
                        position = entry.get('position', {})
                        players.append({
                            'player_id': player.get('id'),
                            'player_name': player.get('name'),
                            'position_id': position.get('id'),
                            'position_name': position.get('name'),
                            'jersey_number': entry.get('jersey_number')
                        })
                return players
            return []

        # Case 3: If team_id is provided (without player_id or match_id)
        elif team_id is not None:
            # If competition_id and season_id are not provided, search across all competitions
            if competition_id is None and season_id is None:
                competitions = fdm.get_competitions()
                all_players = set()
                player_info = []

                # Search for the team in all competitions
                for _, comp in competitions.iterrows():
                    try:
                        comp_id = comp['competition_id']
                        sea_id = comp['season_id']

                        # Try to get events for this team in this competition/season
                        events = fdm.get_events_for_team(comp_id, sea_id, team_id)

                        if events is None or events.empty or 'tactics_lineup_json' not in events.columns:
                            continue

                        import json
                        lineups = events['tactics_lineup_json'].dropna().unique()

                        for lineup_json in lineups:
                            lineup = json.loads(lineup_json)
                            for entry in lineup:
                                player = entry.get('player', {})
                                player_id = player.get('id')

                                if player_id not in all_players:
                                    all_players.add(player_id)
                                    position = entry.get('position', {})
                                    player_info.append({
                                        'player_id': player_id,
                                        'player_name': player.get('name'),
                                        'position_id': position.get('id'),
                                        'position_name': position.get('name'),
                                        'jersey_number': entry.get('jersey_number')
                                    })
                    except Exception as e:
                        # Skip any competitions that cause errors
                        continue

                return player_info

            # Otherwise get events for the team with competition_id and season_id
            elif competition_id is not None and season_id is not None:
                events = fdm.get_events_for_team(competition_id, season_id, team_id)
                if events is None or events.empty:
                    return []

                if 'tactics_lineup_json' in events.columns and events['tactics_lineup_json'].notnull().any():
                    import json
                    lineups = events['tactics_lineup_json'].dropna().unique()
                    players = []
                    for lineup_json in lineups:
                        lineup = json.loads(lineup_json)
                        for entry in lineup:
                            player = entry.get('player', {})
                            position = entry.get('position', {})
                            players.append({
                                'player_id': player.get('id'),
                                'player_name': player.get('name'),
                                'position_id': position.get('id'),
                                'position_name': position.get('name'),
                                'jersey_number': entry.get('jersey_number')
                            })
                    return players
                return []
            return None

        # Case 4: If competition_id and season_id are provided (without team_id, player_id, or match_id)
        elif competition_id is not None and season_id is not None:
            matches = fdm.get_matches(competition_id, season_id)
            if matches.empty:
                return []

            # Collect players from all matches - this could be resource intensive
            # Limit to first few matches for performance
            all_players = set()
            player_info = []

            for i, match in matches.iterrows():
                if i > 5:  # Limit to first 5 matches for performance
                    break

                try:
                    events = fdm.get_events(match['match_id'])
                    if events is None or events.empty or 'tactics_lineup_json' not in events.columns:
                        continue

                    lineups = events['tactics_lineup_json'].dropna().unique()
                    import json

                    for lineup_json in lineups:
                        lineup = json.loads(lineup_json)
                        for entry in lineup:
                            player = entry.get('player', {})
                            player_id = player.get('id')

                            if player_id not in all_players:
                                all_players.add(player_id)
                                position = entry.get('position', {})
                                player_info.append({
                                    'player_id': player_id,
                                    'player_name': player.get('name'),
                                    'position_id': position.get('id'),
                                    'position_name': position.get('name'),
                                    'jersey_number': entry.get('jersey_number')
                                })
                except:
                    continue

            return player_info

        # Case 5: If only competition_id or only season_id is provided
        elif competition_id is not None or season_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Both competition_id and season_id are required when not filtering by team_id, player_id, or match_id"
            )
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing players: {str(e)}")
