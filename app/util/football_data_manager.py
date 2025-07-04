# app/util/football_data_manager.py

import os
import pickle
import datetime
from pathlib import Path
import pandas as pd
from statsbombpy import sb
from typing import Dict, List, Tuple, Optional, Union
import requests
import json


class FootballDataManager:
    """
    Manages football data across multiple competitions with in-memory caching
    and serialization capabilities.
    """
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the FootballDataManager.
        
        Args:
            cache_dir: Directory for serialized data storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory cache structure
        self.competitions_cache = None
        self.matches_cache = {}  # {competition_id_season_id: matches_df}
        self.events_cache = {}   # {match_id: events_df}
        self.frames_cache = {}   # {match_id: freeze_frame_df}
        
    def _flatten_competitions_json(self, competitions_json: list) -> pd.DataFrame:
        """
        Flatten the raw competitions JSON into a DataFrame with all relevant fields.
        """
        records = []
        for key, c in competitions_json.items():
            record = {
                'competition_id': c.get('competition_id'),
                'season_id': c.get('season_id'),
                'competition_name': c.get('competition_name'),
                'competition_gender': c.get('competition_gender'),
                'country_name': c.get('country_name'),
                'competition_youth': c.get('competition_youth'),
                'competition_international': c.get('competition_international'),
                'season_name': c.get('season_name'),
                'match_available_360': c.get('match_available_360'),
                'match_available': c.get('match_available'),
                # Add more fields as needed
            }
            records.append(record)
        return pd.DataFrame(records)

    def _flatten_matches_json(self, matches_json: list) -> pd.DataFrame:
        """
        Flatten the raw matches JSON into a DataFrame with all relevant fields.
        """
        records = []
        for key, m in matches_json.items():
            record = {
                'match_id': m.get('match_id'),
                'match_date': m.get('match_date'),
                'kick_off': m.get('kick_off'),
                'competition_id': m.get('competition', {}).get('competition_id') if isinstance(m.get('competition'), dict) else m.get('competition_id'),
                'competition_name': m.get('competition', {}).get('competition_name') if isinstance(m.get('competition'), dict) else m.get('competition_name'),
                'country_name': m.get('competition', {}).get('country_name') if isinstance(m.get('competition'), dict) else m.get('country_name'),
                'season_id': m.get('season', {}).get('season_id') if isinstance(m.get('season'), dict) else m.get('season_id'),
                'season_name': m.get('season', {}).get('season_name') if isinstance(m.get('season'), dict) else m.get('season_name'),
                'home_team_id': m.get('home_team', {}).get('home_team_id') if isinstance(m.get('home_team'), dict) else m.get('home_team_id'),
                'home_team_name': m.get('home_team', {}).get('home_team_name') if isinstance(m.get('home_team'), dict) else m.get('home_team'),
                'home_team_gender': m.get('home_team', {}).get('home_team_gender') if isinstance(m.get('home_team'), dict) else None,
                'home_team_group': m.get('home_team', {}).get('home_team_group') if isinstance(m.get('home_team'), dict) else None,
                'home_team_country_id': m.get('home_team', {}).get('country', {}).get('id') if isinstance(m.get('home_team', {}).get('country'), dict) else None,
                'home_team_country_name': m.get('home_team', {}).get('country', {}).get('name') if isinstance(m.get('home_team', {}).get('country'), dict) else None,
                'home_manager_id': m.get('home_team', {}).get('managers', [{}])[0].get('id') if m.get('home_team', {}).get('managers') else None,
                'home_manager_name': m.get('home_team', {}).get('managers', [{}])[0].get('name') if m.get('home_team', {}).get('managers') else None,
                'away_team_id': m.get('away_team', {}).get('away_team_id') if isinstance(m.get('away_team'), dict) else m.get('away_team_id'),
                'away_team_name': m.get('away_team', {}).get('away_team_name') if isinstance(m.get('away_team'), dict) else m.get('away_team'),
                'away_team_gender': m.get('away_team', {}).get('away_team_gender') if isinstance(m.get('away_team'), dict) else None,
                'away_team_group': m.get('away_team', {}).get('away_team_group') if isinstance(m.get('away_team'), dict) else None,
                'away_team_country_id': m.get('away_team', {}).get('country', {}).get('id') if isinstance(m.get('away_team', {}).get('country'), dict) else None,
                'away_team_country_name': m.get('away_team', {}).get('country', {}).get('name') if isinstance(m.get('away_team', {}).get('country'), dict) else None,
                'away_manager_id': m.get('away_team', {}).get('managers', [{}])[0].get('id') if m.get('away_team', {}).get('managers') else None,
                'away_manager_name': m.get('away_team', {}).get('managers', [{}])[0].get('name') if m.get('away_team', {}).get('managers') else None,
                'home_score': m.get('home_score'),
                'away_score': m.get('away_score'),
                'match_status': m.get('match_status'),
                'match_status_360': m.get('match_status_360'),
                'last_updated': m.get('last_updated'),
                'last_updated_360': m.get('last_updated_360'),
                'data_version': m.get('metadata', {}).get('data_version') if isinstance(m.get('metadata'), dict) else None,
                'shot_fidelity_version': m.get('metadata', {}).get('shot_fidelity_version') if isinstance(m.get('metadata'), dict) else None,
                'xy_fidelity_version': m.get('metadata', {}).get('xy_fidelity_version') if isinstance(m.get('metadata'), dict) else None,
                'match_week': m.get('match_week'),
                'competition_stage_id': m.get('competition_stage', {}).get('id') if isinstance(m.get('competition_stage'), dict) else None,
                'competition_stage_name': m.get('competition_stage', {}).get('name') if isinstance(m.get('competition_stage'), dict) else None,
                'stadium_id': m.get('stadium', {}).get('id') if isinstance(m.get('stadium'), dict) else None,
                'stadium_name': m.get('stadium', {}).get('name') if isinstance(m.get('stadium'), dict) else None,
                'stadium_country_id': m.get('stadium', {}).get('country', {}).get('id') if isinstance(m.get('stadium', {}).get('country'), dict) else None,
                'stadium_country_name': m.get('stadium', {}).get('country', {}).get('name') if isinstance(m.get('stadium', {}).get('country'), dict) else None,
                'referee_id': m.get('referee', {}).get('id') if isinstance(m.get('referee'), dict) else None,
                'referee_name': m.get('referee', {}).get('name') if isinstance(m.get('referee'), dict) else None,
                'referee_country_id': m.get('referee', {}).get('country', {}).get('id') if isinstance(m.get('referee', {}).get('country'), dict) else None,
                'referee_country_name': m.get('referee', {}).get('country', {}).get('name') if isinstance(m.get('referee', {}).get('country'), dict) else None,
                # Add more fields as needed
            }
            records.append(record)
        return pd.DataFrame(records)

    def _flatten_events_json(self, events_json: list) -> pd.DataFrame:
        """
        Flatten the raw events JSON into a DataFrame with all relevant fields, including tactics and lineup.
        """
        records = []
        for key, e in events_json.items():
            record = {
                'id': e.get('id'),
                'index': e.get('index'),
                'period': e.get('period'),
                'timestamp': e.get('timestamp'),
                'minute': e.get('minute'),
                'second': e.get('second'),
                'type_id': e.get('type', {}).get('id') if isinstance(e.get('type'), dict) else None,
                'type_name': e.get('type', {}).get('name') if isinstance(e.get('type'), dict) else e.get('type'),
                'possession': e.get('possession'),
                'possession_team_id': e.get('possession_team', {}).get('id') if isinstance(e.get('possession_team'), dict) else None,
                'possession_team_name': e.get('possession_team', {}).get('name') if isinstance(e.get('possession_team'), dict) else e.get('possession_team'),
                'play_pattern_id': e.get('play_pattern', {}).get('id') if isinstance(e.get('play_pattern'), dict) else None,
                'play_pattern_name': e.get('play_pattern', {}).get('name') if isinstance(e.get('play_pattern'), dict) else e.get('play_pattern'),
                'team_id': e.get('team', {}).get('id') if isinstance(e.get('team'), dict) else None,
                'team_name': e.get('team', {}).get('name') if isinstance(e.get('team'), dict) else e.get('team'),
                'player_id': e.get('player', {}).get('id') if isinstance(e.get('player'), dict) else None,
                'player_name': e.get('player', {}).get('name') if isinstance(e.get('player'), dict) else e.get('player'),
                'location': e.get('location'),
                'duration': e.get('duration'),
                'match_id': e.get('match_id'),
                # Tactics fields
                'tactics_formation': e.get('tactics', {}).get('formation') if isinstance(e.get('tactics'), dict) else None,
                'tactics_lineup': e.get('tactics', {}).get('lineup') if isinstance(e.get('tactics'), dict) else None,
                # Add more fields as needed, including nested ones
            }
            # Optionally flatten more nested fields (e.g., pass, shot, etc.)
            if 'pass' in e:
                record['pass_end_location'] = e['pass'].get('end_location')
                record['pass_outcome'] = e['pass'].get('outcome', {}).get('name') if isinstance(e['pass'].get('outcome'), dict) else e['pass'].get('outcome')
            if 'shot' in e:
                record['shot_end_location'] = e['shot'].get('end_location')
                record['shot_outcome'] = e['shot'].get('outcome', {}).get('name') if isinstance(e['shot'].get('outcome'), dict) else e['shot'].get('outcome')
                record['shot_statsbomb_xg'] = e['shot'].get('statsbomb_xg')
            # Optionally flatten lineup (store as JSON string for DataFrame compatibility)
            if 'tactics' in e and isinstance(e['tactics'], dict) and 'lineup' in e['tactics']:
                import json as _json
                record['tactics_lineup_json'] = _json.dumps(e['tactics']['lineup'])
            records.append(record)
        df = pd.DataFrame(records)
        if 'id' in df.columns:
            df.set_index('id', inplace=True)
        return df

    def get_competitions(self, force_refresh: bool = False, 
                         only_with_360: bool = True, 
                         exclude_women: bool = True) -> pd.DataFrame:
        """
        Get available competitions, with optional filtering.
        """
        if self.competitions_cache is None or force_refresh:
            competitions_json = sb.competitions(fmt='json')
            competitions = self._flatten_competitions_json(competitions_json)
            # Apply filters if specified
            if only_with_360:
                competitions = competitions[competitions['match_available_360'].notna()]
            if exclude_women:
                competitions = competitions[competitions['competition_gender'] != 'women']
            self.competitions_cache = competitions
        if not isinstance(self.competitions_cache, pd.DataFrame):
            self.competitions_cache = pd.DataFrame(self.competitions_cache)
        return self.competitions_cache

    def get_matches(self, competition_id: int, season_id: int, 
                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Get matches for a specific competition and season.
        """
        cache_key = f"{competition_id}_{season_id}"
        if cache_key not in self.matches_cache or force_refresh:
            matches_json = sb.matches(competition_id=competition_id, season_id=season_id, fmt='json')
            matches = self._flatten_matches_json(matches_json)
            self.matches_cache[cache_key] = matches
        return self.matches_cache[cache_key]

    def get_events(self, match_id: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get events data for a specific match.
        """
        if match_id not in self.events_cache or force_refresh:
            events_json = sb.events(match_id=match_id, fmt='json')
            events_df = self._flatten_events_json(events_json)
            self.events_cache[match_id] = events_df
        return self.events_cache[match_id]
    
    def get_freeze_frames(self, match_id: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get 360 freeze frame data for a specific match.
        
        Args:
            match_id: The match ID
            force_refresh: If True, fetches from API even if cached
            
        Returns:
            DataFrame of freeze frame data or empty DataFrame if not available
        """
        if match_id not in self.frames_cache or force_refresh:
            try:
                frames = sb.frames(match_id=match_id)
                self.frames_cache[match_id] = frames
            except (requests.exceptions.HTTPError, json.JSONDecodeError) as e:
                # Handle 404 errors or JSON decoding errors
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    print(f"360 data not available for match {match_id}. Returning empty DataFrame.")
                else:
                    print(f"Error fetching 360 data for match {match_id}: {str(e)}. Returning empty DataFrame.")
                
                # Create empty DataFrame with expected structure
                self.frames_cache[match_id] = pd.DataFrame({
                    'id': [], 'visible_area': [], 'match_id': [],
                    'teammate': [], 'actor': [], 'keeper': [], 'location': []
                })
            
        return self.frames_cache[match_id]
    
    def _process_events(self, events_data) -> pd.DataFrame:
        """
        Process events data into a standardized DataFrame.
        
        Args:
            events_data: Raw events data from statsbombpy
            
        Returns:
            Processed DataFrame
        """
        # Define standard columns based on your current implementation
        columns = ['location', 'pass_end_location', 'shot_end_location', 
                  'player', 'team', 'type', 'possession', 'play_pattern',
                  'minute', 'second', 'period', 'timestamp', 'id',
                  'match_id', 'pass_outcome', 'shot_outcome', 'shot_statsbomb_xg',
                  'possession_team', 'position', 'shot_freeze_frame']
        
        # Create DataFrame based on input type
        if isinstance(events_data, dict):
            df = pd.DataFrame.from_dict(events_data, orient='index')
        else:
            df = pd.DataFrame(events_data)
        
        # Ensure all columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = None
        
        # Set index to event id
        if 'id' in df.columns:
            df.set_index('id', inplace=True)
        
        return df
    
    def serialize_data(self, data_type: str, identifier: str, data: pd.DataFrame) -> None:
        """
        Serialize data to disk.
        
        Args:
            data_type: Type of data ('competitions', 'matches', 'events', 'frames')
            identifier: Identifier for the data (e.g., match_id or competition_id)
            data: DataFrame to serialize
        """
        # Create directory if it doesn't exist
        directory = self.cache_dir / data_type
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save data
        file_path = directory / f"{identifier}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_serialized_data(self, data_type: str, identifier: str) -> Optional[pd.DataFrame]:
        """
        Load serialized data from disk.
        
        Args:
            data_type: Type of data ('competitions', 'matches', 'events', 'frames')
            identifier: Identifier for the data (e.g., match_id or competition_id)
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        file_path = self.cache_dir / data_type / f"{identifier}.pkl"
        
        if not file_path.exists():
            return None
            
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def prepare_data_for_analysis(self, competition_ids: Optional[List[int]] = None, 
                                max_matches_per_competition: int = 5) -> Dict:
        """
        Prepare a comprehensive dataset for analysis across multiple competitions.
        
        Args:
            competition_ids: List of competition IDs to include (None = use all)
            max_matches_per_competition: Maximum matches to include per competition
            
        Returns:
            Dictionary with analysis-ready data
        """
        analysis_data = {
            'competitions': {},
            'summary': {
                'total_competitions': 0,
                'total_matches': 0,
                'total_events': 0,
                'matches_with_360': 0
            }
        }
        
        # Get competitions
        competitions = self.get_competitions()
        
        # Filter to requested competitions if specified
        if competition_ids:
            competitions = competitions[competitions['competition_id'].isin(competition_ids)]
        
        # Process each competition
        for _, comp in competitions.iterrows():
            comp_id = comp['competition_id'] 
            season_id = comp['season_id']
            
            # Get matches for this competition
            matches = self.get_matches(comp_id, season_id)
            
            # Limit number of matches if needed
            match_subset = matches.head(max_matches_per_competition)
            
            # Store competition data
            analysis_data['competitions'][comp_id] = {
                'name': comp['competition_name'],
                'season': comp['season_name'],
                'matches': {}
            }
            
            # Process each match
            for _, match in match_subset.iterrows():
                match_id = match['match_id']
                
                # Get events for this match
                events = self.get_events(match_id)
                
                # Attempt to get frames for this match (may be empty)
                frames = self.get_freeze_frames(match_id)
                
                # Store match data
                analysis_data['competitions'][comp_id]['matches'][match_id] = {
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'score': f"{match['home_score']}-{match['away_score']}",
                    'events': events,
                    'freeze_frames': frames,
                    'has_360_data': not frames.empty
                }
                
                # Increment 360 data counter if available
                if not frames.empty:
                    analysis_data['summary']['matches_with_360'] += 1
            
            # Update summary counts
            analysis_data['summary']['total_competitions'] += 1
            analysis_data['summary']['total_matches'] += len(match_subset)
            analysis_data['summary']['total_events'] += sum(
                len(m['events']) for m in analysis_data['competitions'][comp_id]['matches'].values()
            )
        
        return analysis_data
    
    def save_analysis_dataset(self, analysis_data: Dict, dataset_name: Optional[str] = None) -> str:
        """
        Save complete analysis dataset to disk in a structured format.
        
        Args:
            analysis_data: Analysis dataset dictionary returned by prepare_data_for_analysis
            dataset_name: Optional name for this dataset (defaults to timestamp)
            
        Returns:
            Path to saved dataset directory
        """
        # Create a unique name for this dataset if not provided
        if dataset_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = f"analysis_{timestamp}"
        
        # Create directory for this dataset
        dataset_dir = self.cache_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        # Save dataset metadata/summary
        summary = {
            'created_at': datetime.datetime.now().isoformat(),
            'summary': analysis_data['summary'],
            'competition_ids': list(analysis_data['competitions'].keys())
        }
        with open(dataset_dir / 'summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
            
        # Create competitions directory
        competitions_dir = dataset_dir / 'competitions'
        competitions_dir.mkdir(exist_ok=True)
        
        # Save each competition's data
        for comp_id, comp_data in analysis_data['competitions'].items():
            # Create competition directory
            comp_dir = competitions_dir / str(comp_id)
            comp_dir.mkdir(exist_ok=True)
            
            # Save competition metadata
            comp_meta = {
                'name': comp_data['name'],
                'season': comp_data['season'],
                'match_ids': list(comp_data['matches'].keys())
            }
            with open(comp_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(comp_meta, f)
            
            # Save each match's data
            matches_dir = comp_dir / 'matches'
            matches_dir.mkdir(exist_ok=True)
            
            for match_id, match_data in comp_data['matches'].items():
                # Create match directory
                match_dir = matches_dir / str(match_id)
                match_dir.mkdir(exist_ok=True)
                
                # Save match metadata
                match_meta = {
                    'home_team': match_data['home_team'],
                    'away_team': match_data['away_team'],
                    'score': match_data['score'],
                    'has_360_data': match_data['has_360_data']
                }
                with open(match_dir / 'metadata.pkl', 'wb') as f:
                    pickle.dump(match_meta, f)
                
                # Save events data
                with open(match_dir / 'events.pkl', 'wb') as f:
                    pickle.dump(match_data['events'], f)
                
                # Save freeze frames if available
                if match_data['has_360_data']:
                    with open(match_dir / 'frames.pkl', 'wb') as f:
                        pickle.dump(match_data['freeze_frames'], f)
        
        print(f"Analysis dataset saved to {dataset_dir}")
        return str(dataset_dir)
    
    def load_analysis_dataset(self, dataset_path: str, load_data: bool = True) -> Dict:
        """
        Load a previously saved analysis dataset from disk.
        
        Args:
            dataset_path: Path to the dataset directory
            load_data: If True, loads all events and frames data (can be memory intensive)
            
        Returns:
            Loaded analysis dataset dictionary
        """
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory {dataset_path} does not exist")
            
        # Load summary
        with open(dataset_dir / 'summary.pkl', 'rb') as f:
            summary_data = pickle.load(f)
            
        # Initialize analysis data structure
        analysis_data = {
            'competitions': {},
            'summary': summary_data['summary']
        }
        
        # Load each competition
        competitions_dir = dataset_dir / 'competitions'
        for comp_id_dir in competitions_dir.iterdir():
            if not comp_id_dir.is_dir():
                continue
                
            comp_id = int(comp_id_dir.name)
            
            # Load competition metadata
            with open(comp_id_dir / 'metadata.pkl', 'rb') as f:
                comp_meta = pickle.load(f)
            
            # Initialize competition data
            analysis_data['competitions'][comp_id] = {
                'name': comp_meta['name'],
                'season': comp_meta['season'],
                'matches': {}
            }
            
            # Load matches
            matches_dir = comp_id_dir / 'matches'
            for match_id_dir in matches_dir.iterdir():
                if not match_id_dir.is_dir():
                    continue
                    
                match_id = int(match_id_dir.name)
                
                # Load match metadata
                with open(match_id_dir / 'metadata.pkl', 'rb') as f:
                    match_meta = pickle.load(f)
                
                # Initialize match data with metadata
                match_data = {
                    'home_team': match_meta['home_team'],
                    'away_team': match_meta['away_team'],
                    'score': match_meta['score'],
                    'has_360_data': match_meta['has_360_data']
                }
                
                # Optionally load events and frames data
                if load_data:
                    # Load events
                    with open(match_id_dir / 'events.pkl', 'rb') as f:
                        match_data['events'] = pickle.load(f)
                    
                    # Load freeze frames if available
                    if match_meta['has_360_data'] and (match_id_dir / 'frames.pkl').exists():
                        with open(match_id_dir / 'frames.pkl', 'rb') as f:
                            match_data['freeze_frames'] = pickle.load(f)
                    else:
                        # Create empty DataFrame with expected structure
                        match_data['freeze_frames'] = pd.DataFrame({
                            'id': [], 'visible_area': [], 'match_id': [],
                            'teammate': [], 'actor': [], 'keeper': [], 'location': []
                        })
                else:
                    # Create placeholder objects that will be loaded on demand
                    match_data['events'] = None
                    match_data['freeze_frames'] = None
                    
                # Add match data to competition
                analysis_data['competitions'][comp_id]['matches'][match_id] = match_data
        
        print(f"Loaded analysis dataset from {dataset_path}")
        print(f"Summary: {analysis_data['summary']}")
        return analysis_data
    
    def get_matches_for_team(self, competition_id: int, season_id: int, team_id: int) -> pd.DataFrame:
        """
        Get all matches for a given team in a competition and season.
        Returns a DataFrame of matches where the team is home or away.
        """
        matches = self.get_matches(competition_id, season_id)
        return matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]

    def get_events_for_team(self, competition_id: int, season_id: int, team_id: int) -> pd.DataFrame:
        """
        Get all events for a given team in a competition and season.
        Returns a DataFrame of events for all matches where the team participated.
        """
        matches = self.get_matches_for_team(competition_id, season_id, team_id)
        all_events = []
        for _, match in matches.iterrows():
            events = self.get_events(match['match_id'])
            team_events = events[events['team_id'] == team_id] if 'team_id' in events else events[events['team'] == team_id]
            all_events.append(team_events)
        if all_events:
            return pd.concat(all_events)
        else:
            return pd.DataFrame()