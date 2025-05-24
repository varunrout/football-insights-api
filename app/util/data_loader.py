# app/util/data_loader.py

import pandas as pd
from statsbombpy import sb

def convert_events_to_df(events_data):
    # Desired columns
    columns = ['50_50', 'bad_behaviour_card', 'ball_receipt_outcome',
       'block_deflection', 'block_save_block', 'carry_end_location',
       'clearance_aerial_won', 'clearance_body_part', 'clearance_head',
       'clearance_left_foot', 'clearance_right_foot', 'counterpress',
       'dribble_outcome', 'duel_outcome', 'duel_type', 'duration',
       'foul_committed_advantage', 'foul_committed_card',
       'foul_committed_penalty', 'foul_won_advantage', 'foul_won_defensive',
       'foul_won_penalty', 'goalkeeper_body_part', 'goalkeeper_end_location',
       'goalkeeper_outcome', 'goalkeeper_position', 'goalkeeper_punched_out',
       'goalkeeper_technique', 'goalkeeper_type', 'id', 'index',
       'injury_stoppage_in_chain', 'interception_outcome', 'location',
       'match_id', 'minute', 'off_camera', 'out', 'pass_aerial_won',
       'pass_angle', 'pass_assisted_shot_id', 'pass_body_part', 'pass_cross',
       'pass_cut_back', 'pass_deflected', 'pass_end_location',
       'pass_goal_assist', 'pass_height', 'pass_length', 'pass_outcome',
       'pass_outswinging', 'pass_recipient', 'pass_recipient_id',
       'pass_shot_assist', 'pass_switch', 'pass_technique',
       'pass_through_ball', 'pass_type', 'period', 'play_pattern', 'player',
       'player_id', 'position', 'possession', 'possession_team',
       'possession_team_id', 'related_events', 'second', 'shot_aerial_won',
       'shot_body_part', 'shot_deflected', 'shot_end_location',
       'shot_first_time', 'shot_freeze_frame', 'shot_key_pass_id',
       'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type',
       'substitution_outcome', 'substitution_outcome_id',
       'substitution_replacement', 'substitution_replacement_id', 'tactics',
       'team', 'team_id', 'timestamp', 'type', 'under_pressure']
    
    # Create a DataFrame based on the type of events_data
    if isinstance(events_data, dict):
        df = pd.DataFrame.from_dict(events_data, orient='index')
    else:
        df = pd.DataFrame(events_data)
    print(f"DEBUG: Initial events DataFrame shape: {df.shape}")
    
    # Ensure all desired columns exist, add missing ones as None
    for col in columns:
        if col not in df.columns:
            print(f"DEBUG: Missing column '{col}' detected. Adding it with default None")
            df[col] = None
    df = df[columns]
    
    print(f"DEBUG: DataFrame shape after standardizing columns: {df.shape}")
    df.set_index('id', inplace=True)
    print("DEBUG: Set 'id' as index. Final events DataFrame shape:", df.shape)
    return df

def pull_freeze_frame(match_id):
    freeze_frame = sb.frames(match_id=match_id)
    print(f"Freeze frame data for match_id {match_id}:")
    print(freeze_frame.info())
    return freeze_frame

def load_data():
    """
    Loads competitions, filters to competitions with match_360_available (excluding women competitions),
    and then loads events and freeze frame data for the first available match.
    """
    competitions = sb.competitions()
    # Filter: match_360_available not None and exclude women competitions
    competitions = competitions[(competitions['match_360_available'].notna()) &
                                (competitions['competition_gender'] != 'women')]
    print("DEBUG: Competitions filtered. Total:", competitions.shape[0])
    
    matches = sb.matches(competition_id=competitions.iloc[0]['competition_id'],
                         season_id=competitions.iloc[0]['season_id'])
    print("DEBUG: Matches loaded. Total:", matches.shape[0])
    
    # Pick first match for demonstration
    match_id = matches['match_id'].iloc[0]
    print("DEBUG: Using match_id:", match_id)
    
    events = sb.events(match_id=match_id)
    events_df = convert_events_to_df(events)
    
    freeze_frame_df = pull_freeze_frame(match_id)
    
    return events_df, freeze_frame_df, competitions, matches
