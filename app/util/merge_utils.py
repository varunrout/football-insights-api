# app/util/merge_utils.py
import pandas as pd

def merge_events_360(events_df, freeze_df):
    """
    Merge events data with 360 freeze-frame data.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing event data
    freeze_df : pandas.DataFrame
        DataFrame containing 360 freeze-frame data
    
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with event and freeze-frame data
    """
    # Check if inputs are pandas DataFrames
    if not isinstance(events_df, pd.DataFrame) or not isinstance(freeze_df, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames")
    
   
    
    # Merge the dataframes on id field
    merged_df = events_df.merge(freeze_df, on='id', how='left')
    
    return merged_df
