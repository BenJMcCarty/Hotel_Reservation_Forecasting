"""
Data transformation module to convert reservation records to daily records.
"""
import pandas as pd
from typing import List


def expand_reservations_to_daily(
    df: pd.DataFrame,
    arrival_col: str = 'arrival_date',
    departure_col: str = 'departure_date'
) -> pd.DataFrame:
    """
    Convert single reservation records to daily records based on arrival and departure dates.
    
    Each reservation will be expanded to have one row per day between arrival and departure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with reservation records
    arrival_col : str
        Column name for arrival date
    departure_col : str
        Column name for departure date
        
    Returns:
    --------
    pd.DataFrame
        Expanded DataFrame with daily records
    """
    # Ensure date columns are datetime
    df[arrival_col] = pd.to_datetime(df[arrival_col])
    df[departure_col] = pd.to_datetime(df[departure_col])
    
    # Create a list to store expanded records
    daily_records = []
    
    for idx, row in df.iterrows():
        # Generate date range for this reservation
        date_range = pd.date_range(
            start=row[arrival_col],
            end=row[departure_col],
            inclusive='left'  # Exclude checkout day
        )
        
        # Create a record for each day
        for date in date_range:
            daily_record = row.copy()
            daily_record['date'] = date
            daily_records.append(daily_record)
    
    # Convert to DataFrame
    daily_df = pd.DataFrame(daily_records)
    
    return daily_df


def aggregate_daily_occupancy(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily records to get occupancy counts per day.
    
    Parameters:
    -----------
    daily_df : pd.DataFrame
        DataFrame with daily reservation records
        
    Returns:
    --------
    pd.DataFrame
        Daily occupancy counts
    """
    occupancy = daily_df.groupby('date').size().reset_index(name='occupancy')
    occupancy = occupancy.sort_values('date')
    
    return occupancy
