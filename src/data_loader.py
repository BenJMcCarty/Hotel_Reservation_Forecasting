"""
Data loading module for hotel reservation data.
"""
import pandas as pd
import requests
from typing import Optional


def download_data(url: str, output_path: str) -> None:
    """
    Download data from a URL and save to local file.
    
    Parameters:
    -----------
    url : str
        URL to download data from
    output_path : str
        Path to save the downloaded file
    """
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Data downloaded successfully to {output_path}")


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw hotel reservation data.
    
    Parameters:
    -----------
    file_path : str
        Path to the raw data file
        
    Returns:
    --------
    pd.DataFrame
        Raw reservation data
    """
    # Adjust based on your file format (CSV, Excel, etc.)
    df = pd.read_csv(file_path)
    return df
