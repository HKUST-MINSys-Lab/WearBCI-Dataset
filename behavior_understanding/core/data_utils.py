import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import os

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data and handles basic timestamp normalization."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Normalize timestamp if present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            # Fallback for numeric timestamps
            pass
            
    return df

def segment_data(df: pd.DataFrame, window_size_sec: float, sampling_rate: float) -> List[pd.DataFrame]:
    """Segments a DataFrame into chunks based on window size and sampling rate."""
    samples_per_window = int(window_size_sec * sampling_rate)
    segments = []
    
    for i in range(0, len(df), samples_per_window):
        segment = df.iloc[i : i + samples_per_window]
        if len(segment) >= samples_per_window * 0.8: # Require 80% of window samples
            segments.append(segment)
            
    return segments

def save_json_results(results: Any, output_path: str):
    """Saves analysis results to a JSON file."""
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
