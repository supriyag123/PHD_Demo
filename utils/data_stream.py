import pandas as pd
import numpy as np
import datetime as dt
from typing import Iterator, Dict, Any

def simulate_stream(filename: str = "data/MetroPT.csv") -> Iterator[Dict[str, Any]]:
    """
    Simulates real-time data stream from MetroPT dataset
    """
    try:
        # Load MetroPT data (assuming it exists, otherwise generate synthetic data)
        try:
            df = pd.read_csv(filename)
            # Assume MetroPT has columns like 'timestamp', 'sensor_1', 'sensor_2', etc.
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1T')
        except FileNotFoundError:
            # Generate synthetic MetroPT-like data if file doesn't exist
            df = generate_synthetic_metropt_data()
        
        # Ensure we have sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_') or col in ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']]
        if not sensor_cols:
            # Create sensor columns from existing numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            sensor_cols = [f'sensor_{i+1}' for i in range(min(5, len(numeric_cols)))]
            for i, col in enumerate(sensor_cols):
                if i < len(numeric_cols):
                    df[col] = df[numeric_cols[i]]
                else:
                    df[col] = np.random.normal(50, 10, len(df))
    
        # Stream the data
        for _, row in df.iterrows():
            timestamp = row.get('timestamp', dt.datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            readings = {col: row[col] for col in sensor_cols}
            
            yield {
                "timestamp": timestamp,
                "readings": readings
            }
            
    except Exception as e:
        # Fallback to synthetic data
        yield from generate_synthetic_stream()

def generate_synthetic_metropt_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic MetroPT-like transportation data"""
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1T')
    
    # Simulate transportation metrics with realistic patterns
    base_patterns = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 20 + 50
    
    data = {
        'timestamp': timestamps,
        'sensor_1': base_patterns + np.random.normal(0, 5, n_samples),  # Passenger count
        'sensor_2': base_patterns * 0.8 + np.random.normal(0, 3, n_samples),  # Door cycles
        'sensor_3': np.abs(base_patterns * 0.6) + np.random.normal(0, 2, n_samples),  # Speed
        'sensor_4': 25 + np.sin(np.linspace(0, 2*np.pi, n_samples)) * 5 + np.random.normal(0, 1, n_samples),  # Temperature
        'sensor_5': base_patterns * 1.2 + np.random.normal(0, 8, n_samples)  # Vibration
    }
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for idx in anomaly_indices:
        sensor_to_anomalize = np.random.choice(['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'])
        data[sensor_to_anomalize][idx] += np.random.choice([-1, 1]) * np.random.uniform(30, 50)
    
    return pd.DataFrame(data)

def generate_synthetic_stream() -> Iterator[Dict[str, Any]]:
    """Fallback synthetic stream generator"""
    while True:
        timestamp = dt.datetime.now()
        readings = {
            f"sensor_{i}": np.random.normal(50, 10) for i in range(1, 6)
        }
        yield {
            "timestamp": timestamp,
            "readings": readings
        }
