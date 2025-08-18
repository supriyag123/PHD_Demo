
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class GlobalAnomalyAgent:
    """Global anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples = 10
        
    def detect(self, df: pd.DataFrame) -> bool:
        """Detect global anomaly across all sensors"""
        if len(df) < self.min_samples:
            return False
        
        try:
            # Prepare data (exclude timestamp if present)
            data_cols = [col for col in df.columns if col != 'timestamp']
            data = df[data_cols].dropna()
            
            if len(data) < self.min_samples:
                return False
            
            # Train or retrain model if needed
            if not self.is_trained or len(data) > len(getattr(self, '_last_training_data', [])):
                self._train_model(data)
            
            # Predict on latest sample
            latest_sample = data.tail(1)
            latest_scaled = self.scaler.transform(latest_sample)
            prediction = self.isolation_forest.predict(latest_scaled)
            
            return prediction[0] == -1  # -1 indicates anomaly
            
        except Exception as e:
            return False
    
    def _train_model(self, data: pd.DataFrame):
        """Train the Isolation Forest model"""
        try:
            if len(data) >= self.min_samples:
                data_scaled = self.scaler.fit_transform(data)
                self.isolation_forest.fit(data_scaled)
                self.is_trained = True
                self._last_training_data = data.copy()
        except Exception as e:
            pass
