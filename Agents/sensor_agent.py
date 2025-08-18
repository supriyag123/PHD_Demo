
import numpy as np
from collections import deque
from typing import List

class SensorAgent:
    """Individual sensor agent for local anomaly detection"""
    
    def __init__(self, sensor_id: str, window_size: int = 10, threshold_std: float = 2.0):
        self.sensor_id = sensor_id
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.history = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
        
    def detect(self, value: float) -> bool:
        """Detect local anomaly using statistical threshold"""
        self.history.append(value)
        
        if len(self.history) < self.window_size:
            return False
        
        # Update baseline statistics
        self.baseline_mean = np.mean(self.history)
        self.baseline_std = np.std(self.history)
        
        if self.baseline_std == 0:
            return False
        
        # Z-score anomaly detection
        z_score = abs(value - self.baseline_mean) / self.baseline_std
        return z_score > self.threshold_std
    
    def get_status(self) -> dict:
        """Get current sensor status"""
        return {
            'sensor_id': self.sensor_id,
            'current_mean': self.baseline_mean,
            'current_std': self.baseline_std,
            'history_size': len(self.history)
        }
