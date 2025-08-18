
from typing import Dict, List, Tuple
import datetime as dt

class MasterAgent:
    """Master agent that orchestrates all other agents and makes final decisions"""
    
    def __init__(self, sensor_agents: Dict, global_agent, adaptive_agent):
        self.sensor_agents = sensor_agents
        self.global_agent = global_agent
        self.adaptive_agent = adaptive_agent
        self.logs = []
        self.decision_history = []
        
    def decide(self, local_alerts: List[str], global_anomaly: bool) -> Tuple[bool, str]:
        """Make final anomaly decision based on all agent inputs"""
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        
        # Decision logic
        local_alert_count = len(local_alerts)
        
        # Reasoning logic
        if global_anomaly and local_alert_count >= 2:
            decision = True
            reasoning = f"🚨 CRITICAL: Global anomaly detected with {local_alert_count} local alerts ({', '.join(local_alerts)})"
        elif global_anomaly:
            decision = True
            reasoning = f"⚠️ WARNING: Global pattern anomaly detected across sensor network"
        elif local_alert_count >= 3:
            decision = True
            reasoning = f"🔍 THRESHOLD: Multiple sensor alerts ({local_alert_count}/5) - {', '.join(local_alerts)}"
        elif local_alert_count >= 1:
            decision = False
            reasoning = f"📊 MONITORING: Local alerts on {', '.join(local_alerts)} - within normal variance"
        else:
            decision = False
            reasoning = "✅ NORMAL: All sensors operating within expected parameters"
        
        # Log decision
        log_entry = f"[{timestamp}] Decision: {'ANOMALY' if decision else 'NORMAL'} | Local: {local_alert_count} | Global: {global_anomaly}"
        self.logs.append(log_entry)
        
        # Keep log size manageable
        if len(self.logs) > 50:
            self.logs = self.logs[-25:]
        
        self.decision_history.append({
            'timestamp': timestamp,
            'decision': decision,
            'local_alerts': local_alert_count,
            'global_anomaly': global_anomaly
        })
        
        return decision, reasoning
    
    def get_logs(self) -> List[str]:
        """Get recent decision logs"""
        return self.logs[-10:]  # Return last 10 logs
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        if not self.decision_history:
            return {"total_decisions": 0, "anomaly_rate": 0.0}
        
        total = len(self.decision_history)
        anomalies = sum(1 for d in self.decision_history if d['decision'])
        
        return {
            "total_decisions": total,
            "anomaly_rate": anomalies / total if total > 0 else 0.0,
            "recent_anomalies": anomalies
        }
