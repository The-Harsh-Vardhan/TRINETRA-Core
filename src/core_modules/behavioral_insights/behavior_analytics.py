from datetime import datetime, timedelta
import random
import pprint
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Simulated visit log
visit_log = {}

@dataclass
class TrajectoryPoint:
    x: int
    y: int
    timestamp: datetime
    zone: str = ""
    action: str = ""

class BehaviorAnalytics:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.trajectories: Dict[str, List[TrajectoryPoint]] = {}
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.zones: Dict[str, np.ndarray] = {}
        self.dwell_times: Dict[str, Dict[str, float]] = {}  # person_id -> zone -> time
        self.patterns: Dict[str, List[str]] = {}  # person_id -> list of actions
        
    def define_zone(self, name: str, points: np.ndarray):
        """Define a zone in the frame using polygon points"""
        self.zones[name] = points
        
    def is_in_zone(self, point: Tuple[int, int], zone_name: str) -> bool:
        """Check if a point is inside a defined zone"""
        if zone_name not in self.zones:
            return False
        return cv2.pointPolygonTest(self.zones[zone_name], point, False) >= 0
    
    def update_trajectory(self, person_id: str, x: int, y: int, timestamp: datetime):
        """Update person's trajectory"""
        if person_id not in self.trajectories:
            self.trajectories[person_id] = []
            
        # Find current zone
        current_zone = ""
        for zone_name in self.zones:
            if self.is_in_zone((x, y), zone_name):
                current_zone = zone_name
                break
                
        # Detect and log actions
        action = self._detect_action(person_id, x, y, current_zone)
        
        point = TrajectoryPoint(x, y, timestamp, current_zone, action)
        self.trajectories[person_id].append(point)
        
        # Update heatmap
        self._update_heatmap(x, y)
        
        # Update dwell times
        self._update_dwell_time(person_id, current_zone, timestamp)
    
    def _detect_action(self, person_id: str, x: int, y: int, current_zone: str) -> str:
        """Detect actions based on movement patterns"""
        if person_id not in self.trajectories or not self.trajectories[person_id]:
            return "entry"
            
        last_point = self.trajectories[person_id][-1]
        
        # Zone transition
        if last_point.zone != current_zone:
            return f"zone_change_{last_point.zone}_to_{current_zone}"
            
        # Stationary detection
        if self._is_stationary(x, y, last_point):
            return "stationary"
            
        return "moving"
    
    def _is_stationary(self, x: int, y: int, last_point: TrajectoryPoint, threshold: int = 10) -> bool:
        """Detect if person is stationary"""
        distance = np.sqrt((x - last_point.x)**2 + (y - last_point.y)**2)
        return distance < threshold
    
    def _update_heatmap(self, x: int, y: int, radius: int = 20):
        """Update presence heatmap"""
        y_indices, x_indices = np.ogrid[:self.frame_height, :self.frame_width]
        mask = (x_indices - x)**2 + (y_indices - y)**2 <= radius**2
        self.heatmap[mask] += 1
    
    def _update_dwell_time(self, person_id: str, zone: str, timestamp: datetime):
        """Update dwell time in zones"""
        if not zone:
            return
            
        if person_id not in self.dwell_times:
            self.dwell_times[person_id] = {}
            
        if zone not in self.dwell_times[person_id]:
            self.dwell_times[person_id][zone] = 0.0
            
        if person_id in self.trajectories and len(self.trajectories[person_id]) > 1:
            last_point = self.trajectories[person_id][-2]
            time_diff = (timestamp - last_point.timestamp).total_seconds()
            self.dwell_times[person_id][zone] += time_diff
    
    def get_heatmap(self, normalized: bool = True) -> np.ndarray:
        """Get the current heatmap"""
        if normalized:
            max_value = np.max(self.heatmap)
            if max_value > 0:
                return (self.heatmap / max_value * 255).astype(np.uint8)
        return self.heatmap.astype(np.uint8)
    
    def get_zone_analytics(self, person_id: Optional[str] = None) -> Dict:
        """Get analytics for zones"""
        if person_id:
            return {
                "dwell_times": self.dwell_times.get(person_id, {}),
                "trajectory": [(p.x, p.y, p.zone, p.action) 
                             for p in self.trajectories.get(person_id, [])]
            }
        
        # Aggregate analytics for all people
        total_dwell_times = {}
        for zone in self.zones:
            total_dwell_times[zone] = sum(
                times.get(zone, 0) 
                for times in self.dwell_times.values()
            )
            
        return {
            "total_dwell_times": total_dwell_times,
            "unique_visitors": len(self.trajectories),
            "popular_zones": sorted(
                total_dwell_times.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }
    
    def detect_patterns(self, person_id: str) -> List[str]:
        """Detect behavioral patterns"""
        if person_id not in self.trajectories:
            return []
            
        trajectory = self.trajectories[person_id]
        patterns = []
        
        # Detect zone visiting patterns
        zone_sequence = [point.zone for point in trajectory if point.zone]
        if zone_sequence:
            patterns.append(f"Zone sequence: {' -> '.join(zone_sequence)}")
        
        # Detect dwell time patterns
        if person_id in self.dwell_times:
            high_dwell_zones = [
                zone for zone, time in self.dwell_times[person_id].items()
                if time > 300  # More than 5 minutes
            ]
            if high_dwell_zones:
                patterns.append(f"High dwell time in: {', '.join(high_dwell_zones)}")
        
        # Detect movement patterns
        actions = [point.action for point in trajectory]
        stationary_count = actions.count("stationary")
        if stationary_count > len(actions) * 0.5:
            patterns.append("Predominantly stationary behavior")
        elif stationary_count < len(actions) * 0.2:
            patterns.append("Highly mobile behavior")
            
        return patterns
    
    def save_analytics(self, filepath: str):
        """Save analytics data to file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "zone_analytics": self.get_zone_analytics(),
            "trajectories": {
                person_id: [(p.x, p.y, p.zone, p.action, p.timestamp.isoformat())
                           for p in trajectory]
                for person_id, trajectory in self.trajectories.items()
            },
            "dwell_times": self.dwell_times
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

# Step 1: Log entry time
def log_entry(customer_id):
    visit_log[customer_id] = {
        "entry_time": datetime.now(),
        "items": [],
        "total_value": 0,
        "staff": None,
        "exit_time": None
    }

# Step 2: Log billing data
def log_billing(customer_id, order_items, value, staff):
    if customer_id not in visit_log:
        log_entry(customer_id)  # fallback
    visit_log[customer_id]["items"].extend(order_items)
    visit_log[customer_id]["total_value"] += value
    visit_log[customer_id]["staff"] = staff

# Step 3: Log exit time
def log_exit(customer_id):
    if customer_id in visit_log:
        visit_log[customer_id]["exit_time"] = datetime.now()

# Step 4: Compute behavior insights
def compute_behavior(customer_id):
    data = visit_log.get(customer_id)
    if not data:
        return None

    entry = data["entry_time"]
    exit_ = data["exit_time"] or datetime.now()
    duration = (exit_ - entry).total_seconds() / 60.0  # in minutes

    # Simulate waiting time
    waiting_time = random.uniform(1, 5)

    return {
        "time_spent": f"{duration:.2f} minutes",
        "waiting_time": f"{waiting_time:.1f} minutes",
        "favorite_items": list(set(data["items"])),
        "order_value": data["total_value"],
        "staff_served": data["staff"]
    }

# 🔍 Simulate session
customer_id = "john_doe"

log_entry(customer_id)
log_billing(customer_id, ["Latte", "Pastry"], 320, "Ritika")
log_exit(customer_id)

# Report
insights = compute_behavior(customer_id)
print("\n[TRINETRA - Behavioral Summary]")
pprint.pprint(insights)