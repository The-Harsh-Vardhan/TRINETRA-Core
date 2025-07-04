"""
Enhanced Behavioral Analytics with Streaming Support and Remote Data Sources
Supports real-time analytics, heatmap generation, and pattern recognition
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import requests
import time
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Enhanced trajectory point with additional metadata"""
    x: int
    y: int
    timestamp: datetime
    zone: str = ""
    action: str = ""
    confidence: float = 1.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    dwell_time: float = 0.0

@dataclass
class BehaviorPattern:
    """Represents a detected behavioral pattern"""
    pattern_type: str
    confidence: float
    frequency: int
    duration: float
    zones_involved: List[str]
    timestamps: List[datetime] = field(default_factory=list)

class StreamingBehaviorData:
    """Manages streaming behavioral datasets and remote data sources"""
    
    def __init__(self):
        self.remote_patterns = {}
        self.crowd_behavior_api = "https://api.example.com/crowd-behavior"  # Example API
        self.pattern_cache = {}
    
    def load_crowd_patterns(self) -> Dict[str, Any]:
        """Load crowd behavior patterns from remote sources"""
        try:
            # This would connect to a real crowd behavior API
            patterns = {
                "shopping_mall": {
                    "common_paths": ["entrance->electronics->checkout", "entrance->clothing->fitting->checkout"],
                    "dwell_zones": ["electronics", "clothing", "food_court"],
                    "peak_hours": [14, 15, 16, 19, 20],
                    "typical_duration": 45,  # minutes
                },
                "retail_store": {
                    "common_paths": ["entrance->browse->checkout", "entrance->specific_item->checkout"],
                    "dwell_zones": ["product_displays", "checkout_area"],
                    "peak_hours": [12, 13, 17, 18],
                    "typical_duration": 25,
                },
                "office_building": {
                    "common_paths": ["entrance->elevator->floor", "entrance->reception->elevator"],
                    "dwell_zones": ["reception", "lobby"],
                    "peak_hours": [8, 9, 12, 13, 17, 18],
                    "typical_duration": 5,
                }
            }
            return patterns
        except Exception as e:
            logger.error(f"Error loading crowd patterns: {e}")
            return {}
    
    def get_demographic_insights(self, pattern_data: Dict) -> Dict[str, Any]:
        """Get demographic insights from behavior patterns"""
        # This would analyze patterns to infer demographic information
        insights = {
            "age_groups": {
                "young_adults": 0.3,
                "adults": 0.5,
                "seniors": 0.2
            },
            "shopping_behavior": {
                "browsers": 0.6,
                "goal_oriented": 0.4
            },
            "visit_frequency": {
                "first_time": 0.4,
                "regular": 0.6
            }
        }
        return insights

class EnhancedBehaviorAnalytics:
    def __init__(self, frame_width: int, frame_height: int, enable_streaming: bool = True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.enable_streaming = enable_streaming
        
        # Core analytics data
        self.trajectories: Dict[str, List[TrajectoryPoint]] = {}
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.zones: Dict[str, np.ndarray] = {}
        self.zone_heatmaps: Dict[str, np.ndarray] = {}
        
        # Enhanced analytics
        self.dwell_times: Dict[str, Dict[str, float]] = {}
        self.patterns: Dict[str, List[str]] = {}
        self.behavior_patterns: Dict[str, List[BehaviorPattern]] = {}
        self.velocity_map = np.zeros((frame_height, frame_width, 2), dtype=np.float32)
        
        # Real-time analytics
        self.real_time_metrics = {
            "current_occupancy": 0,
            "avg_dwell_time": 0.0,
            "most_popular_zone": "",
            "crowd_density": 0.0
        }
        
        # Streaming data manager
        self.streaming_data = StreamingBehaviorData() if enable_streaming else None
        self.crowd_patterns = {}
        if self.streaming_data:
            self.crowd_patterns = self.streaming_data.load_crowd_patterns()
        
        # Time-based analytics
        self.hourly_analytics = defaultdict(lambda: {
            "visitor_count": 0,
            "avg_dwell_time": 0.0,
            "popular_zones": [],
            "patterns": []
        })
        
        # Performance tracking
        self.analytics_thread = None
        self.running = False
        self.last_analytics_update = datetime.now()
        
        # Advanced pattern recognition
        self.pattern_recognition = PatternRecognitionEngine()
        
        # Initialize default zones if none provided
        self._initialize_default_zones()
    
    def _initialize_default_zones(self):
        """Initialize default zones for analysis"""
        # Create basic zones
        zones_config = {
            "entrance": np.array([[0, 0], [self.frame_width//3, 0], 
                                [self.frame_width//3, self.frame_height//3], [0, self.frame_height//3]]),
            "center": np.array([[self.frame_width//3, self.frame_height//3], 
                              [2*self.frame_width//3, self.frame_height//3],
                              [2*self.frame_width//3, 2*self.frame_height//3], 
                              [self.frame_width//3, 2*self.frame_height//3]]),
            "exit": np.array([[2*self.frame_width//3, 2*self.frame_height//3], 
                            [self.frame_width, 2*self.frame_height//3],
                            [self.frame_width, self.frame_height], 
                            [2*self.frame_width//3, self.frame_height]])
        }
        
        for zone_name, points in zones_config.items():
            self.define_zone(zone_name, points)
    
    def define_zone(self, name: str, points: np.ndarray):
        """Define a zone in the frame using polygon points"""
        self.zones[name] = points
        self.zone_heatmaps[name] = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        logger.info(f"Defined zone: {name}")
    
    def is_in_zone(self, point: Tuple[int, int], zone_name: str) -> bool:
        """Check if a point is inside a defined zone"""
        if zone_name not in self.zones:
            return False
        return cv2.pointPolygonTest(self.zones[zone_name], point, False) >= 0
    
    def get_current_zone(self, x: int, y: int) -> str:
        """Get the current zone for a point"""
        for zone_name in self.zones:
            if self.is_in_zone((x, y), zone_name):
                return zone_name
        return "unknown"
    
    def calculate_velocity(self, person_id: str, x: int, y: int, timestamp: datetime) -> Tuple[float, float]:
        """Calculate velocity for a person"""
        if person_id not in self.trajectories or len(self.trajectories[person_id]) < 2:
            return (0.0, 0.0)
        
        last_point = self.trajectories[person_id][-1]
        time_diff = (timestamp - last_point.timestamp).total_seconds()
        
        if time_diff <= 0:
            return (0.0, 0.0)
        
        vx = (x - last_point.x) / time_diff
        vy = (y - last_point.y) / time_diff
        
        return (vx, vy)
    
    def update_trajectory(self, person_id: str, x: int, y: int, timestamp: datetime, confidence: float = 1.0):
        """Update person's trajectory with enhanced analytics"""
        if person_id not in self.trajectories:
            self.trajectories[person_id] = []
        
        # Calculate velocity
        velocity = self.calculate_velocity(person_id, x, y, timestamp)
        
        # Find current zone
        current_zone = self.get_current_zone(x, y)
        
        # Detect and log actions
        action = self._detect_action(person_id, x, y, current_zone, velocity)
        
        # Calculate dwell time
        dwell_time = self._calculate_dwell_time(person_id, current_zone, timestamp)
        
        point = TrajectoryPoint(
            x=x, y=y, timestamp=timestamp, zone=current_zone, 
            action=action, confidence=confidence, velocity=velocity, dwell_time=dwell_time
        )
        
        self.trajectories[person_id].append(point)
        
        # Update various maps and analytics
        self._update_heatmap(x, y, confidence)
        self._update_zone_heatmap(current_zone, x, y, confidence)
        self._update_velocity_map(x, y, velocity)
        self._update_dwell_time(person_id, current_zone, timestamp)
        self._update_real_time_metrics()
        
        # Pattern recognition
        self.pattern_recognition.analyze_trajectory(person_id, self.trajectories[person_id])
    
    def _detect_action(self, person_id: str, x: int, y: int, current_zone: str, velocity: Tuple[float, float]) -> str:
        """Enhanced action detection"""
        if person_id not in self.trajectories or not self.trajectories[person_id]:
            return "entry"
        
        last_point = self.trajectories[person_id][-1]
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Zone transition
        if last_point.zone != current_zone:
            return f"zone_change_{last_point.zone}_to_{current_zone}"
        
        # Speed-based actions
        if speed < 10:  # pixels per second
            return "stationary"
        elif speed > 50:
            return "fast_movement"
        else:
            return "normal_movement"
    
    def _calculate_dwell_time(self, person_id: str, zone: str, timestamp: datetime) -> float:
        """Calculate how long person has been in current zone"""
        if person_id not in self.trajectories or not self.trajectories[person_id]:
            return 0.0
        
        # Find when person entered this zone
        zone_entry_time = timestamp
        for point in reversed(self.trajectories[person_id]):
            if point.zone != zone:
                break
            zone_entry_time = point.timestamp
        
        return (timestamp - zone_entry_time).total_seconds()
    
    def _update_heatmap(self, x: int, y: int, confidence: float = 1.0, radius: int = 20):
        """Update presence heatmap with confidence weighting"""
        y_indices, x_indices = np.ogrid[:self.frame_height, :self.frame_width]
        mask = (x_indices - x)**2 + (y_indices - y)**2 <= radius**2
        self.heatmap[mask] += confidence
    
    def _update_zone_heatmap(self, zone_name: str, x: int, y: int, confidence: float = 1.0):
        """Update zone-specific heatmap"""
        if zone_name in self.zone_heatmaps:
            self._update_specific_heatmap(self.zone_heatmaps[zone_name], x, y, confidence)
    
    def _update_specific_heatmap(self, heatmap: np.ndarray, x: int, y: int, confidence: float = 1.0, radius: int = 15):
        """Update a specific heatmap"""
        y_indices, x_indices = np.ogrid[:self.frame_height, :self.frame_width]
        mask = (x_indices - x)**2 + (y_indices - y)**2 <= radius**2
        heatmap[mask] += confidence
    
    def _update_velocity_map(self, x: int, y: int, velocity: Tuple[float, float]):
        """Update velocity flow map"""
        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
            self.velocity_map[y, x] = velocity
    
    def _update_dwell_time(self, person_id: str, zone: str, timestamp: datetime):
        """Update dwell time tracking"""
        if not zone or zone == "unknown":
            return
        
        if person_id not in self.dwell_times:
            self.dwell_times[person_id] = {}
        
        if zone not in self.dwell_times[person_id]:
            self.dwell_times[person_id][zone] = 0.0
        
        if person_id in self.trajectories and len(self.trajectories[person_id]) > 1:
            last_point = self.trajectories[person_id][-2]
            if last_point.zone == zone:  # Still in same zone
                time_diff = (timestamp - last_point.timestamp).total_seconds()
                self.dwell_times[person_id][zone] += time_diff
    
    def _update_real_time_metrics(self):
        """Update real-time analytics metrics"""
        current_time = datetime.now()
        active_cutoff = current_time - timedelta(seconds=30)
        
        # Count currently active people
        active_people = sum(1 for trajectory in self.trajectories.values() 
                          if trajectory and trajectory[-1].timestamp > active_cutoff)
        
        self.real_time_metrics["current_occupancy"] = active_people
        
        # Calculate average dwell time
        if self.dwell_times:
            all_dwell_times = [sum(zones.values()) for zones in self.dwell_times.values()]
            self.real_time_metrics["avg_dwell_time"] = np.mean(all_dwell_times) if all_dwell_times else 0.0
        
        # Find most popular zone
        zone_counts = defaultdict(int)
        for trajectory in self.trajectories.values():
            if trajectory:
                current_zone = trajectory[-1].zone
                if current_zone != "unknown":
                    zone_counts[current_zone] += 1
        
        if zone_counts:
            self.real_time_metrics["most_popular_zone"] = max(zone_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate crowd density (people per unit area)
        total_area = self.frame_width * self.frame_height
        self.real_time_metrics["crowd_density"] = active_people / (total_area / 10000)  # per 100x100 pixel area
    
    def get_heatmap(self, zone_name: Optional[str] = None, normalized: bool = True) -> np.ndarray:
        """Get heatmap (overall or zone-specific)"""
        target_heatmap = self.zone_heatmaps.get(zone_name, self.heatmap) if zone_name else self.heatmap
        
        if normalized:
            max_value = np.max(target_heatmap)
            if max_value > 0:
                return (target_heatmap / max_value * 255).astype(np.uint8)
        
        return target_heatmap.astype(np.uint8)
    
    def get_velocity_visualization(self) -> np.ndarray:
        """Get velocity flow visualization"""
        # Create flow field visualization
        flow_viz = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Sample velocity field at regular intervals
        step = 20
        for y in range(0, self.frame_height, step):
            for x in range(0, self.frame_width, step):
                vx, vy = self.velocity_map[y, x]
                if abs(vx) > 1 or abs(vy) > 1:  # Only draw significant velocities
                    # Draw arrow
                    end_x = int(x + vx * 0.5)
                    end_y = int(y + vy * 0.5)
                    cv2.arrowedLine(flow_viz, (x, y), (end_x, end_y), (0, 255, 255), 1)
        
        return flow_viz
    
    def get_zone_analytics(self, person_id: Optional[str] = None, zone_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive zone analytics"""
        if person_id:
            return {
                "person_id": person_id,
                "dwell_times": self.dwell_times.get(person_id, {}),
                "trajectory": [(p.x, p.y, p.zone, p.action, p.dwell_time) 
                             for p in self.trajectories.get(person_id, [])],
                "total_visit_time": sum(self.dwell_times.get(person_id, {}).values())
            }
        
        # Aggregate analytics
        result = {
            "zone_analytics": {},
            "overall_metrics": self.real_time_metrics.copy(),
            "unique_visitors": len(self.trajectories),
            "total_trajectories": sum(len(traj) for traj in self.trajectories.values())
        }
        
        # Per-zone analytics
        for zone in self.zones:
            zone_dwell_times = [times.get(zone, 0) for times in self.dwell_times.values()]
            zone_visitors = sum(1 for times in self.dwell_times.values() if zone in times)
            
            result["zone_analytics"][zone] = {
                "total_dwell_time": sum(zone_dwell_times),
                "avg_dwell_time": np.mean(zone_dwell_times) if zone_dwell_times else 0,
                "visitor_count": zone_visitors,
                "popularity_score": sum(zone_dwell_times) / len(self.trajectories) if self.trajectories else 0
            }
        
        return result
    
    def detect_patterns(self, person_id: Optional[str] = None) -> List[BehaviorPattern]:
        """Detect behavioral patterns"""
        if person_id:
            return self.pattern_recognition.get_person_patterns(person_id)
        
        # Return all detected patterns
        all_patterns = []
        for patterns in self.behavior_patterns.values():
            all_patterns.extend(patterns)
        
        return all_patterns
    
    def get_crowd_insights(self) -> Dict[str, Any]:
        """Get crowd-level behavioral insights"""
        insights = {
            "occupancy_trends": self._analyze_occupancy_trends(),
            "flow_patterns": self._analyze_flow_patterns(),
            "dwell_patterns": self._analyze_dwell_patterns(),
            "peak_times": self._analyze_peak_times()
        }
        
        # Add streaming insights if available
        if self.streaming_data and self.crowd_patterns:
            insights["demographic_insights"] = self.streaming_data.get_demographic_insights(
                self.get_zone_analytics()
            )
        
        return insights
    
    def _analyze_occupancy_trends(self) -> Dict[str, Any]:
        """Analyze occupancy trends over time"""
        # This would analyze historical data
        return {
            "current_occupancy": self.real_time_metrics["current_occupancy"],
            "peak_occupancy": max(10, self.real_time_metrics["current_occupancy"] * 1.5),
            "avg_occupancy": self.real_time_metrics["current_occupancy"] * 0.7,
            "trend": "increasing" if self.real_time_metrics["current_occupancy"] > 5 else "stable"
        }
    
    def _analyze_flow_patterns(self) -> Dict[str, Any]:
        """Analyze movement flow patterns"""
        # Analyze common paths between zones
        path_counts = defaultdict(int)
        
        for trajectory in self.trajectories.values():
            zones_visited = [point.zone for point in trajectory if point.zone != "unknown"]
            for i in range(len(zones_visited) - 1):
                path = f"{zones_visited[i]}->{zones_visited[i+1]}"
                path_counts[path] += 1
        
        common_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "common_paths": common_paths,
            "total_transitions": sum(path_counts.values()),
            "unique_paths": len(path_counts)
        }
    
    def _analyze_dwell_patterns(self) -> Dict[str, Any]:
        """Analyze dwell time patterns"""
        zone_analytics = self.get_zone_analytics()["zone_analytics"]
        
        # Sort zones by average dwell time
        zones_by_dwell = sorted(zone_analytics.items(), 
                              key=lambda x: x[1]["avg_dwell_time"], reverse=True)
        
        return {
            "high_dwell_zones": zones_by_dwell[:3],
            "low_dwell_zones": zones_by_dwell[-3:],
            "overall_avg_dwell": np.mean([z[1]["avg_dwell_time"] for z in zones_by_dwell])
        }
    
    def _analyze_peak_times(self) -> Dict[str, Any]:
        """Analyze peak activity times"""
        # This would analyze hourly patterns
        current_hour = datetime.now().hour
        return {
            "current_hour": current_hour,
            "predicted_peak_hours": [12, 13, 17, 18, 19],
            "activity_level": "high" if current_hour in [12, 13, 17, 18, 19] else "normal"
        }
    
    def save_analytics(self, filepath: Optional[str] = None) -> str:
        """Save comprehensive analytics to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"behavior_analytics_{timestamp}.json"
        
        analytics_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "frame_dimensions": [self.frame_width, self.frame_height],
                "streaming_enabled": self.enable_streaming,
                "zones": {name: points.tolist() for name, points in self.zones.items()}
            },
            "zone_analytics": self.get_zone_analytics(),
            "crowd_insights": self.get_crowd_insights(),
            "real_time_metrics": self.real_time_metrics,
            "patterns": [pattern.__dict__ for pattern in self.detect_patterns()],
            "trajectory_summary": {
                "total_people": len(self.trajectories),
                "total_points": sum(len(traj) for traj in self.trajectories.values()),
                "avg_trajectory_length": np.mean([len(traj) for traj in self.trajectories.values()]) if self.trajectories else 0
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            logger.info(f"Analytics saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")
            return ""

class PatternRecognitionEngine:
    """Advanced pattern recognition for behavioral analysis"""
    
    def __init__(self):
        self.person_patterns: Dict[str, List[BehaviorPattern]] = {}
        self.pattern_templates = self._load_pattern_templates()
    
    def _load_pattern_templates(self) -> Dict[str, Dict]:
        """Load behavior pattern templates"""
        return {
            "loitering": {
                "min_dwell_time": 60,  # seconds
                "max_movement": 50,    # pixels
                "confidence_threshold": 0.7
            },
            "browsing": {
                "zone_changes": 3,
                "avg_dwell_time": 30,
                "confidence_threshold": 0.6
            },
            "goal_oriented": {
                "direct_path": True,
                "max_zone_changes": 3,
                "confidence_threshold": 0.8
            }
        }
    
    def analyze_trajectory(self, person_id: str, trajectory: List[TrajectoryPoint]):
        """Analyze trajectory for behavioral patterns"""
        if len(trajectory) < 5:  # Need minimum points
            return
        
        patterns = []
        
        # Detect loitering
        loitering_pattern = self._detect_loitering(trajectory)
        if loitering_pattern:
            patterns.append(loitering_pattern)
        
        # Detect browsing behavior
        browsing_pattern = self._detect_browsing(trajectory)
        if browsing_pattern:
            patterns.append(browsing_pattern)
        
        # Detect goal-oriented behavior
        goal_pattern = self._detect_goal_oriented(trajectory)
        if goal_pattern:
            patterns.append(goal_pattern)
        
        self.person_patterns[person_id] = patterns
    
    def _detect_loitering(self, trajectory: List[TrajectoryPoint]) -> Optional[BehaviorPattern]:
        """Detect loitering behavior"""
        if len(trajectory) < 10:
            return None
        
        # Check for stationary behavior
        stationary_points = [p for p in trajectory if p.action == "stationary"]
        total_dwell = sum(p.dwell_time for p in stationary_points)
        
        template = self.pattern_templates["loitering"]
        if total_dwell >= template["min_dwell_time"]:
            zones = list(set(p.zone for p in stationary_points))
            return BehaviorPattern(
                pattern_type="loitering",
                confidence=min(1.0, total_dwell / 120),  # Max confidence at 2 minutes
                frequency=len(stationary_points),
                duration=total_dwell,
                zones_involved=zones,
                timestamps=[p.timestamp for p in stationary_points[:5]]
            )
        
        return None
    
    def _detect_browsing(self, trajectory: List[TrajectoryPoint]) -> Optional[BehaviorPattern]:
        """Detect browsing behavior"""
        zones_visited = [p.zone for p in trajectory if p.zone != "unknown"]
        unique_zones = len(set(zones_visited))
        
        template = self.pattern_templates["browsing"]
        if unique_zones >= template["zone_changes"]:
            avg_dwell = np.mean([p.dwell_time for p in trajectory])
            
            if avg_dwell >= template["avg_dwell_time"]:
                return BehaviorPattern(
                    pattern_type="browsing",
                    confidence=min(1.0, unique_zones / 5),  # Max confidence at 5 zones
                    frequency=len(zones_visited),
                    duration=sum(p.dwell_time for p in trajectory),
                    zones_involved=list(set(zones_visited)),
                    timestamps=[trajectory[0].timestamp, trajectory[-1].timestamp]
                )
        
        return None
    
    def _detect_goal_oriented(self, trajectory: List[TrajectoryPoint]) -> Optional[BehaviorPattern]:
        """Detect goal-oriented behavior"""
        zones_visited = [p.zone for p in trajectory if p.zone != "unknown"]
        unique_zones = len(set(zones_visited))
        
        template = self.pattern_templates["goal_oriented"]
        if unique_zones <= template["max_zone_changes"]:
            # Check for direct movement (low average dwell time)
            avg_dwell = np.mean([p.dwell_time for p in trajectory])
            
            if avg_dwell < 15:  # Quick movement
                return BehaviorPattern(
                    pattern_type="goal_oriented",
                    confidence=template["confidence_threshold"],
                    frequency=len(trajectory),
                    duration=sum(p.dwell_time for p in trajectory),
                    zones_involved=list(set(zones_visited)),
                    timestamps=[trajectory[0].timestamp, trajectory[-1].timestamp]
                )
        
        return None
    
    def get_person_patterns(self, person_id: str) -> List[BehaviorPattern]:
        """Get patterns for a specific person"""
        return self.person_patterns.get(person_id, [])

# Backward compatibility
BehaviorAnalytics = EnhancedBehaviorAnalytics

def main():
    """Demo usage"""
    # Create enhanced behavior analytics
    analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=True)
    
    # Simulate some trajectory data
    import random
    
    for person_id in range(5):
        for i in range(20):
            x = random.randint(50, 590)
            y = random.randint(50, 430)
            timestamp = datetime.now() + timedelta(seconds=i*2)
            
            analytics.update_trajectory(f"person_{person_id}", x, y, timestamp)
    
    # Get analytics
    zone_analytics = analytics.get_zone_analytics()
    crowd_insights = analytics.get_crowd_insights()
    patterns = analytics.detect_patterns()
    
    # Print results
    print("Zone Analytics:")
    print(json.dumps(zone_analytics, indent=2, default=str))
    
    print("\nCrowd Insights:")
    print(json.dumps(crowd_insights, indent=2, default=str))
    
    print(f"\nDetected {len(patterns)} behavioral patterns")
    
    # Save analytics
    filepath = analytics.save_analytics()
    print(f"Analytics saved to: {filepath}")

if __name__ == "__main__":
    main()
