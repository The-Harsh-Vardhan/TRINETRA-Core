from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
from ...core_modules.behavioral_insights.behavior_analytics import BehaviorAnalytics
from ...database.client import Database

router = APIRouter()

@router.post("/log")
async def log_behavior(
    person_id: str,
    x: int,
    y: int,
    zone: Optional[str] = None
):
    try:
        behavior_data = {
            "timestamp": datetime.now(),
            "person_id": person_id,
            "position": {"x": x, "y": y},
            "zone": zone
        }
        event_id = await Database.log_behavior(behavior_data)
        return {"message": "Behavior logged successfully", "event_id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{person_id}")
async def get_behavior_analysis(person_id: str):
    try:
        behavior_analytics = BehaviorAnalytics(frame_width=1920, frame_height=1080)  # Adjust dimensions as needed
        analysis = behavior_analytics.get_zone_analytics(person_id)
        patterns = behavior_analytics.detect_patterns(person_id)
        
        return {
            "analysis": analysis,
            "patterns": patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/heatmap")
async def get_heatmap():
    try:
        behavior_analytics = BehaviorAnalytics(frame_width=1920, frame_height=1080)  # Adjust dimensions as needed
        heatmap = behavior_analytics.get_heatmap(normalized=True)
        
        # Convert numpy array to list for JSON serialization
        return {"heatmap": heatmap.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
