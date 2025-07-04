from fastapi import APIRouter, HTTPException
from typing import Dict, List
from datetime import datetime
from ...core_modules.entrance_tracking.people_counter import PeopleCounter
from ...database.client import Database

router = APIRouter()

@router.get("/stats")
async def get_entrance_stats(
    start_time: datetime,
    end_time: datetime
):
    try:
        stats = await Database.get_entrance_stats({
            "start": start_time,
            "end": end_time
        })
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log")
async def log_entrance_event(
    direction: str,
    camera_id: str,
    person_count: int
):
    try:
        entrance_data = {
            "timestamp": datetime.now(),
            "direction": direction,
            "camera_id": camera_id,
            "person_count": person_count
        }
        event_id = await Database.log_entrance(entrance_data)
        return {"message": "Event logged successfully", "event_id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
