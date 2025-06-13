from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Optional
import motor.motor_asyncio
from pydantic import BaseModel

app = FastAPI(title="TRINETRA Core API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = "mongodb://localhost:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.trinetra_core

# Data models
class CustomerJourney(BaseModel):
    customer_id: str
    timestamp: datetime
    camera_id: str
    location: Dict[str, float]  # x, y coordinates
    zone: str
    duration: float  # time spent in seconds

class FaceRecognition(BaseModel):
    customer_id: str
    timestamp: datetime
    confidence: float
    camera_id: str

class BehaviorMetrics(BaseModel):
    customer_id: str
    visit_count: int
    avg_duration: float
    last_visit: datetime
    customer_segment: str
    recognition_rate: float

# API Routes
@app.get("/")
async def root():
    return {"status": "TRINETRA Core API is running"}

@app.get("/analytics/traffic")
async def get_traffic_analytics(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Get traffic analytics data"""
    query = {}
    if start_time and end_time:
        query["timestamp"] = {"$gte": start_time, "$lte": end_time}
    
    traffic_data = await db.traffic_analytics.find(query).to_list(1000)
    return traffic_data

@app.get("/analytics/customer/{customer_id}")
async def get_customer_analytics(customer_id: str):
    """Get analytics for a specific customer"""
    customer = await db.customers.find_one({"customer_id": customer_id})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.get("/tracking/journey/{customer_id}")
async def get_customer_journey(customer_id: str):
    """Get journey data for a specific customer"""
    journey = await db.journeys.find({"customer_id": customer_id}).to_list(1000)
    return journey

@app.get("/recognition/history/{customer_id}")
async def get_recognition_history(customer_id: str):
    """Get face recognition history for a customer"""
    history = await db.recognitions.find({"customer_id": customer_id}).to_list(1000)
    return history

@app.get("/analytics/behavior/{customer_id}")
async def get_behavior_metrics(customer_id: str):
    """Get behavioral metrics for a customer"""
    metrics = await db.behavior_metrics.find_one({"customer_id": customer_id})
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return metrics
