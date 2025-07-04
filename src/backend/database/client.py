from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[Any] = None

    # Collections
    faces_collection: Optional[Any] = None
    entrance_data_collection: Optional[Any] = None
    behavior_collection: Optional[Any] = None

    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        if not cls.client:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            cls.client = AsyncIOMotorClient(mongodb_url)
            cls.db = cls.client.trinetra_core
            
            # Initialize collections
            cls.faces_collection = cls.db.faces
            cls.entrance_data_collection = cls.db.entrance_data
            cls.behavior_collection = cls.db.behavior_analytics

    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None

    @classmethod
    async def get_db(cls):
        """Get database instance"""
        if not cls.db:
            await cls.connect_db()
        return cls.db

    # Face Recognition CRUD operations
    @classmethod
    async def insert_face(cls, face_data: Dict[str, Any]) -> str:
        """Insert face data into database"""
        result = await cls.faces_collection.insert_one(face_data)
        return str(result.inserted_id)

    @classmethod
    async def get_face(cls, face_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve face data by ID"""
        return await cls.faces_collection.find_one({"_id": ObjectId(face_id)})

    # Entrance Tracking CRUD operations
    @classmethod
    async def log_entrance(cls, entrance_data: Dict[str, Any]) -> str:
        """Log entrance/exit event"""
        result = await cls.entrance_data_collection.insert_one(entrance_data)
        return str(result.inserted_id)

    @classmethod
    async def get_entrance_stats(cls, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Get entrance statistics for a time range"""
        stats = await cls.entrance_data_collection.aggregate([
            {"$match": {"timestamp": {"$gte": time_range["start"], "$lte": time_range["end"]}}},
            {"$group": {
                "_id": None,
                "total_entries": {"$sum": {"$cond": [{"$eq": ["$direction", "entry"]}, 1, 0]}},
                "total_exits": {"$sum": {"$cond": [{"$eq": ["$direction", "exit"]}, 1, 0]}}
            }}
        ]).to_list(1)
        return stats[0] if stats else {"total_entries": 0, "total_exits": 0}

    # Behavioral Analytics CRUD operations
    @classmethod
    async def log_behavior(cls, behavior_data: Dict[str, Any]) -> str:
        """Log behavioral data"""
        result = await cls.behavior_collection.insert_one(behavior_data)
        return str(result.inserted_id)

    @classmethod
    async def get_behavior_analysis(cls, query: Dict[str, Any]) -> list:
        """Get behavioral analysis based on query parameters"""
        return await cls.behavior_collection.find(query).to_list(None)
