from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[Any] = None

    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        if not cls.client:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            cls.client = AsyncIOMotorClient(mongodb_url)
            cls.db = cls.client.trinetra_core

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
