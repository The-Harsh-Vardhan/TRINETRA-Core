import pytest
from httpx import AsyncClient
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
from src.backend.main import app
from src.backend.database.client import Database

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture(autouse=True)
async def setup_database():
    await Database.connect_db()
    yield
    await Database.close_db()

@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to TRINETRA Core API"}

@pytest.mark.asyncio
class TestFaceRecognition:
    async def test_face_registration(self, client):
        # Create a sample image with a face
        img_path = "tests/test_data/sample_face.jpg"
        if not os.path.exists(img_path):
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(img, "Test", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, img)

        with open(img_path, "rb") as f:
            response = await client.post(
                "/api/face/register",
                files={"image": ("image.jpg", f, "image/jpeg")},
                data={"name": "Test Person"}
            )
        
        assert response.status_code == 200
        assert "message" in response.json()

    async def test_face_recognition(self, client):
        img_path = "tests/test_data/sample_face.jpg"
        with open(img_path, "rb") as f:
            response = await client.post(
                "/api/face/recognize",
                files={"image": ("image.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        assert "results" in response.json()

@pytest.mark.asyncio
class TestEntranceTracking:
    async def test_entrance_logging(self, client):
        data = {
            "direction": "entry",
            "camera_id": "cam1",
            "person_count": 2
        }
        response = await client.post("/api/entrance/log", json=data)
        assert response.status_code == 200
        assert "event_id" in response.json()

    async def test_entrance_stats(self, client):
        # First log some data
        await self.test_entrance_logging(client)
        
        # Then get stats
        now = datetime.now()
        params = {
            "start_time": (now - timedelta(hours=1)).isoformat(),
            "end_time": now.isoformat()
        }
        response = await client.get("/api/entrance/stats", params=params)
        assert response.status_code == 200
        stats = response.json()
        assert "total_entries" in stats
        assert "total_exits" in stats

@pytest.mark.asyncio
class TestBehaviorAnalytics:
    async def test_behavior_logging(self, client):
        data = {
            "person_id": "test_person",
            "x": 100,
            "y": 200,
            "zone": "entrance"
        }
        response = await client.post("/api/behavior/log", json=data)
        assert response.status_code == 200
        assert "event_id" in response.json()

    async def test_behavior_analysis(self, client):
        # First log some behavior
        await self.test_behavior_logging(client)
        
        # Then get analysis
        response = await client.get("/api/behavior/analysis/test_person")
        assert response.status_code == 200
        analysis = response.json()
        assert "analysis" in analysis
        assert "patterns" in analysis

    async def test_heatmap(self, client):
        response = await client.get("/api/behavior/heatmap")
        assert response.status_code == 200
        assert "heatmap" in response.json()

@pytest.mark.asyncio
class TestDatabaseOperations:
    async def test_face_database_operations(self):
        # Test inserting face data
        face_data = {
            "name": "Test Person",
            "registered_at": datetime.now()
        }
        face_id = await Database.insert_face(face_data)
        assert face_id is not None
        
        # Test retrieving face data
        retrieved_face = await Database.get_face(face_id)
        assert retrieved_face is not None
        assert retrieved_face["name"] == "Test Person"

    async def test_entrance_database_operations(self):
        # Test logging entrance
        entrance_data = {
            "timestamp": datetime.now(),
            "direction": "entry",
            "camera_id": "cam1",
            "person_count": 2
        }
        event_id = await Database.log_entrance(entrance_data)
        assert event_id is not None
        
        # Test getting entrance stats
        now = datetime.now()
        stats = await Database.get_entrance_stats({
            "start": now - timedelta(hours=1),
            "end": now
        })
        assert stats is not None
        assert isinstance(stats, dict)
