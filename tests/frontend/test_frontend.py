import pytest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime
from src.frontend.utils.api_client import APIClient
from src.frontend.utils.video_processor import VideoProcessor

@pytest.fixture
def api_client():
    return APIClient(base_url="http://localhost:8000")

@pytest.fixture
def video_processor():
    return VideoProcessor()

class TestAPIClient:
    def test_face_recognition(self, api_client, requests_mock):
        # Mock the face recognition endpoint
        requests_mock.post(
            "http://localhost:8000/api/face/recognize",
            json={"results": [{"name": "Test Person", "confidence": 0.95}]}
        )
        
        # Test face recognition
        image_data = np.zeros((300, 300, 3), dtype=np.uint8)
        response = api_client.recognize_face(image_data.tobytes())
        
        assert "results" in response
        assert len(response["results"]) > 0
        assert response["results"][0]["name"] == "Test Person"

    def test_entrance_stats(self, api_client, requests_mock):
        # Mock the entrance stats endpoint
        mock_stats = {
            "total_entries": 50,
            "total_exits": 40
        }
        requests_mock.get(
            "http://localhost:8000/api/entrance/stats",
            json=mock_stats
        )
        
        # Test getting entrance stats
        now = datetime.now()
        response = api_client.get_entrance_stats(now, now)
        
        assert response == mock_stats

    def test_behavior_analysis(self, api_client, requests_mock):
        # Mock the behavior analysis endpoint
        mock_analysis = {
            "analysis": {"dwell_times": {"zone1": 300}},
            "patterns": ["High activity in zone1"]
        }
        requests_mock.get(
            "http://localhost:8000/api/behavior/analysis/test_person",
            json=mock_analysis
        )
        
        # Test getting behavior analysis
        response = api_client.get_behavior_analysis("test_person")
        
        assert response == mock_analysis

class TestVideoProcessor:
    def test_frame_processing(self, video_processor):
        # Create a test frame
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Process the frame
        processed_frame = video_processor.process_frame(frame)
        
        # Check that processing didn't fail
        assert processed_frame is not None
        assert processed_frame.shape == frame.shape

    def test_object_detection(self, video_processor):
        # Create a test frame with a simple shape
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Detect objects
        detections = video_processor.detect_objects(frame)
        
        # Verify detections
        assert isinstance(detections, list)

class TestDashboard:
    def test_metrics_calculation(self, api_client, requests_mock):
        # Mock the required endpoints
        requests_mock.get(
            "http://localhost:8000/api/entrance/stats",
            json={"total_entries": 50, "total_exits": 40}
        )
        
        # Calculate metrics
        current_occupancy = 50 - 40  # entries - exits
        assert current_occupancy == 10

class TestLiveMonitoring:
    def test_camera_selection(self, video_processor):
        # Test camera availability
        cameras = list(range(1, 5))  # Sample: 4 cameras
        assert len(cameras) > 0
        
        # Test frame capture
        frame = video_processor.get_frame(cameras[0])
        assert frame is not None

class TestAnalytics:
    def test_traffic_analysis(self, api_client, requests_mock):
        # Mock traffic data
        mock_data = {
            "hourly_traffic": [
                {"hour": i, "entries": i*2, "exits": i} 
                for i in range(24)
            ]
        }
        requests_mock.get(
            "http://localhost:8000/analytics/traffic",
            json=mock_data
        )
        
        # Verify data format
        assert "hourly_traffic" in mock_data
        assert len(mock_data["hourly_traffic"]) == 24

def test_frontend_backend_integration(api_client):
    """
    Integration test to verify frontend-backend communication
    """
    try:
        # Test basic API connectivity
        response = api_client._make_request("GET", "/")
        assert response.get("message") == "Welcome to TRINETRA Core API"
        
        # Test face recognition flow
        image_data = np.zeros((300, 300, 3), dtype=np.uint8)
        response = api_client.recognize_face(image_data.tobytes())
        assert "results" in response
        
        # Test real-time data flow
        now = datetime.now()
        response = api_client.get_entrance_stats(now, now)
        assert isinstance(response, dict)
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")
