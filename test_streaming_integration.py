"""
Integration Test for Enhanced TRINETRA Core with Streaming Capabilities
Tests the new streaming features for face recognition, entrance tracking, and behavioral analytics
"""

import unittest
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core_modules.face_recognition.enhanced_face_recognition import EnhancedFaceRecognitionSystem
from src.core_modules.entrance_tracking.enhanced_people_counter import EnhancedPeopleCounter
from src.core_modules.entrance_tracking.enhanced_multi_camera_tracker import EnhancedMultiCameraTracker
from src.core_modules.behavioral_insights.enhanced_behavior_analytics import EnhancedBehaviorAnalytics

class TestStreamingCapabilities(unittest.TestCase):
    """Test streaming capabilities of enhanced modules"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_size = (640, 480)
        self.test_video_path = self._create_test_video()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_video(self):
        """Create a test video file"""
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, self.test_image_size)
        
        for i in range(30):  # 3 seconds at 10 fps
            # Create a frame with moving rectangle (simulating person)
            frame = np.zeros((self.test_image_size[1], self.test_image_size[0], 3), dtype=np.uint8)
            x = int(50 + i * 10)  # Moving rectangle
            y = 200
            cv2.rectangle(frame, (x, y), (x+50, y+100), (255, 255, 255), -1)
            out.write(frame)
        
        out.release()
        return video_path
    
    def _create_test_image(self):
        """Create a test image with a face-like rectangle"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Add a face-like rectangle
        cv2.rectangle(image, (250, 150), (350, 250), (255, 200, 180), -1)
        return image

class TestEnhancedFaceRecognition(TestStreamingCapabilities):
    """Test enhanced face recognition with streaming"""
    
    def test_initialization_with_streaming(self):
        """Test face recognition system initialization with streaming enabled"""
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=True)
        
        self.assertTrue(face_system.enable_streaming)
        self.assertIsNotNone(face_system.streaming_manager)
        self.assertIsInstance(face_system.known_faces, dict)
    
    def test_initialization_without_streaming(self):
        """Test face recognition system initialization with streaming disabled"""
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=False)
        
        self.assertFalse(face_system.enable_streaming)
        self.assertIsNone(face_system.streaming_manager)
    
    def test_face_processing(self):
        """Test face processing functionality"""
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=False)
        test_image = self._create_test_image()
        
        # Test process_frame
        results, processed_frame = face_system.process_frame(test_image)
        
        self.assertIsInstance(results, list)
        self.assertIsNotNone(processed_frame)
        self.assertEqual(processed_frame.shape, test_image.shape)
    
    def test_statistics_generation(self):
        """Test statistics generation"""
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=False)
        stats = face_system.get_statistics()
        
        self.assertIn('total_faces', stats)
        self.assertIn('streaming_faces', stats)
        self.assertIn('local_faces', stats)
        self.assertIn('database_path', stats)
        self.assertIn('streaming_enabled', stats)

class TestEnhancedPeopleCounter(TestStreamingCapabilities):
    """Test enhanced people counter with streaming"""
    
    def test_initialization_with_streaming(self):
        """Test people counter initialization with streaming"""
        counter = EnhancedPeopleCounter(enable_streaming=True)
        
        self.assertTrue(counter.enable_streaming)
        self.assertIsNotNone(counter.video_manager)
    
    def test_initialization_with_video_file(self):
        """Test initialization with local video file"""
        counter = EnhancedPeopleCounter(video_source=self.test_video_path, enable_streaming=True)
        
        self.assertIsNotNone(counter.video_source)
        self.assertTrue(counter.should_loop)
    
    def test_video_capture_initialization(self):
        """Test video capture initialization"""
        counter = EnhancedPeopleCounter(video_source=self.test_video_path, enable_streaming=False)
        success = counter.initialize_video_capture()
        
        # Should succeed with test video
        self.assertTrue(success)
        self.assertIsNotNone(counter.cap)
        
        # Clean up
        counter.cleanup()
    
    def test_people_detection(self):
        """Test people detection functionality"""
        counter = EnhancedPeopleCounter(enable_streaming=False)
        test_frame = self._create_test_image()
        
        detections = counter.detect_people(test_frame)
        
        # Should return detections object
        self.assertIsNotNone(detections)
    
    def test_frame_processing(self):
        """Test frame processing"""
        counter = EnhancedPeopleCounter(enable_streaming=False)
        test_frame = self._create_test_image()
        
        processed_frame = counter.process_frame(test_frame)
        
        self.assertIsNotNone(processed_frame)
        self.assertEqual(processed_frame.shape, test_frame.shape)
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        counter = EnhancedPeopleCounter(enable_streaming=False)
        stats = counter.get_statistics()
        
        required_keys = ['people_in', 'people_out', 'net_count', 'active_tracks', 
                        'current_fps', 'frame_count', 'source', 'streaming_enabled']
        
        for key in required_keys:
            self.assertIn(key, stats)

class TestEnhancedMultiCameraTracker(TestStreamingCapabilities):
    """Test enhanced multi-camera tracker"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = EnhancedMultiCameraTracker(enable_streaming=True)
        
        self.assertTrue(tracker.enable_streaming)
        self.assertIsNotNone(tracker.streaming_manager)
        self.assertIsInstance(tracker.camera_streams, dict)
    
    def test_camera_addition(self):
        """Test adding cameras"""
        tracker = EnhancedMultiCameraTracker(enable_streaming=False)
        
        # Add webcam
        tracker.add_camera("test_cam", 0, "webcam")
        self.assertIn("test_cam", tracker.camera_streams)
        
        # Add file source
        tracker.add_camera("file_cam", self.test_video_path, "file")
        self.assertIn("file_cam", tracker.camera_streams)
    
    def test_source_type_detection(self):
        """Test automatic source type detection"""
        tracker = EnhancedMultiCameraTracker(enable_streaming=False)
        
        # Test different source types
        self.assertEqual(tracker._detect_source_type(0), "webcam")
        self.assertEqual(tracker._detect_source_type("test.mp4"), "file")
        self.assertEqual(tracker._detect_source_type("rtsp://test"), "rtsp")
        self.assertEqual(tracker._detect_source_type("http://test.mp4"), "url")
    
    def test_frame_processing(self):
        """Test frame processing"""
        tracker = EnhancedMultiCameraTracker(enable_streaming=False)
        test_frame = self._create_test_image()
        
        processed_frame = tracker.process_frame("test_cam", test_frame)
        
        self.assertIsNotNone(processed_frame)
    
    def test_analytics_generation(self):
        """Test analytics generation"""
        tracker = EnhancedMultiCameraTracker(enable_streaming=False)
        tracker.add_camera("test_cam", 0, "webcam")
        
        analytics = tracker.get_analytics()
        
        self.assertIn('total_tracks', analytics)
        self.assertIn('cameras', analytics)
        self.assertIn('streaming_enabled', analytics)

class TestEnhancedBehaviorAnalytics(TestStreamingCapabilities):
    """Test enhanced behavior analytics"""
    
    def test_initialization(self):
        """Test behavior analytics initialization"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=True)
        
        self.assertTrue(analytics.enable_streaming)
        self.assertIsNotNone(analytics.streaming_data)
        self.assertEqual(analytics.frame_width, 640)
        self.assertEqual(analytics.frame_height, 480)
    
    def test_zone_definition(self):
        """Test zone definition"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Define a test zone
        zone_points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        analytics.define_zone("test_zone", zone_points)
        
        self.assertIn("test_zone", analytics.zones)
        self.assertTrue(analytics.is_in_zone((150, 150), "test_zone"))
        self.assertFalse(analytics.is_in_zone((50, 50), "test_zone"))
    
    def test_trajectory_update(self):
        """Test trajectory updating"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Add some trajectory points
        for i in range(5):
            analytics.update_trajectory("person_1", 100 + i*10, 100 + i*5, datetime.now())
        
        self.assertIn("person_1", analytics.trajectories)
        self.assertEqual(len(analytics.trajectories["person_1"]), 5)
    
    def test_heatmap_generation(self):
        """Test heatmap generation"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Add some points to generate heatmap
        for i in range(10):
            analytics.update_trajectory("person_1", 100 + i, 100 + i, datetime.now())
        
        heatmap = analytics.get_heatmap()
        
        self.assertEqual(heatmap.shape, (480, 640))
        self.assertTrue(np.any(heatmap > 0))  # Should have some heat
    
    def test_analytics_generation(self):
        """Test analytics generation"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Add some data
        analytics.update_trajectory("person_1", 100, 100, datetime.now())
        
        zone_analytics = analytics.get_zone_analytics()
        crowd_insights = analytics.get_crowd_insights()
        
        self.assertIn("overall_metrics", zone_analytics)
        self.assertIn("occupancy_trends", crowd_insights)
    
    def test_pattern_detection(self):
        """Test pattern detection"""
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Create a trajectory that might show patterns
        base_time = datetime.now()
        for i in range(20):
            # Simulate stationary behavior (loitering pattern)
            analytics.update_trajectory("person_1", 100, 100, 
                                      base_time + timedelta(seconds=i*5))
        
        patterns = analytics.detect_patterns("person_1")
        
        # Should detect some patterns
        self.assertIsInstance(patterns, list)

class TestStreamingIntegration(TestStreamingCapabilities):
    """Test integration between streaming components"""
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with streaming components"""
        # Initialize all components
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=False)
        counter = EnhancedPeopleCounter(enable_streaming=False)
        analytics = EnhancedBehaviorAnalytics(640, 480, enable_streaming=False)
        
        # Create test frame
        test_frame = self._create_test_image()
        
        # Process through each component
        face_results, face_frame = face_system.process_frame(test_frame.copy())
        counter_frame = counter.process_frame(test_frame.copy())
        
        # Add trajectory data to analytics
        analytics.update_trajectory("test_person", 320, 240, datetime.now())
        
        # Verify results
        self.assertIsNotNone(face_frame)
        self.assertIsNotNone(counter_frame)
        self.assertIn("test_person", analytics.trajectories)
    
    def test_performance_benchmarking(self):
        """Test performance of enhanced components"""
        import time
        
        # Test face recognition performance
        face_system = EnhancedFaceRecognitionSystem(enable_streaming=False)
        test_image = self._create_test_image()
        
        start_time = time.time()
        for _ in range(10):
            face_system.process_frame(test_image)
        face_processing_time = time.time() - start_time
        
        # Should process reasonably fast
        self.assertLess(face_processing_time, 30.0)  # 10 frames in less than 30 seconds
        
        print(f"Face recognition: {face_processing_time:.2f}s for 10 frames")

def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("TRINETRA Core Enhanced Streaming Integration Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnhancedFaceRecognition,
        TestEnhancedPeopleCounter,
        TestEnhancedMultiCameraTracker,
        TestEnhancedBehaviorAnalytics,
        TestStreamingIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
