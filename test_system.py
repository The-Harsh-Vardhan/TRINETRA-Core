import cv2
import numpy as np
from src.core_modules.face_recognition.face_recognition_main import FaceRecognitionSystem
from src.core_modules.entrance_tracking.people_counter import PeopleCounter
from src.core_modules.behavioral_insights.behavior_analytics import BehaviorAnalytics
import asyncio
from datetime import datetime

async def test_face_recognition():
    print("\nTesting Face Recognition Module...")
    try:
        face_system = FaceRecognitionSystem()
        # Test with a sample image
        img = cv2.imread('src/test_videos/sample.jpg')
        if img is None:
            print("⚠️ Test image not found")
            return
        
        results = face_system.process_frame(img)
        print("✅ Face recognition module working")
        print(f"Found {len(results)} faces in test image")
    except Exception as e:
        print(f"❌ Face recognition test failed: {str(e)}")

async def test_people_counter():
    print("\nTesting People Counter Module...")
    try:
        counter = PeopleCounter()
        # Test with a sample video frame
        img = cv2.imread('src/test_videos/sample.jpg')
        if img is None:
            print("⚠️ Test image not found")
            return
            
        results = counter.process_frame(img)
        print("✅ People counter module working")
        print(f"Detected {counter.people_in} entries and {counter.people_out} exits")
    except Exception as e:
        print(f"❌ People counter test failed: {str(e)}")

async def test_behavior_analytics():
    print("\nTesting Behavior Analytics Module...")
    try:
        analytics = BehaviorAnalytics(1920, 1080)
        
        # Test zone definition
        analytics.define_zone("entrance", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        
        # Test trajectory tracking
        analytics.update_trajectory("person1", 50, 50, datetime.now())
        analytics.update_trajectory("person1", 60, 60, datetime.now())
        
        # Get analytics
        zone_stats = analytics.get_zone_analytics("person1")
        patterns = analytics.detect_patterns("person1")
        
        print("✅ Behavior analytics module working")
        print("Zone statistics:", zone_stats)
        print("Detected patterns:", patterns)
    except Exception as e:
        print(f"❌ Behavior analytics test failed: {str(e)}")

async def main():
    print("Starting TRINETRA-Core System Tests...")
    
    # Test core modules
    await test_face_recognition()
    await test_people_counter()
    await test_behavior_analytics()
    
    print("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(main())
