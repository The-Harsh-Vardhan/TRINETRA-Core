import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import sys
sys.path.append('../..')
from core_modules.entrance_tracking.people_counter import PeopleCounter
from core_modules.face_recognition.face_recognition_main import FaceRecognition

class LiveMonitoring:
    def __init__(self):
        self.people_counter = PeopleCounter()
        self.face_recognition = FaceRecognition()
        
    def process_video_feed(self, video_source):
        """Process video feed for people counting and face recognition"""
        frame = self.get_video_frame(video_source)
        if frame is not None:
            # People counting
            count_frame = self.people_counter.process_frame(frame)
            # Face recognition
            face_frame = self.face_recognition.process_frame(frame)
            return count_frame, face_frame
        return None, None
    
    @staticmethod
    def get_video_frame(video_source):
        """Get frame from video source"""
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        return None

def display_live_monitoring():
    st.title("Live Monitoring")
    
    # Camera selection
    camera_sources = ["Camera 1", "Camera 2", "Camera 3"]
    selected_camera = st.selectbox("Select Camera", camera_sources)
    
    # Initialize monitoring
    monitoring = LiveMonitoring()
    
    # Create two columns for different views
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("People Counter View")
        # Placeholder for people counter feed
        st.image("https://via.placeholder.com/400x300", caption="People Counter Feed")
        
        # Counter metrics
        metrics = {
            "Current Count": 25,
            "Total Today": 450,
            "Peak Hour": "14:00-15:00"
        }
        
        for label, value in metrics.items():
            st.metric(label, value)
    
    with col2:
        st.subheader("Face Recognition View")
        # Placeholder for face recognition feed
        st.image("https://via.placeholder.com/400x300", caption="Face Recognition Feed")
        
        # Recent detections
        st.subheader("Recent Detections")
        detections = [
            {"time": "14:35:22", "person_id": "P123", "confidence": "95%"},
            {"time": "14:35:18", "person_id": "P456", "confidence": "92%"},
            {"time": "14:35:15", "person_id": "P789", "confidence": "88%"}
        ]
        
        for detection in detections:
            st.text(f"Time: {detection['time']}")
            st.text(f"Person ID: {detection['person_id']}")
            st.text(f"Confidence: {detection['confidence']}")
            st.divider()

def main():
    st.set_page_config(
        page_title="TRINETRA-Core Live Monitoring",
        page_icon="📹",
        layout="wide"
    )
    display_live_monitoring()

if __name__ == "__main__":
    main()
