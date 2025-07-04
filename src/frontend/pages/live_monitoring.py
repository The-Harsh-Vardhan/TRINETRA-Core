import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import sys
sys.path.append('../..')
from core_modules.entrance_tracking.people_counter import PeopleCounter
from core_modules.face_recognition.face_recognition_main import FaceRecognition
from utils.common import StreamlitUtils, ChartUtils, DataUtils

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
    StreamlitUtils.setup_page("Live Monitoring", "📹")
    
    # Camera selection with status indicator
    camera_sources = ["Camera 1", "Camera 2", "Camera 3"]
    selected_camera = st.selectbox("Select Camera", camera_sources)
    
    # Camera status
    StreamlitUtils.display_status_indicator("success", f"{selected_camera} is online and streaming")
    
    # Initialize monitoring
    monitoring = LiveMonitoring()
    
    # Create two columns for different views
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚶 People Counter View")
        # Placeholder for people counter feed
        st.image("https://via.placeholder.com/400x300", caption="People Counter Feed")
        
        # Counter metrics using shared utilities
        counter_metrics = {
            "Current Count": {"value": "25", "delta": "+3"},
            "Total Today": {"value": DataUtils.format_number(450), "delta": "+15%"},
            "Peak Hour": {"value": "14:00-15:00", "delta": None}
        }
        StreamlitUtils.display_metrics_row(counter_metrics, columns=1)
        
        # Real-time traffic chart
        traffic_data = DataUtils.get_mock_data("traffic")[-6:]  # Last 6 hours
        ChartUtils.create_traffic_chart(traffic_data, "Last 6 Hours Traffic")
    
    with col2:
        st.subheader("👤 Face Recognition View")
        # Placeholder for face recognition feed
        st.image("https://via.placeholder.com/400x300", caption="Face Recognition Feed")
        
        # Recognition metrics
        recognition_metrics = {
            "Recognition Rate": {"value": "94.2%", "delta": "+1.2%"},
            "Total Recognized": {"value": DataUtils.format_number(1234), "delta": "+25"},
            "Unknown Faces": {"value": "12", "delta": "-3"}
        }
        StreamlitUtils.display_metrics_row(recognition_metrics, columns=1)
        
        # Recent detections
        st.subheader("🔍 Recent Detections")
        detections = [
            {"time": "14:35:22", "person_id": "P123", "confidence": "95%"},
            {"time": "14:35:18", "person_id": "P456", "confidence": "92%"},
            {"time": "14:35:15", "person_id": "P789", "confidence": "88%"}
        ]
        
        for detection in detections:
            with st.container():
                col_time, col_id, col_conf = st.columns([2, 2, 1])
                with col_time:
                    st.text(f"⏰ {detection['time']}")
                with col_id:
                    st.text(f"👤 {detection['person_id']}")
                with col_conf:
                    st.text(f"📊 {detection['confidence']}")
                st.divider()

def main():
    display_live_monitoring()

if __name__ == "__main__":
    main()
