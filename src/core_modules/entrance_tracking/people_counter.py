import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import json
import os
from datetime import datetime, timedelta

class PeopleCounter:
    def __init__(self, video_source=None):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        self.entrance_line = None
        self.people_in = 0
        self.people_out = 0
        self.tracked_ids = defaultdict(lambda: {
            "last_y": None, 
            "counted": False,
            "direction": None,
            "last_update": None
        })
        
        # Video source handling
        self.video_source = video_source if video_source else self.get_default_video()
        self.cap = None
        self.should_loop = self.get_loop_config()
        self.last_cleanup = datetime.now()
        self.frame_count = 0

    def get_default_video(self):
        """Get default video source from config or use test video"""
        try:
            # First try to get from config
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'camera_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                for camera in config['cameras']:
                    if camera['type'] == 'entrance':
                        return camera['source']
            
            # If no entrance camera in config, try test video
            test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_videos', 'entrance.mp4')
            if os.path.exists(test_video):
                return test_video
            
            # Fallback to webcam
            return 0
        except Exception as e:
            print(f"Error loading video source: {e}")
            return 0
            print(f"Error loading camera config: {e}")
            return 0

    def get_loop_config(self):
        try:
            with open('camera_config.json', 'r') as f:
                config = json.load(f)
                return config.get('video_loop', True)
        except:
            return True

    def start_video(self):
        """Initialize video capture"""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")
        return self.cap.isOpened()

    def read_frame(self):
        """Read a frame from video with looping support"""
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret and self.should_loop:
            # Reset video to beginning and clear tracking
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.cleanup_tracks()  # Clear tracking data on loop
            ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            # Cleanup stale tracks every 30 frames
            if self.frame_count % 30 == 0:
                self.cleanup_tracks()
        
        return ret, frame

    def cleanup_tracks(self):
        """Remove stale tracks and cleanup tracking data"""
        current_time = datetime.now()
        stale_threshold = timedelta(seconds=2)  # Remove tracks not updated in 2 seconds
        
        # Find and remove stale tracks
        stale_tracks = []
        for track_id, track_info in self.tracked_ids.items():
            if track_info["last_update"] and (current_time - track_info["last_update"]) > stale_threshold:
                stale_tracks.append(track_id)
        
        # Remove stale tracks
        for track_id in stale_tracks:
            del self.tracked_ids[track_id]
        
        self.last_cleanup = current_time

    def set_entrance_line(self, frame):
        height, width = frame.shape[:2]
        # Set entrance line at the middle of the frame width
        self.entrance_line = int(width * 0.5)

    def process_frame(self, frame):
        """Process a frame and detect/track people"""
        if self.entrance_line is None:
            self.set_entrance_line(frame)

        # Run YOLO detection
        results = self.model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        
        # Filter for person class (class_id = 0 in COCO)
        mask = np.array([class_id == 0 for class_id in detections.class_id], dtype=bool)
        detections = detections[mask]
        
        # Update tracking
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Count people crossing the line
        for detection in tracked_detections:
            track_id = detection.track_id
            bbox = detection.bbox
            # Use center point of bounding box for more reliable tracking
            current_x = (bbox[0] + bbox[2]) / 2  # x coordinate of center of bounding box
            
            # Get tracking history
            track_info = self.tracked_ids[track_id]
            last_x = track_info.get("last_x")
            
            if last_x is not None:
                if not track_info["counted"]:
                    # Check if center point crossed the line
                    if last_x < self.entrance_line and current_x >= self.entrance_line:
                        # Moving right = entering
                        self.people_in += 1
                        track_info["counted"] = True
                        track_info["direction"] = "in"
                    elif last_x > self.entrance_line and current_x <= self.entrance_line:
                        # Moving left = exiting
                        self.people_out += 1
                        track_info["counted"] = True
                        track_info["direction"] = "out"
                else:
                    # Reset counting if person moves back significantly
                    if track_info["direction"] == "in" and current_x < self.entrance_line - 50:
                        track_info["counted"] = False
                    elif track_info["direction"] == "out" and current_x > self.entrance_line + 50:
                        track_info["counted"] = False
            
            # Update tracking history
            track_info["last_x"] = current_x
            track_info["last_update"] = datetime.now()
        
        # Draw detections and line on frame
        frame = self.draw_results(frame, tracked_detections)
        return frame

    def draw_results(self, frame, detections):
        """Draw detection boxes and counting line on frame"""
        # Draw entrance line (vertical)
        height = frame.shape[0]
        cv2.line(frame, (self.entrance_line, 0), (self.entrance_line, height), 
                (0, 255, 0), 2)
        
        # Draw arrow indicators
        arrow_y = height - 50
        # Left arrow (out)
        cv2.arrowedLine(frame, (self.entrance_line - 50, arrow_y), 
                       (self.entrance_line - 100, arrow_y), 
                       (255, 0, 0), 2, tipLength=0.5)
        cv2.putText(frame, "OUT", (self.entrance_line - 100, arrow_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Right arrow (in)
        cv2.arrowedLine(frame, (self.entrance_line + 50, arrow_y),
                       (self.entrance_line + 100, arrow_y),
                       (0, 255, 0), 2, tipLength=0.5)
        cv2.putText(frame, "IN", (self.entrance_line + 70, arrow_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection.bbox.astype(int)
            track_id = detection.track_id
            track_info = self.tracked_ids[track_id]
            
            # Choose color based on counting status
            if track_info["counted"]:
                color = (0, 255, 0) if track_info["direction"] == "in" else (255, 0, 0)
            else:
                color = (255, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw ID and direction
            label = f"ID: {track_id}"
            if "direction" in track_info:
                label += f" ({track_info['direction']})"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_y = int((bbox[1] + bbox[3]) / 2)
            center_x = int((bbox[0] + bbox[2]) / 2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        # Draw counts and status
        cv2.putText(frame, f"Left to Right (IN): {self.people_in}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right to Left (OUT): {self.people_out}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Currently Inside: {self.people_in - self.people_out}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame

    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        try:
            self.start_video()
            
            while True:
                ret, frame = self.read_frame()
                if not ret:
                    break

                # Process frame and get detections
                processed_frame = self.process_frame(frame)
                
                # Draw entrance line
                cv2.line(processed_frame, (0, self.entrance_line), 
                        (processed_frame.shape[1], self.entrance_line), 
                        (0, 255, 0), 2)
                
                # Display counts
                count_text = f"In: {self.people_in} Out: {self.people_out}"
                cv2.putText(processed_frame, count_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('People Counter', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()

def run_counter(video_source=None):
    """Run the people counter on a video source"""
    counter = PeopleCounter(video_source)
    
    try:
        counter.start_video()
        
        while True:
            ret, frame = counter.read_frame()
            if not ret:
                break
                
            # Process frame
            processed_frame = counter.process_frame(frame)
            
            # Display frame
            cv2.imshow('People Counter', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        counter.cleanup()
        
    return counter.people_in, counter.people_out

if __name__ == "__main__":
    # Get video source from command line or use default
    import sys
    video_source = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Starting people counter...")
    print("Press 'q' to quit")
    
    people_in, people_out = run_counter(video_source)
    print(f"\nFinal counts:")
    print(f"People in: {people_in}")
    print(f"People out: {people_out}")
    print(f"Currently inside: {people_in - people_out}")