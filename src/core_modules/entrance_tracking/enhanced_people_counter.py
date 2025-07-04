"""
Enhanced People Counter with Streaming Support and Remote Video Sources
Supports various video sources including URLs, RTSP streams, and local files
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import json
import os
import requests
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingVideoSource:
    """Manages different types of video sources"""
    
    def __init__(self):
        self.temp_files = []
        self.stream_cache = {}
    
    def get_video_source(self, source: Union[str, int], source_type: str = "auto") -> Union[str, int, None]:
        """Process different types of video sources"""
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        if source_type == "url" and isinstance(source, str):
            return self._handle_url_source(source)
        elif source_type == "rtsp":
            return source  # RTSP streams are handled directly by OpenCV
        elif source_type == "file":
            return source if os.path.exists(source) else None
        elif source_type == "webcam":
            return source
        else:
            return source
    
    def _detect_source_type(self, source: Union[str, int]) -> str:
        """Auto-detect source type"""
        if isinstance(source, int):
            return "webcam"
        elif isinstance(source, str):
            if source.startswith("rtsp://") or source.startswith("rtmp://"):
                return "rtsp"
            elif source.startswith("http"):
                return "url"
            elif os.path.isfile(source):
                return "file"
        return "unknown"
    
    def _handle_url_source(self, url: str) -> Optional[str]:
        """Handle URL video sources"""
        try:
            # For direct video file URLs, try to download
            if url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                return self._download_video(url)
            else:
                # For streaming URLs, return as-is
                return url
        except Exception as e:
            logger.error(f"Error handling URL source: {e}")
            return None
    
    def _download_video(self, url: str) -> Optional[str]:
        """Download video from URL for local processing"""
        try:
            logger.info(f"Downloading video from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                self.temp_files.append(temp_file.name)
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
                
                temp_file.close()
                logger.info(f"Video downloaded to: {temp_file.name}")
                return temp_file.name
            else:
                logger.error(f"Failed to download video: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.error(f"Error cleaning up {temp_file}: {e}")

class EnhancedPeopleCounter:
    def __init__(self, video_source=None, enable_streaming: bool = True):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        
        # Counting variables
        self.entrance_line = None
        self.people_in = 0
        self.people_out = 0
        self.tracked_ids = defaultdict(lambda: {
            "last_y": None, 
            "counted": False,
            "direction": None,
            "last_update": None,
            "trajectory": []
        })
        
        # Streaming support
        self.enable_streaming = enable_streaming
        self.video_manager = StreamingVideoSource() if enable_streaming else None
        
        # Video source handling
        self.original_source = video_source
        self.video_source = self._process_video_source(video_source)
        self.cap = None
        self.should_loop = self._should_loop_video()
        
        # Analytics
        self.last_cleanup = datetime.now()
        self.frame_count = 0
        self.hourly_counts = defaultdict(lambda: {"in": 0, "out": 0})
        self.daily_stats = {"date": datetime.now().date(), "total_in": 0, "total_out": 0}
        
        # Demo video URLs for testing
        self.demo_videos = [
            "https://sample-videos.com/zip/10/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        ]
        
        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
    
    def _process_video_source(self, source):
        """Process video source through streaming manager"""
        if source is None:
            return self._get_default_video()
        
        if self.video_manager:
            processed = self.video_manager.get_video_source(source)
            return processed if processed is not None else self._get_default_video()
        
        return source
    
    def _get_default_video(self):
        """Get default video source"""
        try:
            # Try config file first
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'dataset_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    entrance_videos = config.get('datasets', {}).get('entrance_videos', {}).get('sources', [])
                    if entrance_videos:
                        return entrance_videos[0]
            
            # Try test video
            test_video = os.path.join(os.path.dirname(__file__), '..', '..', 'test_videos', 'entrance.mp4')
            if os.path.exists(test_video):
                return test_video
            
            # Use demo video URL if streaming is enabled
            if self.enable_streaming and self.demo_videos:
                logger.info("Using demo video URL")
                return self.demo_videos[0]
            
            # Fallback to webcam
            logger.info("Falling back to webcam")
            return 0
            
        except Exception as e:
            logger.error(f"Error getting default video: {e}")
            return 0
    
    def _should_loop_video(self):
        """Determine if video should loop"""
        if isinstance(self.video_source, str):
            return self.video_source.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        return False
    
    def initialize_video_capture(self):
        """Initialize video capture with error handling"""
        try:
            logger.info(f"Initializing video capture with source: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                logger.error("Failed to open video source")
                return False
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Set entrance line (middle of the frame)
            self.entrance_line = height // 2
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            return False
    
    def detect_people(self, frame):
        """Detect people in frame using YOLO"""
        try:
            results = self.model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            
            # Filter for person class (class_id = 0 in COCO dataset)
            if len(detections.class_id) > 0:
                mask = np.array([class_id == 0 for class_id in detections.class_id], dtype=bool)
                detections = detections[mask]
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in people detection: {e}")
            return sv.Detections.empty()
    
    def track_people(self, detections, frame):
        """Track people across frames"""
        try:
            detections = self.tracker.update(detections=detections, frame=frame)
            return detections
        except Exception as e:
            logger.error(f"Error in people tracking: {e}")
            return detections
    
    def count_people(self, detections, timestamp):
        """Count people crossing the entrance line"""
        current_time = datetime.now()
        hour_key = current_time.hour
        
        for detection_id, bbox in zip(detections.tracker_id, detections.xyxy):
            if detection_id is not None:
                x1, y1, x2, y2 = bbox
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2
                
                track_info = self.tracked_ids[detection_id]
                track_info['trajectory'].append((center_x, center_y, timestamp))
                track_info['last_update'] = current_time
                
                # Count logic
                if track_info['last_y'] is not None and not track_info['counted']:
                    if track_info['last_y'] < self.entrance_line and center_y >= self.entrance_line:
                        # Person entered
                        self.people_in += 1
                        self.hourly_counts[hour_key]["in"] += 1
                        track_info['counted'] = True
                        track_info['direction'] = 'in'
                        logger.info(f"Person {detection_id} entered. Total in: {self.people_in}")
                        
                    elif track_info['last_y'] > self.entrance_line and center_y <= self.entrance_line:
                        # Person exited
                        self.people_out += 1
                        self.hourly_counts[hour_key]["out"] += 1
                        track_info['counted'] = True
                        track_info['direction'] = 'out'
                        logger.info(f"Person {detection_id} exited. Total out: {self.people_out}")
                
                track_info['last_y'] = center_y
    
    def draw_annotations(self, frame, detections):
        """Draw bounding boxes and counting information"""
        # Draw entrance line
        if self.entrance_line:
            cv2.line(frame, (0, self.entrance_line), (frame.shape[1], self.entrance_line), (0, 255, 255), 3)
            cv2.putText(frame, "ENTRANCE LINE", (10, self.entrance_line - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw bounding boxes and IDs
        for detection_id, bbox in zip(detections.tracker_id, detections.xyxy):
            if detection_id is not None:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get direction and color
                track_info = self.tracked_ids[detection_id]
                direction = track_info.get('direction', 'unknown')
                
                if direction == 'in':
                    color = (0, 255, 0)  # Green for entering
                elif direction == 'out':
                    color = (0, 0, 255)  # Red for exiting
                else:
                    color = (255, 255, 0)  # Yellow for tracking
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and direction
                label = f"ID: {detection_id}"
                if direction != 'unknown':
                    label += f" ({direction})"
                
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw count information
        self._draw_count_info(frame)
        
        return frame
    
    def _draw_count_info(self, frame):
        """Draw counting statistics on frame"""
        # Count display
        cv2.putText(frame, f"IN: {self.people_in}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.people_out}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"NET: {self.people_in - self.people_out}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # FPS display
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Source info
        source_info = f"Source: {type(self.original_source).__name__}"
        cv2.putText(frame, source_info, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def cleanup_old_tracks(self):
        """Remove old tracking data to prevent memory issues"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=5)
        
        tracks_to_remove = []
        for track_id, track_info in self.tracked_ids.items():
            if track_info['last_update'] and track_info['last_update'] < cutoff_time:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_ids[track_id]
        
        if tracks_to_remove:
            logger.info(f"Cleaned up {len(tracks_to_remove)} old tracks")
    
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_statistics(self):
        """Get counting statistics"""
        current_time = datetime.now()
        
        return {
            'people_in': self.people_in,
            'people_out': self.people_out,
            'net_count': self.people_in - self.people_out,
            'hourly_counts': dict(self.hourly_counts),
            'active_tracks': len(self.tracked_ids),
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'source': str(self.original_source),
            'streaming_enabled': self.enable_streaming,
            'timestamp': current_time.isoformat()
        }
    
    def save_statistics(self):
        """Save statistics to file"""
        stats = self.get_statistics()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"people_counting_stats_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Statistics saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    def process_frame(self, frame):
        """Process a single frame"""
        if frame is None:
            return None
        
        self.frame_count += 1
        timestamp = time.time()
        
        # Detect and track people
        detections = self.detect_people(frame)
        detections = self.track_people(detections, frame)
        
        # Count people crossing the line
        self.count_people(detections, timestamp)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame, detections)
        
        # Update FPS
        self.update_fps()
        
        # Periodic cleanup
        if self.frame_count % 300 == 0:  # Every ~10 seconds at 30 FPS
            self.cleanup_old_tracks()
        
        return annotated_frame
    
    def run(self):
        """Main processing loop"""
        if not self.initialize_video_capture():
            return
        
        logger.info("Starting people counting...")
        logger.info(f"Video source: {self.video_source}")
        logger.info("Press 'q' to quit, 's' to save statistics")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.should_loop:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.info("Looping video")
                        continue
                    else:
                        logger.info("End of video/stream")
                        break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    cv2.imshow('Enhanced People Counter', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_statistics()
                elif key == ord('r'):
                    # Reset counts
                    self.people_in = 0
                    self.people_out = 0
                    self.tracked_ids.clear()
                    logger.info("Counts reset")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        
        if self.video_manager:
            self.video_manager.cleanup()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

# Backward compatibility
PeopleCounter = EnhancedPeopleCounter

def main():
    """Demo usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced People Counter")
    parser.add_argument("--source", default=None, help="Video source (file, URL, webcam index)")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming features")
    
    args = parser.parse_args()
    
    counter = EnhancedPeopleCounter(
        video_source=args.source,
        enable_streaming=not args.no_streaming
    )
    
    counter.run()

if __name__ == "__main__":
    main()
