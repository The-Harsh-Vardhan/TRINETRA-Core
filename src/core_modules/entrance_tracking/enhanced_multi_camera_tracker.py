"""
Enhanced Multi-Camera Tracker with Streaming and Remote Video Support
Supports YouTube videos, RTSP streams, and remote video URLs
"""

import os
import cv2
import numpy as np
from collections import defaultdict
import time
from ultralytics import YOLO
import supervision as sv
from deepface import DeepFace
import requests
import tempfile
import threading
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingVideoManager:
    """Manages streaming video sources and remote video URLs"""
    
    def __init__(self):
        self.temp_files = []
        self.stream_cache = {}
    
    def download_video_from_url(self, url: str, cache_duration: int = 3600) -> str:
        """Download video from URL and cache temporarily"""
        try:
            cache_key = f"video_{hash(url)}"
            
            # Check if already cached
            if cache_key in self.stream_cache:
                cache_info = self.stream_cache[cache_key]
                if (datetime.now() - cache_info['timestamp']).seconds < cache_duration:
                    return cache_info['path']
            
            logger.info(f"Downloading video from URL: {url}")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                self.temp_files.append(temp_file.name)
                
                # Download in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                
                # Cache info
                self.stream_cache[cache_key] = {
                    'path': temp_file.name,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Video downloaded successfully: {temp_file.name}")
                return temp_file.name
            else:
                logger.error(f"Failed to download video: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video from URL: {e}")
            return None
    
    def get_youtube_stream_url(self, youtube_url: str) -> str:
        """Get direct stream URL from YouTube (requires yt-dlp)"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'best[height<=720]',  # Get best quality up to 720p
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info['url']
                
        except ImportError:
            logger.warning("yt-dlp not installed. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            logger.error(f"Error extracting YouTube stream: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {e}")

class EnhancedMultiCameraTracker:
    def __init__(self, enable_streaming: bool = True):
        self.model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        self.camera_streams = {}
        self.customer_tracks = defaultdict(list)
        self.face_embeddings = {}
        self.enable_streaming = enable_streaming
        self.streaming_manager = StreamingVideoManager() if enable_streaming else None
        self.processing_threads = {}
        self.running = False
        
        # Demo video URLs for testing
        self.demo_videos = [
            "https://sample-videos.com/zip/10/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "https://file-examples.com/storage/fe96aeb6ba1e57bf8babc5e/2017/10/file_example_AVI_1280_1_5MG.avi"
        ]
    
    def add_camera(self, camera_id: str, source: Any, source_type: str = "auto"):
        """
        Add a camera stream with enhanced source type detection
        
        Args:
            camera_id: Unique identifier for the camera
            source: Video source (file, URL, RTSP stream, webcam index)
            source_type: "auto", "file", "url", "rtsp", "youtube", "webcam"
        """
        processed_source = self._process_source(source, source_type)
        
        if processed_source is not None:
            self.camera_streams[camera_id] = {
                'source': processed_source,
                'original_source': source,
                'source_type': source_type,
                'capture': None,
                'active_tracks': {},
                'last_frame': None,
                'frame_count': 0,
                'fps': 30,
                'thread': None
            }
            logger.info(f"Camera {camera_id} added with source: {source}")
        else:
            logger.error(f"Failed to process source for camera {camera_id}: {source}")
    
    def _process_source(self, source: Any, source_type: str) -> Any:
        """Process different types of video sources"""
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        if source_type == "youtube" and self.streaming_manager:
            stream_url = self.streaming_manager.get_youtube_stream_url(source)
            return stream_url if stream_url else source
        
        elif source_type == "url" and self.streaming_manager:
            # For direct video URLs, try to download if it's a video file
            if source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                downloaded_path = self.streaming_manager.download_video_from_url(source)
                return downloaded_path if downloaded_path else source
            return source
        
        elif source_type in ["file", "rtsp", "webcam"]:
            return source
        
        return source
    
    def _detect_source_type(self, source: Any) -> str:
        """Auto-detect source type"""
        if isinstance(source, int):
            return "webcam"
        elif isinstance(source, str):
            if source.startswith("rtsp://"):
                return "rtsp"
            elif "youtube.com" in source or "youtu.be" in source:
                return "youtube"
            elif source.startswith("http"):
                return "url"
            elif os.path.isfile(source):
                return "file"
        return "unknown"
    
    def initialize_cameras(self):
        """Initialize all camera captures"""
        for camera_id, camera_info in self.camera_streams.items():
            try:
                cap = cv2.VideoCapture(camera_info['source'])
                if cap.isOpened():
                    camera_info['capture'] = cap
                    camera_info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30
                    logger.info(f"Camera {camera_id} initialized successfully")
                else:
                    logger.error(f"Failed to open camera {camera_id}")
            except Exception as e:
                logger.error(f"Error initializing camera {camera_id}: {e}")
    
    def get_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face embedding from detected person"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            embedding = DeepFace.represent(face_img, model_name="Facenet")[0]["embedding"]
            return np.array(embedding)
            
        except Exception as e:
            logger.debug(f"Error extracting face embedding: {e}")
            return None

    def match_person_across_cameras(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> bool:
        """Match person across different cameras using face embeddings"""
        if embedding1 is None or embedding2 is None:
            return False
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance < threshold

    def update_customer_journey(self, camera_id: str, person_id: int, position: Tuple[float, float], timestamp: float):
        """Update customer journey with new position"""
        journey_point = {
            'camera_id': camera_id,
            'position': position,
            'timestamp': timestamp,
            'zone': self._get_zone_from_position(position)
        }
        self.customer_tracks[person_id].append(journey_point)
    
    def _get_zone_from_position(self, position: Tuple[float, float]) -> str:
        """Determine zone based on position (can be customized)"""
        x, y = position
        if x < 300:
            return "entrance"
        elif x > 600:
            return "exit"
        else:
            return "main_area"

    def process_frame(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        """Process frame from a single camera with enhanced tracking"""
        if frame is None:
            return None
        
        try:
            # Detect people using YOLO
            results = self.model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            
            # Filter for person class (class_id = 0 in COCO dataset)
            if len(detections.class_id) > 0:
                mask = np.array([class_id == 0 for class_id in detections.class_id], dtype=bool)
                detections = detections[mask]

            # Track detections
            detections = self.tracker.update(detections=detections, frame=frame)

            # Process each detection
            for detection_id, bbox in zip(detections.tracker_id, detections.xyxy):
                if detection_id is not None:
                    # Get face embedding for new tracks
                    if detection_id not in self.face_embeddings:
                        embedding = self.get_face_embedding(frame, bbox)
                        if embedding is not None:
                            self.face_embeddings[detection_id] = embedding

                    # Update position
                    center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                    self.update_customer_journey(camera_id, detection_id, center, time.time())

                    # Draw bounding box with enhanced info
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display tracking info
                    label = f"ID: {detection_id}"
                    zone = self._get_zone_from_position(center)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Zone: {zone}", (x1, y1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Add camera info overlay
            info_text = f"Camera: {camera_id} | Detections: {len(detections.tracker_id) if detections.tracker_id is not None else 0}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
            return frame

    def process_camera_stream(self, camera_id: str):
        """Process individual camera stream in separate thread"""
        camera_info = self.camera_streams[camera_id]
        cap = camera_info['capture']
        
        while self.running and cap and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    if camera_info['source_type'] == 'file':
                        # Loop video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Process frame
                processed_frame = self.process_frame(camera_id, frame)
                camera_info['last_frame'] = processed_frame
                camera_info['frame_count'] += 1
                
                # Control frame rate
                time.sleep(1.0 / camera_info['fps'])
                
            except Exception as e:
                logger.error(f"Error in camera {camera_id} processing thread: {e}")
                break
    
    def start_streaming(self):
        """Start processing all camera streams"""
        self.running = True
        self.initialize_cameras()
        
        # Start processing threads for each camera
        for camera_id in self.camera_streams:
            thread = threading.Thread(target=self.process_camera_stream, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.processing_threads[camera_id] = thread
            logger.info(f"Started processing thread for camera {camera_id}")
    
    def stop_streaming(self):
        """Stop all camera streams"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads.values():
            thread.join(timeout=2.0)
        
        # Cleanup cameras
        for camera_info in self.camera_streams.values():
            if camera_info['capture']:
                camera_info['capture'].release()
        
        # Cleanup streaming manager
        if self.streaming_manager:
            self.streaming_manager.cleanup()
        
        cv2.destroyAllWindows()
        logger.info("All camera streams stopped")

    def run_display(self):
        """Main display loop"""
        self.start_streaming()
        
        try:
            while self.running:
                display_frames = []
                
                # Collect frames from all cameras
                for camera_id, camera_info in self.camera_streams.items():
                    frame = camera_info.get('last_frame')
                    if frame is not None:
                        # Resize frame for display
                        height, width = frame.shape[:2]
                        if width > 640:
                            scale = 640 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        display_frames.append((camera_id, frame))
                
                # Display frames
                if display_frames:
                    if len(display_frames) == 1:
                        cv2.imshow(f'Camera {display_frames[0][0]}', display_frames[0][1])
                    else:
                        # Create grid layout for multiple cameras
                        rows = int(np.ceil(np.sqrt(len(display_frames))))
                        cols = int(np.ceil(len(display_frames) / rows))
                        
                        grid_frames = []
                        for i in range(rows):
                            row_frames = []
                            for j in range(cols):
                                idx = i * cols + j
                                if idx < len(display_frames):
                                    row_frames.append(display_frames[idx][1])
                                else:
                                    # Add blank frame
                                    h, w = display_frames[0][1].shape[:2]
                                    row_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                            
                            if row_frames:
                                grid_frames.append(np.hstack(row_frames))
                        
                        if grid_frames:
                            combined_frame = np.vstack(grid_frames)
                            cv2.imshow('Multi-Camera View', combined_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analytics
                    self.save_analytics()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop_streaming()

    def get_customer_journey(self, person_id: int) -> List[Dict]:
        """Retrieve the complete journey of a specific customer"""
        return self.customer_tracks.get(person_id, [])

    def generate_heatmap(self, camera_id: str, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Generate heatmap of customer movements for a specific camera"""
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for tracks in self.customer_tracks.values():
            for point in tracks:
                if point['camera_id'] == camera_id:
                    x, y = map(int, point['position'])
                    # Ensure coordinates are within bounds
                    if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                        cv2.circle(heatmap, (x, y), 20, (1,), -1)

        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        return {
            'total_tracks': len(self.customer_tracks),
            'cameras': list(self.camera_streams.keys()),
            'face_embeddings': len(self.face_embeddings),
            'streaming_enabled': self.enable_streaming,
            'active_cameras': sum(1 for info in self.camera_streams.values() 
                                if info.get('capture') and info['capture'].isOpened())
        }
    
    def save_analytics(self):
        """Save analytics to file"""
        analytics = self.get_analytics()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tracking_analytics_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(analytics, f, indent=2, default=str)
            logger.info(f"Analytics saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")

def main():
    """Demo usage"""
    tracker = EnhancedMultiCameraTracker(enable_streaming=True)
    
    # Add demo cameras
    tracker.add_camera("webcam", 0, "webcam")
    
    # Add demo video URLs (uncomment to test)
    # tracker.add_camera("demo1", tracker.demo_videos[0], "url")
    
    # Add YouTube video (uncomment and provide valid URL to test)
    # tracker.add_camera("youtube", "https://www.youtube.com/watch?v=example", "youtube")
    
    # Add RTSP stream (uncomment and provide valid stream to test)
    # tracker.add_camera("rtsp1", "rtsp://camera_ip:port/stream", "rtsp")
    
    logger.info("Starting enhanced multi-camera tracker...")
    tracker.run_display()

if __name__ == "__main__":
    main()
