import cv2
import numpy as np
from typing import Tuple, Optional
import threading
import queue

class VideoProcessor:
    def __init__(self):
        self._frame_queue = queue.Queue(maxsize=30)
        self._stop_flag = threading.Event()
    
    def start_capture(self, source: int = 0) -> None:
        """Start capturing video from the specified source"""
        self._stop_flag.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            args=(source,)
        )
        self.capture_thread.start()
    
    def stop_capture(self) -> None:
        """Stop capturing video"""
        self._stop_flag.set()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
    
    def _capture_frames(self, source: int) -> None:
        """Continuously capture frames from the video source"""
        cap = cv2.VideoCapture(source)
        
        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if ret:
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put(frame)
            else:
                break
        
        cap.release()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the queue"""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    @staticmethod
    def resize_frame(frame: np.ndarray, 
                     target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width/width, target_height/height)
        new_size = (int(width * scale), int(height * scale))
        
        # Resize frame
        return cv2.resize(frame, new_size)
    
    @staticmethod
    def draw_bbox(frame: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  label: str,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """Draw bounding box with label on frame"""
        x, y, w, h = bbox
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - baseline),
            (x + label_size[0], y),
            color,
            cv2.FILLED
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x, y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        return frame
    
    @staticmethod
    def add_timestamp(frame: np.ndarray) -> np.ndarray:
        """Add timestamp to frame"""
        timestamp = cv2.putText(
            frame.copy(),
            f"Time: {cv2.getTickCount() / cv2.getTickFrequency():.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return timestamp
