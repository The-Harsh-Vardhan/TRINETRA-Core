import os
import json
import shutil
from typing import List, Dict, Any
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

class DatasetManager:
    def __init__(self, config_path: str = "config/dataset_config.json"):
        """Initialize Dataset Manager"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create directories if they don't exist
        for dataset_type in self.config:
            if isinstance(self.config[dataset_type], dict) and 'path' in self.config[dataset_type]:
                os.makedirs(self.config[dataset_type]['path'], exist_ok=True)
    
    def add_face_image(self, image_path: str, person_name: str) -> bool:
        """
        Add a face image to the dataset
        Returns: Success status
        """
        config = self.config['face_recognition']
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext not in config['allowed_formats']:
            raise ValueError(f"Unsupported image format. Allowed: {config['allowed_formats']}")
        
        # Create person directory
        person_dir = os.path.join(config['path'], person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Check minimum face size
        if img.shape[0] < config['min_face_size'][0] or img.shape[1] < config['min_face_size'][1]:
            raise ValueError(f"Image too small. Minimum size: {config['min_face_size']}")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = os.path.join(person_dir, f"{timestamp}{ext}")
        shutil.copy2(image_path, new_path)
        
        # Generate augmented images if enabled
        if config['augmentation']['enabled']:
            self._generate_augmented_faces(new_path, person_dir)
        
        return True
    
    def add_entrance_video(self, video_path: str, camera_id: str) -> bool:
        """
        Add an entrance video to the dataset
        Returns: Success status
        """
        config = self.config['entrance_videos']
        ext = os.path.splitext(video_path)[1].lower()
        
        if ext not in config['allowed_formats']:
            raise ValueError(f"Unsupported video format. Allowed: {config['allowed_formats']}")
        
        # Validate video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Check video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if width < config['min_resolution'][0] or height < config['min_resolution'][1]:
            cap.release()
            raise ValueError(f"Video resolution too low. Minimum: {config['min_resolution']}")
        
        if fps < config['fps']:
            cap.release()
            raise ValueError(f"Video FPS too low. Minimum: {config['fps']}")
        
        cap.release()
        
        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = os.path.join(config['path'], f"{camera_id}_{timestamp}{ext}")
        shutil.copy2(video_path, new_path)
        
        return True
    
    def add_behavior_sample(self, video_path: str, tracking_data: Dict[str, Any]) -> bool:
        """
        Add a behavior sample video and its tracking data
        Returns: Success status
        """
        config = self.config['behavior_samples']
        video_ext = os.path.splitext(video_path)[1].lower()
        
        if video_ext not in config['allowed_formats']:
            raise ValueError(f"Unsupported video format. Allowed: {config['allowed_formats']}")
        
        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_new_path = os.path.join(config['path'], f"sample_{timestamp}{video_ext}")
        shutil.copy2(video_path, video_new_path)
        
        # Save tracking data
        tracking_path = os.path.join(
            config['tracking_data'], 
            f"tracking_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(tracking_path), exist_ok=True)
        
        with open(tracking_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
        
        return True
    
    def get_face_samples(self, person_name: str = None) -> List[str]:
        """Get paths of face samples"""
        config = self.config['face_recognition']
        if person_name:
            person_dir = os.path.join(config['path'], person_name)
            if not os.path.exists(person_dir):
                return []
            return [
                os.path.join(person_dir, f) 
                for f in os.listdir(person_dir) 
                if os.path.splitext(f)[1].lower() in config['allowed_formats']
            ]
        else:
            samples = []
            for person in os.listdir(config['path']):
                samples.extend(self.get_face_samples(person))
            return samples
    
    def get_entrance_videos(self, camera_id: str = None) -> List[str]:
        """Get paths of entrance videos"""
        config = self.config['entrance_videos']
        videos = [
            os.path.join(config['path'], f)
            for f in os.listdir(config['path'])
            if os.path.splitext(f)[1].lower() in config['allowed_formats']
        ]
        
        if camera_id:
            return [v for v in videos if camera_id in os.path.basename(v)]
        return videos
    
    def get_behavior_samples(self) -> List[Dict[str, str]]:
        """Get paths of behavior samples with their tracking data"""
        config = self.config['behavior_samples']
        samples = []
        
        for video in os.listdir(config['path']):
            if os.path.splitext(video)[1].lower() not in config['allowed_formats']:
                continue
                
            video_path = os.path.join(config['path'], video)
            timestamp = video.split('_')[1].split('.')[0]
            tracking_path = os.path.join(
                config['tracking_data'],
                f"tracking_{timestamp}.json"
            )
            
            if os.path.exists(tracking_path):
                samples.append({
                    "video": video_path,
                    "tracking": tracking_path
                })
        
        return samples
    
    def _generate_augmented_faces(self, image_path: str, output_dir: str):
        """Generate augmented versions of a face image"""
        config = self.config['face_recognition']['augmentation']
        img = cv2.imread(image_path)
        
        # Rotation augmentation
        for angle in np.linspace(
            config['rotation_range'][0],
            config['rotation_range'][1],
            5
        ):
            if angle == 0:
                continue
                
            matrix = cv2.getRotationMatrix2D(
                (img.shape[1]/2, img.shape[0]/2),
                angle,
                1.0
            )
            rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
            
            # Save rotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            rot_path = os.path.join(
                output_dir,
                f"{base_name}_rot{int(angle)}.jpg"
            )
            cv2.imwrite(rot_path, rotated)
        
        # Brightness augmentation
        for factor in np.linspace(
            config['brightness_range'][0],
            config['brightness_range'][1],
            3
        ):
            if factor == 1.0:
                continue
                
            brightened = cv2.multiply(img, factor)
            
            # Save brightened image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            bright_path = os.path.join(
                output_dir,
                f"{base_name}_bright{int(factor*100)}.jpg"
            )
            cv2.imwrite(bright_path, brightened)
