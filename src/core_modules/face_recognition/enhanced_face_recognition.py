import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import io
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDataManager:
    """Manages streaming face datasets from various sources"""
    
    def __init__(self):
        self.cache_dir = "cache/face_datasets"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_from_huggingface(self, dataset_name: str = "nateraw/celebrity-faces", streaming: bool = True) -> List[Dict]:
        """Load face dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
            
            if streaming:
                dataset = load_dataset(dataset_name, streaming=True)
                # Take first 100 samples for demo
                samples = []
                for i, sample in enumerate(dataset['train']):
                    if i >= 100:  # Limit for demo
                        break
                    samples.append({
                        'image': sample.get('image'),
                        'label': sample.get('text', f'person_{i}')
                    })
                return samples
            else:
                dataset = load_dataset(dataset_name)
                return [{'image': sample['image'], 'label': sample.get('text', f'person_{i}')} 
                       for i, sample in enumerate(dataset['train'])]
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {e}")
            return []
    
    def load_from_url(self, image_urls: List[str], labels: List[str] = None) -> List[Dict]:
        """Load face images from URLs"""
        samples = []
        labels = labels or [f"person_{i}" for i in range(len(image_urls))]
        
        for url, label in zip(image_urls, labels):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    samples.append({'image': image, 'label': label})
                    logger.info(f"Loaded image for {label} from {url}")
            except Exception as e:
                logger.error(f"Error loading image from {url}: {e}")
        
        return samples
    
    def process_streaming_samples(self, samples: List[Dict], max_samples: int = 50) -> Dict[str, Any]:
        """Process streaming samples and extract embeddings"""
        processed_faces = {}
        
        for i, sample in enumerate(samples[:max_samples]):
            try:
                image = sample['image']
                label = sample['label']
                
                # Convert PIL image to numpy array
                if hasattr(image, 'convert'):
                    image = np.array(image.convert('RGB'))
                
                # Extract face embedding
                embedding = DeepFace.represent(image, model_name="Facenet")[0]["embedding"]
                
                processed_faces[f"{label}_{i}"] = {
                    'embedding': embedding,
                    'metadata': {'source': 'streaming', 'original_label': label},
                    'last_seen': datetime.now(),
                    'visit_count': 0
                }
                
                logger.info(f"Processed face {i+1}/{min(len(samples), max_samples)}: {label}")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        return processed_faces

class EnhancedFaceRecognitionSystem:
    def __init__(self, database_path='faces.pkl', enable_streaming: bool = True):
        self.database_path = database_path
        self.enable_streaming = enable_streaming
        self.streaming_manager = StreamingDataManager() if enable_streaming else None
        self.known_faces = self.load_database()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize with streaming data if enabled
        if enable_streaming and len(self.known_faces) == 0:
            self.initialize_with_streaming_data()
        
    def initialize_with_streaming_data(self):
        """Initialize face database with streaming dataset"""
        try:
            logger.info("Initializing with streaming face dataset...")
            
            # Try to load from HuggingFace first
            samples = self.streaming_manager.load_from_huggingface()
            
            if not samples:
                # Fallback to demo URLs
                demo_urls = [
                    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300",
                    "https://images.unsplash.com/photo-1494790108755-2616b612b743?w=300",
                    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=300"
                ]
                samples = self.streaming_manager.load_from_url(demo_urls)
            
            if samples:
                streaming_faces = self.streaming_manager.process_streaming_samples(samples, max_samples=10)
                self.known_faces.update(streaming_faces)
                self.save_database()
                logger.info(f"Initialized with {len(streaming_faces)} streaming faces")
            
        except Exception as e:
            logger.error(f"Error initializing with streaming data: {e}")
    
    def load_database(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def add_face(self, face_image, customer_id, metadata=None):
        try:
            embedding = DeepFace.represent(face_image, model_name="Facenet")[0]["embedding"]
            self.known_faces[customer_id] = {
                'embedding': embedding,
                'metadata': metadata or {},
                'last_seen': datetime.now(),
                'visit_count': 1
            }
            self.save_database()
            return True
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False

    def recognize_face(self, face_image, threshold=0.6):
        try:
            current_embedding = DeepFace.represent(face_image, model_name="Facenet")[0]["embedding"]
            
            best_match = None
            min_distance = float('inf')
            
            for customer_id, data in self.known_faces.items():
                distance = np.linalg.norm(np.array(current_embedding) - np.array(data['embedding']))
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    best_match = customer_id
            
            if best_match:
                # Update customer data
                self.known_faces[best_match]['last_seen'] = datetime.now()
                self.known_faces[best_match]['visit_count'] += 1
                self.save_database()
                return best_match, self.known_faces[best_match]
            
            return None, None

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, None

    def process_frame(self, frame):
        """Process frame for face detection and recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            customer_id, customer_data = self.recognize_face(face_img)
            
            # Draw rectangle around face
            color = (0, 255, 0) if customer_id else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display customer info if recognized
            if customer_id:
                label = f"ID: {customer_id}"
                visits = f"Visits: {customer_data['visit_count']}" if customer_data else ""
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, visits, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                results.append({
                    'bbox': (x, y, w, h),
                    'customer_id': customer_id,
                    'confidence': 1.0 - min_distance if 'min_distance' in locals() else 0.5,
                    'metadata': customer_data.get('metadata', {}) if customer_data else {}
                })
            else:
                results.append({
                    'bbox': (x, y, w, h),
                    'customer_id': None,
                    'confidence': 0.0,
                    'metadata': {}
                })
        
        return results, frame

    def get_statistics(self) -> Dict[str, Any]:
        """Get face recognition statistics"""
        total_faces = len(self.known_faces)
        streaming_faces = sum(1 for face in self.known_faces.values() 
                            if face.get('metadata', {}).get('source') == 'streaming')
        local_faces = total_faces - streaming_faces
        
        return {
            'total_faces': total_faces,
            'streaming_faces': streaming_faces,
            'local_faces': local_faces,
            'database_path': self.database_path,
            'streaming_enabled': self.enable_streaming
        }

    def run(self, source=0):
        """Run face recognition with video capture"""
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results, processed_frame = self.process_frame(frame)
            cv2.imshow('Enhanced Face Recognition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Add new face when 'a' is pressed
                customer_id = f"LOCAL_{len([k for k in self.known_faces.keys() if k.startswith('LOCAL_')])}"
                if self.add_face(frame, customer_id):
                    logger.info(f"Added new customer with ID: {customer_id}")
            elif key == ord('s'):
                # Show statistics
                stats = self.get_statistics()
                logger.info(f"Statistics: {stats}")
        
        cap.release()
        cv2.destroyAllWindows()

# Backward compatibility
FaceRecognitionSystem = EnhancedFaceRecognitionSystem

if __name__ == "__main__":
    face_system = EnhancedFaceRecognitionSystem(enable_streaming=True)
    logger.info(f"System initialized with {len(face_system.known_faces)} known faces")
    face_system.run()
