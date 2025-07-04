import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import os
from pathlib import Path

class FacialKeypointNetwork(nn.Module):
    def __init__(self):
        super(FacialKeypointNetwork, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Facial keypoint prediction
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 136)  # 68 keypoints * 2 (x,y)
        
        # Face embedding
        self.fc_embed = nn.Linear(512, 128)
        
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Facial keypoints
        features = F.relu(self.fc1(x))
        features = self.dropout(features)
        keypoints = self.fc2(features)
        
        # Face embedding
        embedding = self.fc_embed(features)
        
        return keypoints, embedding

class KeypointFaceRecognition:
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize Keypoint-based Face Recognition System"""
        self.device = device
        self.model = FacialKeypointNetwork().to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Known face embeddings database
        self.face_db: Dict[str, torch.Tensor] = {}
        
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for the model"""
        # Convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Resize to 96x96 (standard input size)
        img = cv2.resize(img, (96, 96))
        
        # Normalize
        img = img / 255.0
        img = img.transpose((2, 0, 1))  # HWC to CHW
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def get_face_embedding(self, face_img: np.ndarray) -> torch.Tensor:
        """Get face embedding from image"""
        # Preprocess
        face_tensor = self.preprocess_image(face_img)
        
        # Get embedding
        with torch.no_grad():
            keypoints, embedding = self.model(face_tensor)
            
        return F.normalize(embedding, p=2, dim=1)  # L2 normalize
    
    def add_face(self, name: str, face_img: np.ndarray):
        """Add a face to the database"""
        embedding = self.get_face_embedding(face_img)
        self.face_db[name] = embedding
    
    def recognize_face(self, 
                      face_img: np.ndarray,
                      threshold: float = 0.6) -> Optional[str]:
        """Recognize a face by comparing embeddings"""
        if not self.face_db:
            return None
            
        query_embedding = self.get_face_embedding(face_img)
        
        # Compare with database
        max_similarity = -1
        best_match = None
        
        for name, stored_embedding in self.face_db.items():
            similarity = F.cosine_similarity(
                query_embedding,
                stored_embedding
            ).item()
            
            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_match = name
                
        return best_match
    
    def get_facial_keypoints(self, face_img: np.ndarray) -> np.ndarray:
        """Get facial keypoints for visualization"""
        face_tensor = self.preprocess_image(face_img)
        
        with torch.no_grad():
            keypoints, _ = self.model(face_tensor)
            
        # Reshape to (68, 2) for x,y coordinates
        keypoints = keypoints.cpu().numpy().reshape(-1, 2)
        
        # Scale keypoints to image size
        h, w = face_img.shape[:2]
        keypoints[:, 0] *= w / 96
        keypoints[:, 1] *= h / 96
        
        return keypoints
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Process a frame and return recognized faces with keypoints"""
        faces = self.detect_faces(frame)
        results = []
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            # Get recognition result
            name = self.recognize_face(face_img)
            
            # Get keypoints
            keypoints = self.get_facial_keypoints(face_img)
            
            # Adjust keypoint coordinates to frame coordinates
            keypoints[:, 0] += x
            keypoints[:, 1] += y
            
            results.append({
                "bbox": (x, y, w, h),
                "name": name,
                "keypoints": keypoints.tolist()
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw name
            if name:
                cv2.putText(frame, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 255, 0), 2)
            
            # Draw keypoints
            for (kx, ky) in keypoints:
                cv2.circle(frame, (int(kx), int(ky)), 2, (255, 0, 0), -1)
        
        return results, frame
    
    def save_model(self, path: str):
        """Save the model weights"""
        torch.save(self.model.state_dict(), path)
    
    def train(self,
             train_data: List[Tuple[np.ndarray, np.ndarray]],
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001):
        """Train the model on keypoint data"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert training data to tensors
        X_train = []
        y_train = []
        
        for img, keypoints in train_data:
            X_train.append(self.preprocess_image(img))
            y_train.append(torch.FloatTensor(keypoints).to(self.device))
            
        X_train = torch.cat(X_train)
        y_train = torch.stack(y_train)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                keypoints_pred, _ = self.model(batch_X)
                loss = criterion(keypoints_pred, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
            avg_loss = total_loss / batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        self.model.eval()
