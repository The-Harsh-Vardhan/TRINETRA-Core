from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, List
from datetime import datetime
import cv2
import numpy as np
from ....core_modules.face_recognition.face_recognition_main import FaceRecognitionSystem
from ...database.client import Database

router = APIRouter()
face_system = FaceRecognitionSystem()

@router.post("/register")
async def register_face(name: str, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        success = face_system.train_from_image(name, img)
        if not success:
            raise HTTPException(status_code=400, detail="No face detected in image")
            
        # Save to database
        face_data = {
            "name": name,
            "registered_at": datetime.now()
        }
        await Database.insert_face(face_data)
        
        return {"message": "Face registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results, _ = face_system.process_frame(img)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
