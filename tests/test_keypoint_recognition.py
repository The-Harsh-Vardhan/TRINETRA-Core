import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from src.core_modules.face_recognition.keypoint_face_recognition import KeypointFaceRecognition

def test_live_recognition(model_path: str = None):
    """Test face recognition with webcam"""
    face_recognizer = KeypointFaceRecognition(model_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    print("\nControls:")
    print("R - Register face (when a clear face is visible)")
    print("Q - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results, annotated_frame = face_recognizer.process_frame(frame)
        
        # Display results
        cv2.imshow('Face Recognition Test', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Register face if detected
            if results:
                name = input("\nEnter name for the detected face: ")
                face_data = results[0]
                x, y, w, h = face_data["bbox"]
                face_img = frame[y:y+h, x:x+w]
                face_recognizer.add_face(name, face_img)
                print(f"Registered face for: {name}")
    
    cap.release()
    cv2.destroyAllWindows()

def test_image_recognition(image_path: str, model_path: str = None):
    """Test face recognition on a single image"""
    face_recognizer = KeypointFaceRecognition(model_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    # Process image
    results, annotated_img = face_recognizer.process_frame(img)
    
    # Display results
    print("\nDetected faces:", len(results))
    for result in results:
        name = result.get("name", "Unknown")
        print(f"- {name}")
    
    cv2.imshow('Recognition Result', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Keypoint-based Face Recognition')
    parser.add_argument('--mode', choices=['live', 'image'], default='live',
                      help='Test mode: live (webcam) or image')
    parser.add_argument('--model', help='Path to trained model weights')
    parser.add_argument('--image', help='Path to test image (for image mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        test_live_recognition(args.model)
    else:
        if not args.image:
            print("Error: Please provide an image path for image mode")
            return
        test_image_recognition(args.image, args.model)

if __name__ == "__main__":
    main()
