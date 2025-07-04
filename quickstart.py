#!/usr/bin/env python3
"""
TRINETRA-Core Quick Setup and Run Script
Sets up the project, creates sample data, and starts both backend and frontend
"""

import os
import sys
import subprocess
import time
import tempfile
import requests
from pathlib import Path
import cv2
import numpy as np

def check_python_environment():
    """Check if we're in the right virtual environment"""
    print("🔍 Checking Python environment...")
    
    python_exe = sys.executable
    if "TRINETRA-Core" in python_exe and "venv" in python_exe:
        print(f"✅ Using virtual environment: {python_exe}")
        return True
    else:
        print(f"⚠️  Not in TRINETRA-Core virtual environment: {python_exe}")
        return True  # Continue anyway

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # First install basic requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_sample_face_data():
    """Create sample face recognition data"""
    print("\n👥 Creating sample face recognition data...")
    
    face_dir = Path("datasets/face_recognition")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample person directories
    persons = ["john_doe", "jane_smith", "bob_wilson"]
    
    for person in persons:
        person_dir = face_dir / person
        person_dir.mkdir(exist_ok=True)
        
        # Create simple colored rectangles as sample "face" images
        for i in range(3):
            # Create a colored rectangle as a sample face image
            height, width = 150, 150
            color = (50 + i*50, 100 + i*30, 200 - i*40)  # Different colors for each person
            
            image = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Add some text to make it recognizable
            cv2.putText(image, person[:8], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"img_{i+1}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            image_path = person_dir / f"image_{i+1}.jpg"
            cv2.imwrite(str(image_path), image)
            
        print(f"✅ Created sample data for {person}")
    
    return True

def create_sample_video_data():
    """Create sample video data for entrance tracking"""
    print("\n🎥 Creating sample video data...")
    
    video_dir = Path("datasets/entrance_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test video
    video_path = video_dir / "sample_entrance.mp4"
    
    if not video_path.exists():
        # Create a simple video with moving rectangle (simulating a person)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640, 480))
        
        for frame_num in range(100):  # 5 seconds at 20 fps
            # Create a frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add a moving rectangle (simulating a person)
            x = int(50 + (frame_num * 5) % 540)
            y = 200
            cv2.rectangle(frame, (x, y), (x+50, y+100), (0, 255, 0), -1)
            
            # Add frame number
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created sample video: {video_path}")
    else:
        print(f"✅ Sample video already exists: {video_path}")
    
    return True

def create_env_file():
    """Create a basic .env file if it doesn't exist"""
    print("\n⚙️  Setting up environment configuration...")
    
    env_content = """# TRINETRA-Core Environment Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=trinetra_core
API_HOST=localhost
API_PORT=8000
DEBUG=True
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("✅ .env file already exists")
    
    return True

def test_imports():
    """Test if we can import the main modules"""
    print("\n🧪 Testing module imports...")
    
    try:
        # Test core imports
        import cv2
        import numpy as np
        import streamlit
        import fastapi
        print("✅ Core modules imported successfully")
        
        # Test if our custom modules can be imported
        sys.path.append('src')
        from core_modules.face_recognition.enhanced_face_recognition import EnhancedFaceRecognitionSystem
        from core_modules.entrance_tracking.enhanced_people_counter import EnhancedPeopleCounter
        print("✅ TRINETRA modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting FastAPI backend...")
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    try:
        # Start the backend server
        backend_cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.backend.main:app", 
            "--host", "localhost", 
            "--port", "8000", 
            "--reload"
        ]
        
        print(f"Running: {' '.join(backend_cmd)}")
        backend_process = subprocess.Popen(backend_cmd)
        print("✅ Backend server starting... (Process ID: {})".format(backend_process.pid))
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Test if backend is responding
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend server is responding!")
            else:
                print(f"⚠️  Backend server responded with status {response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️  Backend server may still be starting up...")
        
        return backend_process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("\n🌐 Starting Streamlit frontend...")
    
    try:
        # Start the frontend
        frontend_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "src/frontend/app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        print(f"Running: {' '.join(frontend_cmd)}")
        frontend_process = subprocess.Popen(frontend_cmd)
        print("✅ Frontend server starting... (Process ID: {})".format(frontend_process.pid))
        
        time.sleep(2)
        print("✅ Frontend should be available at: http://localhost:8501")
        
        return frontend_process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main setup and run function"""
    print("🚀 TRINETRA-Core Quick Setup & Run")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_python_environment():
        print("❌ Environment check failed")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Step 3: Create sample data
    if not create_sample_face_data():
        print("❌ Failed to create sample face data")
        return False
    
    if not create_sample_video_data():
        print("❌ Failed to create sample video data")
        return False
    
    # Step 4: Setup environment
    if not create_env_file():
        print("❌ Failed to create environment file")
        return False
    
    # Step 5: Test imports
    if not test_imports():
        print("❌ Module import test failed")
        return False
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 Starting services...")
    
    # Step 6: Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend")
        return False
    
    # Step 7: Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        if backend_process:
            backend_process.terminate()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 TRINETRA-Core is now running!")
    print("📊 Frontend: http://localhost:8501")
    print("🔌 Backend API: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop all services")
    print("=" * 50)
    
    try:
        # Keep the script running and monitor the processes
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("⚠️  Backend process has stopped")
                break
            if frontend_process.poll() is not None:
                print("⚠️  Frontend process has stopped")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ Services stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
