import argparse
import os
from src.utils.dataset_manager import DatasetManager

def setup_face_recognition_dataset(manager: DatasetManager, data_dir: str):
    """Setup face recognition dataset"""
    print("\nSetting up Face Recognition Dataset...")
    
    # Expected structure:
    # data_dir/
    #   person1/
    #     image1.jpg
    #     image2.jpg
    #   person2/
    #     image1.jpg
    #     ...
    
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"\nProcessing images for: {person}")
        for image in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image)
            try:
                manager.add_face_image(image_path, person)
                print(f"✓ Added: {image}")
            except Exception as e:
                print(f"✗ Failed to add {image}: {str(e)}")

def setup_entrance_videos_dataset(manager: DatasetManager, data_dir: str):
    """Setup entrance videos dataset"""
    print("\nSetting up Entrance Videos Dataset...")
    
    # Expected structure:
    # data_dir/
    #   camera1_video1.mp4
    #   camera1_video2.mp4
    #   camera2_video1.mp4
    #   ...
    
    for video in os.listdir(data_dir):
        video_path = os.path.join(data_dir, video)
        if not os.path.isfile(video_path):
            continue
            
        # Extract camera ID from filename (assuming format: cameraX_*.mp4)
        camera_id = video.split('_')[0]
        
        try:
            manager.add_entrance_video(video_path, camera_id)
            print(f"✓ Added: {video}")
        except Exception as e:
            print(f"✗ Failed to add {video}: {str(e)}")

def setup_behavior_samples_dataset(manager: DatasetManager, data_dir: str):
    """Setup behavior samples dataset"""
    print("\nSetting up Behavior Samples Dataset...")
    
    # Expected structure:
    # data_dir/
    #   videos/
    #     sample1.mp4
    #     sample2.mp4
    #   tracking/
    #     sample1.json
    #     sample2.json
    
    videos_dir = os.path.join(data_dir, 'videos')
    tracking_dir = os.path.join(data_dir, 'tracking')
    
    if not (os.path.exists(videos_dir) and os.path.exists(tracking_dir)):
        print("✗ Invalid directory structure. Expected 'videos' and 'tracking' subdirectories.")
        return
    
    for video in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, video)
        tracking_path = os.path.join(tracking_dir, video.replace('.mp4', '.json'))
        
        if not os.path.exists(tracking_path):
            print(f"✗ No tracking data found for {video}")
            continue
        
        try:
            # Read tracking data
            with open(tracking_path, 'r') as f:
                tracking_data = json.load(f)
                
            # Add to dataset
            manager.add_behavior_sample(video_path, tracking_data)
            print(f"✓ Added: {video} with tracking data")
        except Exception as e:
            print(f"✗ Failed to add {video}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Setup TRINETRA datasets')
    parser.add_argument('--data-dir', required=True, help='Root directory containing the datasets')
    parser.add_argument('--type', choices=['all', 'faces', 'entrance', 'behavior'],
                      default='all', help='Type of dataset to setup')
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    if args.type in ['all', 'faces']:
        faces_dir = os.path.join(args.data_dir, 'faces')
        if os.path.exists(faces_dir):
            setup_face_recognition_dataset(manager, faces_dir)
        else:
            print("\n✗ Faces dataset directory not found")
    
    if args.type in ['all', 'entrance']:
        entrance_dir = os.path.join(args.data_dir, 'entrance')
        if os.path.exists(entrance_dir):
            setup_entrance_videos_dataset(manager, entrance_dir)
        else:
            print("\n✗ Entrance videos directory not found")
    
    if args.type in ['all', 'behavior']:
        behavior_dir = os.path.join(args.data_dir, 'behavior')
        if os.path.exists(behavior_dir):
            setup_behavior_samples_dataset(manager, behavior_dir)
        else:
            print("\n✗ Behavior samples directory not found")
    
    print("\nDataset setup complete!")

if __name__ == "__main__":
    main()
