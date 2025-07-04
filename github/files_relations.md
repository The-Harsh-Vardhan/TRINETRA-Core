# TRINETRA-Core File Relations and Structure (Streamlined)

## Core Project Files

### Main Configuration Files

- `requirements.txt`: Lists all Python package dependencies
- `pytest.ini`: Configuration for pytest testing framework
- `config/dataset_config.json`: Configuration for dataset management and paths

### Project Documentation

- `github/PROJECT_REQUIREMENTS.md`: Project requirements and best practices
- `github/IMPLEMENTATION_GUIDE.md`: Step-by-step implementation guide
- `github/files_relations.md`: This file - explains project structure

### Main Scripts

- `package_manager.py`: **NEW** - Unified package management (check, install, fix)
- `setup_datasets.py`: Sets up project datasets
- `verifymongodb.py`: Verifies MongoDB connection
- `test_system.py`: System-wide integration tests

### Archived Files (moved to `archive/`)

- `check_dependencies.py`: _(Replaced by package_manager.py)_
- `check_requirements.py`: _(Replaced by package_manager.py)_
- `fix_dependencies.py`: _(Replaced by package_manager.py)_
- `fix_versions.py`: _(Replaced by package_manager.py)_
- `install_requirements.py`: _(Replaced by package_manager.py)_

## Source Code (`src/`)

### Backend (`src/backend/`)

1. Main Backend Files:

   - `main.py`: **SIMPLIFIED** - Now just imports from api/main.py
   - `api/main.py`: **ENHANCED** - Complete FastAPI server with all endpoints

2. API Routes (`src/backend/api/routes/`):

   - `face_recognition.py`: Face recognition endpoints
   - `entrance_tracking.py`: Entrance monitoring endpoints
   - `behavior_analytics.py`: Behavioral analysis endpoints

3. Database (`src/backend/database/`):
   - `client.py`: MongoDB database client and operations

### Frontend (`src/frontend/`)

1. Main Application:

   - `app.py`: **NEW** - Main Streamlit entry point with navigation

2. Pages (`src/frontend/pages/`):

   - `dashboard.py`: Main dashboard interface
   - `analytics.py`: Analytics visualization
   - `customer_insights.py`: Customer behavior insights
   - `live_monitoring.py`: Real-time monitoring interface

3. Utilities (`src/frontend/utils/`):
   - `api_client.py`: Backend API communication
   - `video_processor.py`: Video stream processing
   - `common.py`: **NEW** - Shared utilities and components

### Core Modules (`src/core_modules/`)

1. Face Recognition (`src/core_modules/face_recognition/`):

   - `__init__.py`: Module initialization
   - `face_recognition_main.py`: Main face recognition implementation
   - `keypoint_face_recognition.py`: Alternative keypoint-based implementation
   - `enhanced_face_recognition.py`: **NEW** - Enhanced with HuggingFace streaming datasets

2. Entrance Tracking (`src/core_modules/entrance_tracking/`):

   - `__init__.py`: Module initialization
   - `multi_camera_tracker.py`: Multi-camera coordination
   - `people_counter.py`: People counting implementation
   - `enhanced_multi_camera_tracker.py`: **NEW** - Multi-camera with streaming support
   - `enhanced_people_counter.py`: **NEW** - People counter with remote video support

3. Behavioral Insights (`src/core_modules/behavioral_insights/`):
   - `__init__.py`: Module initialization
   - `behavior_analytics.py`: Behavior analysis implementation
   - `enhanced_behavior_analytics.py`: **NEW** - Advanced analytics with streaming patterns

### Utilities (`src/utils/`)

- `dataset_manager.py`: Dataset management utilities

## Tests (`tests/`)

1. Backend Tests (`tests/backend/`):

   - `test_api.py`: API endpoint tests

2. Frontend Tests (`tests/frontend/`):

   - `test_frontend.py`: Frontend component tests

3. Test Data (`tests/test_data/`):

   - Contains sample data for testing

4. **NEW** Integration Tests:
   - `test_streaming_integration.py`: **NEW** - Comprehensive streaming capabilities tests

## Enhanced Features (Streaming Support)

### Face Recognition Enhancements

- **HuggingFace Dataset Integration**: Load face datasets from HuggingFace Hub
- **Remote Image Loading**: Download and process images from URLs
- **Streaming Face Recognition**: Process faces from remote video streams
- **Enhanced Statistics**: Detailed analytics on streaming vs local faces

### Entrance Tracking Enhancements

- **Multi-Source Video Support**:
  - Local video files
  - Remote video URLs (direct download)
  - RTSP/RTMP streams
  - YouTube videos (with yt-dlp)
  - Webcam feeds
- **Real-time Performance Monitoring**: FPS tracking and optimization
- **Enhanced People Counting**: Improved accuracy with trajectory analysis
- **Automatic Video Looping**: Continuous processing for demo videos

### Behavioral Analytics Enhancements

- **Advanced Pattern Recognition**:
  - Loitering detection
  - Browsing behavior analysis
  - Goal-oriented movement patterns
- **Real-time Crowd Analytics**: Live occupancy and flow metrics
- **Velocity Flow Mapping**: Movement direction and speed analysis
- **Zone-based Heatmaps**: Detailed zone analytics and popularity metrics
- **Streaming Crowd Patterns**: Integration with remote crowd behavior data

### Multi-Camera Tracking Enhancements

- **Threaded Processing**: Parallel processing for multiple camera streams
- **Cross-Camera Person Matching**: Face embedding-based person re-identification
- **Dynamic Source Management**: Add/remove video sources during runtime
- **Comprehensive Analytics**: Journey tracking across multiple cameras

## File Dependencies and Relationships

### Backend Chain:

```
main.py
└── api/main.py
    ├── routes/*.py
    └── database/client.py
        └── core_modules/*
```

### Frontend Chain:

```
app.py
├── pages/*.py
└── utils/*
    └── core_modules/*
```

### Core Modules Dependencies:

```
face_recognition_main.py
└── database/client.py

people_counter.py
└── database/client.py

behavior_analytics.py
└── database/client.py
```

### Test Dependencies:

```
test_api.py
├── backend/main.py
└── core_modules/*

test_frontend.py
├── frontend/pages/*
└── frontend/utils/*
```

## Recommended Directory Structure Changes

### Consider Moving to Archive:

If any of these files are no longer needed, move them to an `archive` folder:

- `fix_versions.py` (if using `fix_dependencies.py`)
- Duplicate requirement checking scripts

### Consider Consolidating:

- Merge similar testing scripts
- Combine related utility functions
- Consolidate configuration files

## Important Notes

1. All paths in configuration files should use forward slashes (/)
2. Keep test data separate from production data
3. Use relative imports within src/ directory
4. Maintain clear separation between frontend and backend
5. Keep core modules independent and loosely coupled

## Development Workflow

1. Backend changes:

   - Modify core modules
   - Update API routes
   - Run backend tests

2. Frontend changes:

   - Update page components
   - Modify API client if needed
   - Run frontend tests

3. System testing:
   - Run integration tests
   - Verify MongoDB connection
   - Check all requirements

## Simplified Usage

### Package Management

```bash
# Check packages
python package_manager.py check

# Install packages
python package_manager.py install

# Fix version issues
python package_manager.py fix

# Complete reinstall
python package_manager.py reinstall
```

### Running the Application

```bash
# Backend
uvicorn src.backend.main:app --reload

# Frontend
streamlit run src/frontend/app.py
```

## Key Improvements Made

1. **Consolidated Scripts**: 5 separate package management scripts → 1 unified `package_manager.py`
2. **Simplified Backend**: Removed duplication between `main.py` and `api/main.py`
3. **Enhanced Frontend**: Added main entry point and shared utilities
4. **Better Organization**: Moved redundant files to `archive/`
5. **Cleaner Structure**: Reduced code duplication and improved maintainability

## File Dependencies (Simplified)

### Backend Chain:

```
main.py → api/main.py → routes/* → database/client.py → core_modules/*
```

### Frontend Chain:

```
app.py → pages/* → utils/* → core_modules/*
```

### Utility Chain:

```
package_manager.py (standalone)
setup_datasets.py → utils/dataset_manager.py
```
