# TRINETRA Core (Targeted Retail Insights via NETworked Real-time Analytics)

A smart retail surveillance system focused on customer tracking, identification, and behavioral analytics.

## Overview

TRINETRA Core is a streamlined version of the TRINETRA retail analytics system, focusing on three essential capabilities:

1. **Enhanced Entrance & Journey Tracking**: Combined people counting and customer journey tracking
2. **Face Recognition**: Customer identification and re-identification across cameras
3. **Behavioral Analytics**: Pattern analysis and customer insights generation

## Project Structure

```
TRINETRA-Core/
├── src/
│   ├── core_modules/
│   │   ├── entrance_tracking/     # People counting and tracking
│   │   ├── face_recognition/      # Face detection and recognition
│   │   └── behavioral_insights/   # Analytics processing
│   ├── frontend/                  # Streamlit dashboard
│   ├── backend/                   # FastAPI backend
│   └── test_videos/              # Sample videos for testing
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend: `cd src/backend && uvicorn main:app --reload`
4. Run the frontend: `cd src/frontend && streamlit run dashboard.py`

## Core Features

1. **Entrance Tracking**

   - Real-time people counting
   - Multi-camera tracking
   - Customer journey mapping

2. **Face Recognition**

   - Customer identification
   - Recognition history
   - Face database management

3. **Behavioral Analytics**
   - Visit patterns analysis
   - Customer segmentation
   - Journey insights

## Technology Stack

- **Frontend**: Streamlit, Plotly
- **Backend**: FastAPI, MongoDB
- **Core Processing**: OpenCV, DeepFace
- **Analytics**: Pandas, NumPy
