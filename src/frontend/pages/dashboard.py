import streamlit as st
import requests
from datetime import datetime
from utils.common import StreamlitUtils, ChartUtils, DataUtils

def display_dashboard():
    StreamlitUtils.setup_page("Dashboard", "📊")
    
    # Create tabs for different sections
    tabs = st.tabs(["Live Monitoring", "Analytics", "Customer Insights"])
    
    with tabs[0]:  # Live Monitoring
        st.subheader("🔴 Live Status")
        
        # Live metrics using shared utilities
        live_metrics = {
            "Current Count": {"value": "25", "delta": "+3"},
            "Recognition Rate": {"value": "94.2%", "delta": "+1.2%"},
            "Camera Status": {"value": "3/3 Online", "delta": None}
        }
        StreamlitUtils.display_metrics_row(live_metrics)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Entrance Traffic")
            # Use shared chart utilities for traffic visualization
            traffic_data = DataUtils.get_mock_data("traffic")[:12]  # Last 12 hours
            ChartUtils.create_traffic_chart(traffic_data, "Real-time Traffic Flow")
            
        with col2:
            st.subheader("Face Recognition")
            # Placeholder for recent face detections
            st.image("https://via.placeholder.com/300x200", caption="Live Feed")
    
    with tabs[1]:  # Analytics
        st.subheader("📈 Traffic Analytics")
        
        # Use shared traffic chart
        daily_traffic_data = DataUtils.get_mock_data("traffic")
        ChartUtils.create_traffic_chart(daily_traffic_data, "24-Hour Traffic Trend")
        
        # Key metrics using shared utilities
        analytics_metrics = DataUtils.get_mock_data("metrics")
        StreamlitUtils.display_metrics_row(analytics_metrics)
        
    with tabs[2]:  # Customer Insights
        st.subheader("👥 Customer Behavior Analysis")
        
        # Sample behavior metrics using shared utilities
        behavior_metrics = {
            "Average Visit Duration": "45 mins",
            "Most Common Path": "Entrance → Electronics → Checkout",
            "Peak Hours": "2 PM - 4 PM",
            "Return Rate": "35%"
        }
        StreamlitUtils.display_metrics_row(behavior_metrics, columns=2)

def main():
    display_dashboard()

if __name__ == "__main__":
    main()
