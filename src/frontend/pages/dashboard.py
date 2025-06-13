import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def display_dashboard():
    st.title("TRINETRA-Core Dashboard")
    
    # Create tabs for different sections
    tabs = st.tabs(["Live Monitoring", "Analytics", "Customer Insights"])
    
    with tabs[0]:  # Live Monitoring
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Entrance Traffic")
            # Placeholder for real-time entrance count
            st.metric("Current Count", "25", "+3")
            
        with col2:
            st.subheader("Face Recognition")
            # Placeholder for recent face detections
            st.image("https://via.placeholder.com/300x200", caption="Live Feed")
    
    with tabs[1]:  # Analytics
        st.subheader("Traffic Analytics")
        # Sample traffic data visualization
        dates = ["2025-06-{}".format(i) for i in range(1, 8)]
        traffic = [120, 145, 132, 168, 155, 190, 172]
        
        fig = px.line(x=dates, y=traffic, 
                     title="Weekly Traffic Trend",
                     labels={"x": "Date", "y": "Visitor Count"})
        st.plotly_chart(fig)
        
    with tabs[2]:  # Customer Insights
        st.subheader("Customer Behavior Analysis")
        # Sample behavior metrics
        metrics = {
            "Average Visit Duration": "45 mins",
            "Most Common Path": "Entrance → Electronics → Checkout",
            "Peak Hours": "2 PM - 4 PM"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)

def main():
    st.set_page_config(
        page_title="TRINETRA-Core",
        page_icon="🎥",
        layout="wide"
    )
    display_dashboard()

if __name__ == "__main__":
    main()
