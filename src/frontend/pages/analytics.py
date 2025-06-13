import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests

class Analytics:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
    
    def get_traffic_data(self, start_date, end_date):
        """Fetch traffic data from backend"""
        try:
            response = requests.get(
                f"{self.api_base_url}/analytics/traffic",
                params={"start_date": start_date, "end_date": end_date}
            )
            return response.json()
        except:
            # Return sample data for development
            return self._get_sample_traffic_data()
    
    def _get_sample_traffic_data(self):
        """Generate sample traffic data for development"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='H'
        )
        return {
            "hourly_counts": [
                {"timestamp": str(date), "count": int(100 + date.hour * 5)}
                for date in dates
            ]
        }

def display_analytics():
    st.title("Analytics Dashboard")
    
    analytics = Analytics()
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Get traffic data
    traffic_data = analytics.get_traffic_data(start_date, end_date)
    
    # Traffic Analysis
    st.subheader("Traffic Analysis")
    
    # Create DataFrame from traffic data
    df = pd.DataFrame(traffic_data["hourly_counts"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Daily traffic trend
    daily_traffic = df.set_index("timestamp").resample("D").sum()
    fig_daily = px.line(
        daily_traffic,
        x=daily_traffic.index,
        y="count",
        title="Daily Traffic Trend"
    )
    st.plotly_chart(fig_daily)
    
    # Hourly pattern
    hourly_pattern = df.groupby(df["timestamp"].dt.hour)["count"].mean()
    fig_hourly = px.bar(
        x=hourly_pattern.index,
        y=hourly_pattern.values,
        title="Average Hourly Traffic Pattern",
        labels={"x": "Hour of Day", "y": "Average Count"}
    )
    st.plotly_chart(fig_hourly)
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_traffic = df["count"].sum()
        st.metric("Total Traffic", f"{total_traffic:,}")
        
    with col2:
        avg_daily = daily_traffic["count"].mean()
        st.metric("Average Daily Traffic", f"{avg_daily:,.0f}")
        
    with col3:
        peak_hour = hourly_pattern.idxmax()
        st.metric("Peak Hour", f"{peak_hour:02d}:00")
    
    # Behavioral Insights
    st.subheader("Behavioral Insights")
    
    # Sample behavior metrics
    behavior_data = {
        "Average Visit Duration": "45 minutes",
        "Most Common Path": "Entrance → Electronics → Checkout",
        "Return Customer Rate": "35%",
        "Peak Shopping Hours": "2 PM - 4 PM"
    }
    
    for metric, value in behavior_data.items():
        st.metric(metric, value)

def main():
    st.set_page_config(
        page_title="TRINETRA-Core Analytics",
        page_icon="📊",
        layout="wide"
    )
    display_analytics()

if __name__ == "__main__":
    main()
