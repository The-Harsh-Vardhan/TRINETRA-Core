import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests
from utils.common import StreamlitUtils, ChartUtils, DataUtils

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
    StreamlitUtils.setup_page("Analytics", "📊")
    
    analytics = Analytics()
    
    # Date range selection using shared utilities
    start_date, end_date = StreamlitUtils.create_time_filter()
    
    # Get traffic data
    traffic_data = analytics.get_traffic_data(start_date, end_date)
    
    # Traffic Analysis Section
    st.subheader("🚶 Traffic Analysis")
    
    # Create DataFrame from traffic data
    df = pd.DataFrame(traffic_data["hourly_counts"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Convert to format expected by ChartUtils
    hourly_data = []
    for _, row in df.iterrows():
        hourly_data.append({
            "hour": row["timestamp"].hour,
            "entries": row["count"],
            "exits": int(row["count"] * 0.8)  # Mock exits data
        })
    
    # Use shared chart utilities
    ChartUtils.create_traffic_chart(hourly_data, "Hourly Traffic Pattern")
    
    # Hourly pattern bar chart
    hourly_pattern = df.groupby(df["timestamp"].dt.hour)["count"].mean()
    hourly_dict = {f"{hour:02d}:00": count for hour, count in hourly_pattern.items()}
    ChartUtils.create_bar_chart(
        hourly_dict, 
        "Average Hourly Traffic Pattern", 
        "Hour of Day", 
        "Average Count"
    )
    
    # Key Metrics using shared utilities
    daily_traffic = df.set_index("timestamp").resample("D").sum()
    total_traffic = df["count"].sum()
    avg_daily = daily_traffic["count"].mean()
    peak_hour = hourly_pattern.idxmax()
    
    traffic_metrics = {
        "Total Traffic": {"value": DataUtils.format_number(total_traffic), "delta": "+12%"},
        "Average Daily": {"value": DataUtils.format_number(avg_daily), "delta": "+5%"},
        "Peak Hour": {"value": f"{peak_hour:02d}:00", "delta": None}
    }
    StreamlitUtils.display_metrics_row(traffic_metrics)
    
    # Behavioral Insights Section
    st.subheader("🧠 Behavioral Insights")
    
    # Use shared utilities for behavior metrics
    behavior_metrics = {
        "Average Visit Duration": DataUtils.format_duration(2700),  # 45 minutes
        "Most Common Path": "Entrance → Electronics → Checkout",
        "Return Customer Rate": "35%",
        "Peak Shopping Hours": "2 PM - 4 PM"
    }
    StreamlitUtils.display_metrics_row(behavior_metrics, columns=2)

def main():
    display_analytics()

if __name__ == "__main__":
    main()
