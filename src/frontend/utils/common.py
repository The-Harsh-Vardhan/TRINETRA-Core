"""
Common utilities and shared components for TRINETRA frontend
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class StreamlitUtils:
    """Common Streamlit utility functions"""
    
    @staticmethod
    def setup_page(title: str, icon: str = "👁️", layout: str = "wide"):
        """Setup page configuration"""
        st.set_page_config(
            page_title=f"TRINETRA - {title}",
            page_icon=icon,
            layout=layout
        )
        st.title(f"{icon} TRINETRA - {title}")
    
    @staticmethod
    def display_metrics_row(metrics: Dict[str, Any], columns: int = 3):
        """Display metrics in a row"""
        cols = st.columns(columns)
        for i, (label, data) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(data, dict):
                    st.metric(
                        label=label,
                        value=data.get("value", "N/A"),
                        delta=data.get("delta", None)
                    )
                else:
                    st.metric(label=label, value=data)
    
    @staticmethod
    def create_time_filter():
        """Create common time range filter"""
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now().date())
        return start_date, end_date
    
    @staticmethod
    def display_status_indicator(status: str, message: str):
        """Display status with appropriate styling"""
        if status == "success":
            st.success(message)
        elif status == "warning":
            st.warning(message)
        elif status == "error":
            st.error(message)
        else:
            st.info(message)

class ChartUtils:
    """Common chart utilities"""
    
    @staticmethod
    def create_traffic_chart(data: List[Dict], title: str = "Traffic Flow"):
        """Create traffic flow chart"""
        if not data:
            st.warning("No traffic data available")
            return
        
        df = pd.DataFrame(data)
        fig = go.Figure()
        
        if 'entries' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['hour'] if 'hour' in df.columns else df.index,
                y=df['entries'],
                name="Entries",
                line=dict(color="#2ecc71")
            ))
        
        if 'exits' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['hour'] if 'hour' in df.columns else df.index,
                y=df['exits'],
                name="Exits",
                line=dict(color="#e74c3c")
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Number of People",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_heatmap(data: List[List], title: str = "Activity Heatmap"):
        """Create heatmap visualization"""
        if not data:
            st.warning("No heatmap data available")
            return
        
        fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis'))
        fig.update_layout(
            title=title,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_bar_chart(data: Dict, title: str, x_label: str, y_label: str):
        """Create bar chart"""
        fig = go.Figure(data=[
            go.Bar(x=list(data.keys()), y=list(data.values()))
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def format_number(num: float) -> str:
        """Format large numbers with appropriate suffixes"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:.0f}"
    
    @staticmethod
    def calculate_percentage_change(current: float, previous: float) -> float:
        """Calculate percentage change"""
        if previous == 0:
            return 0
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def get_mock_data(data_type: str) -> Any:
        """Get mock data for testing"""
        mock_data = {
            "traffic": [
                {"hour": i, "entries": i*2 + 10, "exits": i*1.8 + 8}
                for i in range(24)
            ],
            "metrics": {
                "Total Visitors": {"value": "1,234", "delta": "+12%"},
                "Average Duration": {"value": "15.5m", "delta": "+2.3m"},
                "Recognition Rate": {"value": "94.2%", "delta": "+1.2%"}
            },
            "heatmap": [[i+j for j in range(10)] for i in range(10)]
        }
        return mock_data.get(data_type, {})

# Common constants
CAMERA_STATUSES = ["Online", "Offline", "Maintenance"]
ALERT_TYPES = ["Info", "Warning", "Error", "Success"]
TIME_RANGES = ["Last Hour", "Last 24 Hours", "Last Week", "Last Month"]
