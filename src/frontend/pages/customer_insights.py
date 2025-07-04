import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import requests
from utils.common import StreamlitUtils, ChartUtils, DataUtils

class CustomerInsights:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
    
    def get_customer_data(self, customer_id):
        """Fetch customer data from backend"""
        try:
            response = requests.get(
                f"{self.api_base_url}/analytics/customer/{customer_id}"
            )
            return response.json()
        except:
            # Return sample data for development
            return self._get_sample_customer_data()
    
    def _get_sample_customer_data(self):
        """Generate sample customer data for development"""
        return {
            "customer_id": "C123",
            "visit_history": [
                {
                    "date": str(datetime.now() - timedelta(days=x)),
                    "duration": 45 + x*5,
                    "sections_visited": ["Electronics", "Clothing", "Checkout"]
                }
                for x in range(5)
            ],
            "preferences": {
                "favorite_sections": ["Electronics", "Clothing"],
                "peak_visit_times": "Evening",
                "avg_visit_duration": 45
            }
        }

def display_customer_insights():
    StreamlitUtils.setup_page("Customer Insights", "👥")
    
    insights = CustomerInsights()
    
    # Customer ID input
    customer_id = st.text_input("Enter Customer ID", "C123")
    
    if customer_id:
        customer_data = insights.get_customer_data(customer_id)
        
        # Customer Overview using shared utilities
        st.subheader("📊 Customer Overview")
        
        overview_metrics = {
            "Customer ID": customer_data["customer_id"],
            "Average Visit Duration": DataUtils.format_duration(customer_data['preferences']['avg_visit_duration'] * 60),
            "Preferred Time": customer_data["preferences"]["peak_visit_times"],
            "Favorite Sections": ", ".join(customer_data["preferences"]["favorite_sections"])
        }
        StreamlitUtils.display_metrics_row(overview_metrics, columns=2)
        
        # Visit History
        st.subheader("📈 Visit History")
        
        # Convert visit history to DataFrame
        visits_df = pd.DataFrame(customer_data["visit_history"])
        visits_df["date"] = pd.to_datetime(visits_df["date"])
        
        # Create traffic-style data for visit duration trend
        visit_trend_data = []
        for _, row in visits_df.iterrows():
            visit_trend_data.append({
                "hour": row["date"].hour,
                "entries": row["duration"],
                "exits": 0  # Not applicable for customer visits
            })
        
        # Use shared chart utility
        ChartUtils.create_traffic_chart(visit_trend_data[::-1], "Visit Duration Trend (Recent First)")
        
        # Recent Visits Table
        st.subheader("🕒 Recent Visits")
        
        for visit in customer_data["visit_history"]:
            with st.expander(f"Visit on {visit['date'][:10]}"):
                st.write(f"Duration: {DataUtils.format_duration(visit['duration'] * 60)}")
                st.write("Sections visited:")
                for section in visit['sections_visited']:
                    st.write(f"- {section}")
        
        # Behavioral Analysis using shared utilities
        st.subheader("🧠 Behavioral Analysis")
        
        behavior_metrics = {
            "Shopping Pattern": "Browser",
            "Price Sensitivity": "Medium",
            "Brand Loyalty": "High",
            "Response to Promotions": "Positive"
        }
        StreamlitUtils.display_metrics_row(behavior_metrics, columns=2)

def main():
    display_customer_insights()

if __name__ == "__main__":
    main()
