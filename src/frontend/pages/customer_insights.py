import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import requests

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
    st.title("Customer Insights")
    
    insights = CustomerInsights()
    
    # Customer ID input
    customer_id = st.text_input("Enter Customer ID", "C123")
    
    if customer_id:
        customer_data = insights.get_customer_data(customer_id)
        
        # Customer Overview
        st.subheader("Customer Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Customer ID", customer_data["customer_id"])
            st.metric("Average Visit Duration", 
                     f"{customer_data['preferences']['avg_visit_duration']} mins")
        
        with col2:
            st.metric("Preferred Time", 
                     customer_data["preferences"]["peak_visit_times"])
            st.metric("Favorite Sections", 
                     ", ".join(customer_data["preferences"]["favorite_sections"]))
        
        # Visit History
        st.subheader("Visit History")
        
        # Convert visit history to DataFrame
        visits_df = pd.DataFrame(customer_data["visit_history"])
        visits_df["date"] = pd.to_datetime(visits_df["date"])
        
        # Plot visit durations
        fig_duration = px.line(
            visits_df,
            x="date",
            y="duration",
            title="Visit Duration Trend",
            labels={"duration": "Duration (minutes)"}
        )
        st.plotly_chart(fig_duration)
        
        # Recent Visits Table
        st.subheader("Recent Visits")
        
        for visit in customer_data["visit_history"]:
            with st.expander(f"Visit on {visit['date'][:10]}"):
                st.write(f"Duration: {visit['duration']} minutes")
                st.write("Sections visited:")
                for section in visit['sections_visited']:
                    st.write(f"- {section}")
        
        # Behavioral Analysis
        st.subheader("Behavioral Analysis")
        
        # Sample behavioral metrics
        behavior_metrics = {
            "Shopping Pattern": "Browser",
            "Price Sensitivity": "Medium",
            "Brand Loyalty": "High",
            "Response to Promotions": "Positive"
        }
        
        col1, col2 = st.columns(2)
        
        for i, (metric, value) in enumerate(behavior_metrics.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(metric, value)

def main():
    st.set_page_config(
        page_title="TRINETRA-Core Customer Insights",
        page_icon="👤",
        layout="wide"
    )
    display_customer_insights()

if __name__ == "__main__":
    main()
