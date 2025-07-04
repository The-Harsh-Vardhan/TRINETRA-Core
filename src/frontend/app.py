"""
TRINETRA Core Streamlit Frontend Application
Main entry point for the web interface
"""

import streamlit as st
from pages.dashboard import display_dashboard
from pages.analytics import display_analytics
from pages.customer_insights import display_customer_insights
from pages.live_monitoring import display_live_monitoring
from utils.common import StreamlitUtils

def main():
    """Main application"""
    StreamlitUtils.setup_page("Surveillance System", "👁️")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Live Monitoring", "Analytics", "Customer Insights"]
    )
    
    # Page routing
    if page == "Dashboard":
        display_dashboard()
    elif page == "Live Monitoring":
        display_live_monitoring()
    elif page == "Analytics":
        display_analytics()
    elif page == "Customer Insights":
        display_customer_insights()

if __name__ == "__main__":
    main()
