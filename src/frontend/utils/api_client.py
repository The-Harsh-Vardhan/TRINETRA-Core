import requests
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _make_request(self,
                     method: str,
                     endpoint: str,
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to the API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return {}
    
    # Traffic Analytics Endpoints
    def get_traffic_analytics(self,
                            start_date: datetime,
                            end_date: datetime) -> Dict:
        """Get traffic analytics data"""
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        return self._make_request(
            "GET",
            "analytics/traffic",
            params=params
        )
    
    # Customer Tracking Endpoints
    def get_customer_journey(self, customer_id: str) -> Dict:
        """Get customer journey data"""
        return self._make_request(
            "GET",
            f"tracking/journey/{customer_id}"
        )
    
    def get_customer_history(self, customer_id: str) -> Dict:
        """Get customer recognition history"""
        return self._make_request(
            "GET",
            f"recognition/history/{customer_id}"
        )
    
    # Behavior Analytics Endpoints
    def get_behavior_analytics(self, customer_id: str) -> Dict:
        """Get behavioral analytics data"""
        return self._make_request(
            "GET",
            f"analytics/behavior/{customer_id}"
        )
    
    def get_customer_insights(self, customer_id: str) -> Dict:
        """Get comprehensive customer insights"""
        return self._make_request(
            "GET",
            f"analytics/customer/{customer_id}"
        )
    
    # Batch Operations
    def get_multiple_customers(self, customer_ids: List[str]) -> Dict:
        """Get data for multiple customers"""
        data = {"customer_ids": customer_ids}
        return self._make_request(
            "POST",
            "analytics/customers/batch",
            data=data
        )
    
    # Real-time Monitoring
    def get_current_count(self) -> Dict:
        """Get current people count"""
        return self._make_request(
            "GET",
            "monitoring/count"
        )
    
    def get_live_analytics(self) -> Dict:
        """Get real-time analytics"""
        return self._make_request(
            "GET",
            "monitoring/analytics"
        )
