"""
TRINETRA Core Backend Main Entry Point
This is the main FastAPI application entry point.
"""

# Import the configured app from api module
from .api.main import app

# Re-export for convenience
__all__ = ["app"]
