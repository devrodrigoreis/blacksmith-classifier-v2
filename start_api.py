#!/usr/bin/env python
"""
Starter script for the continuous learning API
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.api import app
import uvicorn

if __name__ == "__main__":
    print("Starting Continuous Learning API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
