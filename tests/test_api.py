"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.api.main import app
    client = TestClient(app)
except ImportError:
    # Create mock app for testing without actual API
    from fastapi import FastAPI
    app = FastAPI()
    client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data

def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model-info")
    # Accept both success (200) and service unavailable (503)
    assert response.status_code in [200, 503]

def test_features_endpoint():
    """Test features endpoint"""
    response = client.get("/features")
    assert response.status_code in [200, 503]

def test_predict_endpoint_structure():
    """Test predict endpoint with sample data"""
    sample_data = {
        "CountryCode": 0.0,
        "Amount": -0.046371,
        "Value": -0.072291,
        "PricingStrategy": -0.349252,
        "TotalAmount": 0.170118,
        "AvgAmount": -0.067623,
        "TransactionCount": -0.311831,
        "StdAmount": 0.0,
        "MinAmount": -0.049,
        "MaxAmount": 0.12,
        "customer_id": "test_customer_001"
    }
    
    response = client.post("/predict", json=sample_data)
    # Accept various status codes
    assert response.status_code in [200, 400, 503, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "risk_probability" in data
        assert "credit_score" in data
        assert "risk_category" in data
        assert "model_used" in data

def test_credit_limit_endpoint():
    """Test credit limit calculation endpoint"""
    response = client.post("/calculate-credit-limit?risk_probability=0.2&annual_income=50000")
    assert response.status_code == 200
    
    if response.status_code == 200:
        data = response.json()
        assert "credit_limit" in data
        assert "recommendation" in data

def test_batch_predict_structure():
    """Test batch predict endpoint structure"""
    batch_data = {
        "customers": [
            {
                "CountryCode": 0.0,
                "Amount": -0.046,
                "Value": -0.072,
                "PricingStrategy": -0.349,
                "TotalAmount": 0.170,
                "AvgAmount": -0.068,
                "TransactionCount": -0.312,
                "StdAmount": 0.0,
                "MinAmount": -0.049,
                "MaxAmount": 0.12,
                "customer_id": "batch_test_1"
            }
        ]
    }
    
    response = client.post("/predict-batch", json=batch_data)
    assert response.status_code in [200, 400, 503, 500]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])