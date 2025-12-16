"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class PredictionInput(BaseModel):
    """
    Input schema for credit risk prediction
    Based on your 37 features from training
    """
    # Top 10 most important features (adjust based on your feature_names.pkl)
    CountryCode: float = Field(..., description="Country code (scaled)")
    Amount: float = Field(..., description="Transaction amount (scaled)")
    Value: float = Field(..., description="Transaction value (scaled)")
    PricingStrategy: float = Field(..., description="Pricing strategy (scaled)")
    TotalAmount: float = Field(..., description="Total transaction amount per customer (scaled)")
    AvgAmount: float = Field(..., description="Average transaction amount (scaled)")
    TransactionCount: float = Field(..., description="Number of transactions (scaled)")
    StdAmount: float = Field(..., description="Standard deviation of amounts (scaled)")
    MinAmount: float = Field(..., description="Minimum transaction amount (scaled)")
    MaxAmount: float = Field(..., description="Maximum transaction amount (scaled)")
    
    # Optional: Add customer ID for tracking
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    
    class Config:
        schema_extra = {
            "example": {
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
                "customer_id": "CustomerId_4406"
            }
        }

class BatchPredictionInput(BaseModel):
    """Input for batch predictions"""
    customers: List[PredictionInput]

class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of being high-risk (0-1)")
    risk_category: str = Field(..., description="Risk category: Low, Medium, or High")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850 range)")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CustomerId_4406",
                "risk_probability": 0.23,
                "risk_category": "Low",
                "credit_score": 720,
                "model_used": "logistic_regression",
                "timestamp": "2025-12-16T07:11:42.123456"
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    timestamp: str = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "logistic_regression",
                "timestamp": "2025-12-16T07:11:42.123456"
            }
        }

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    feature_count: int = Field(..., description="Number of features")
    features: List[str] = Field(..., description="Feature names (first 10)")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    training_date: str = Field(..., description="Training date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "logistic_regression",
                "model_version": "v1.0",
                "feature_count": 37,
                "features": ["CountryCode", "Amount", "Value", "PricingStrategy", "TotalAmount"],
                "performance_metrics": {
                    "roc_auc": 0.9902,
                    "f1_score": 0.4503,
                    "accuracy": 0.9396
                },
                "training_date": "2025-12-16"
            }
        }