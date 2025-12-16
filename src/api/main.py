"""
FastAPI Application for Credit Risk Prediction API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging


import sys
import os
# Get the project root directory (two levels up from this file's location)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add it to Python's module search path
sys.path.insert(0, project_root)

from src.api.pydantic_models import (
    PredictionInput, 
    PredictionOutput, 
    BatchPredictionInput,
    HealthResponse,
    ModelInfo
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk Prediction API",
    description="API for predicting credit risk using alternative data and RFM analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
MODEL = None
SCALER = None
FEATURE_NAMES = []
MODEL_METRICS = {}

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown (optional)
    pass

app = FastAPI(lifespan=lifespan)

async def startup_event():
    """Load model artifacts on application startup"""
    global MODEL, SCALER, FEATURE_NAMES, MODEL_METRICS
    
    try:
        # Load from your Task 5 outputs
        MODEL = joblib.load("models/logistic_regression_model.pkl")
        SCALER = joblib.load("models/scaler.pkl")
        FEATURE_NAMES = joblib.load("models/feature_names.pkl")
        
        # Load metrics if available
        try:
            import json
            with open("reports/metrics.json", "r") as f:
                all_metrics = json.load(f)
                MODEL_METRICS = all_metrics.get("logistic_regression", {})
        except:
            MODEL_METRICS = {
                "roc_auc": 0.9902,
                "f1_score": 0.4503,
                "accuracy": 0.9396,
                "precision": 0.2918,
                "recall": 0.9854
            }
        
        logger.info(f" Model loaded successfully: {type(MODEL).__name__}")
        logger.info(f" Features: {len(FEATURE_NAMES)}")
        logger.info(f" Sample features: {FEATURE_NAMES[:5]}")
        
    except Exception as e:
        logger.error(f" Failed to load model artifacts: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bati Bank Credit Risk Prediction API",
        "service": "Buy-Now-Pay-Later Credit Scoring",
        "version": "1.0.0",
        "status": "active",
        "model": "Logistic Regression (Task 5)",
        "model_loaded": MODEL is not None,
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "batch_predict": "/predict-batch",
            "features": "/features"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL else "degraded",
        model_loaded=MODEL is not None,
        model_name="logistic_regression" if MODEL else None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Get information about the loaded model"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="logistic_regression",
        model_version="v1.0",
        feature_count=len(FEATURE_NAMES),
        features=FEATURE_NAMES[:10],  # Show first 10 features
        performance_metrics=MODEL_METRICS,
        training_date="2025-12-16"  # Update with actual date
    )

@app.get("/features")
async def list_features():
    """List all features used by the model"""
    if not FEATURE_NAMES:
        raise HTTPException(status_code=503, detail="Features not loaded")
    
    return {
        "total_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "sample_features": FEATURE_NAMES[:5]
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(input_data: PredictionInput):
    """
    Predict credit risk for a single customer
    
    Accepts scaled features as input.
    Returns risk probability, category, and credit score.
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Convert input to dictionary
        input_dict = input_data.dict()
        customer_id = input_dict.pop("customer_id", None)
        
        # Create DataFrame
        features_df = pd.DataFrame([input_dict])
        
        # Ensure all required features are present
        for feature in FEATURE_NAMES:
            if feature not in features_df.columns:
                features_df[feature] = 0.0  # Default value for missing features
        
        # Reorder columns to match training data
        features_df = features_df[FEATURE_NAMES]
        
        # Scale features
        features_scaled = SCALER.transform(features_df)
        
        # Make prediction
        risk_probability = MODEL.predict_proba(features_scaled)[0][1]
        
        # Calculate credit score (300-850 range)
        credit_score = int(300 + (1 - risk_probability) * 550)
        
        # Determine risk category
        if risk_probability < 0.3:
            risk_category = "Low"
        elif risk_probability < 0.7:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        return PredictionOutput(
            customer_id=customer_id,
            risk_probability=float(risk_probability),
            risk_category=risk_category,
            credit_score=credit_score,
            model_used="logistic_regression",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict credit risk for multiple customers
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for customer in batch_input.customers:
            input_dict = customer.dict()
            customer_id = input_dict.pop("customer_id", None)
            
            # Create DataFrame
            features_df = pd.DataFrame([input_dict])
            
            # Ensure all required features are present
            for feature in FEATURE_NAMES:
                if feature not in features_df.columns:
                    features_df[feature] = 0.0
            
            features_df = features_df[FEATURE_NAMES]
            features_scaled = SCALER.transform(features_df)
            
            risk_probability = MODEL.predict_proba(features_scaled)[0][1]
            credit_score = int(300 + (1 - risk_probability) * 550)
            
            if risk_probability < 0.3:
                risk_category = "Low"
            elif risk_probability < 0.7:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            predictions.append({
                "customer_id": customer_id,
                "risk_probability": float(risk_probability),
                "risk_category": risk_category,
                "credit_score": credit_score,
                "model_used": "logistic_regression"
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "model": "logistic_regression"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/calculate-credit-limit")
async def calculate_credit_limit(risk_probability: float, annual_income: float):
    """
    Calculate credit limit based on risk probability and income
    """
    if risk_probability < 0.3:
        # Low risk: up to 3 months income
        limit_multiplier = 0.25
    elif risk_probability < 0.7:
        # Medium risk: up to 2 months income
        limit_multiplier = 0.17
    else:
        # High risk: up to 1 month income
        limit_multiplier = 0.08
    
    credit_limit = annual_income * limit_multiplier
    
    return {
        "risk_probability": risk_probability,
        "annual_income": annual_income,
        "credit_limit": round(credit_limit, 2),
        "limit_multiplier": limit_multiplier,
        "recommendation": "approve" if risk_probability < 0.7 else "review"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)