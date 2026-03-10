from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from contextlib import asynccontextmanager

from churn_predictor import CustomerChurnPredictor

# Global variable to hold our predictor instance
predictor = None

# Define lifespan to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading Churn ML Model...")
    predictor = CustomerChurnPredictor()
    
    # Check if model files exist
    if not (os.path.exists("churn_model.pkl") and 
            os.path.exists("churn_scaler.pkl") and 
            os.path.exists("churn_encoders.pkl")):
        print("Model files not found! Please run 'python train_model.py' first.")
        # Alternatively, uncomment the next line to try generating and training them automatically
        # _auto_train(predictor)
    else:
        success = predictor.load_model()
        if not success:
            print("Failed to load existing model. You might need to retrain.")
            
    yield
    # Cleanup on shutdown (if any)
    print("Shutting down model server...")

# Helper to automatically train if missing
def _auto_train(predictor_instance: CustomerChurnPredictor):
    print("Auto-training a new model using sample data...")
    df = predictor_instance.load_data()
    X, y = predictor_instance.preprocess_data(df)
    predictor_instance.train_models(X, y)
    predictor_instance.save_model()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict if a customer will churn based on their features.",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------

class CustomerData(BaseModel):
    # Depending on feature_names, these are typically expected.
    # The ChurnPredictor has a robust matching system, but exact matches are best.
    age: int = Field(default=45, description="Age of the customer")
    gender: str = Field(default="Male", description="Gender (Male/Female)")
    tenure: int = Field(default=12, description="Number of months customer has stayed")
    phone_service: str = Field(default="Yes", description="Has phone service (Yes/No)")
    multiple_lines: str = Field(default="Yes", description="Has multiple lines")
    internet_service: str = Field(default="Fiber optic", description="Internet service type")
    online_security: str = Field(default="No", description="Has online security")
    online_backup: str = Field(default="No", description="Has online backup")
    device_protection: str = Field(default="No", description="Has device protection")
    tech_support: str = Field(default="No", description="Has tech support")
    streaming_tv: str = Field(default="Yes", description="Has streaming TV")
    streaming_movies: str = Field(default="Yes", description="Has streaming movies")
    contract: str = Field(default="Month-to-month", description="Contract duration")
    paperless_billing: str = Field(default="Yes", description="Uses paperless billing")
    payment_method: str = Field(default="Electronic check", description="Payment method")
    monthly_charges: float = Field(default=95.5, description="Monthly charges")
    total_charges: float = Field(default=1200.0, description="Total charges to date")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "gender": "Female",
                "tenure": 24,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "Yes",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "One year",
                "paperless_billing": "Yes",
                "payment_method": "Credit card",
                "monthly_charges": 85.5,
                "total_charges": 2052.0
            }
        }
    }

class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str

# -----------------------------------------------------
# Endpoints
# -----------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Customer Churn Prediction API. See /docs for the swagger playground."
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict churn risk for a single customer.
    """
    if not predictor or not predictor.best_model:
        raise HTTPException(status_code=503, detail="Model is not loaded or not trained yet. Please train the model and restart the server.")
    
    # Convert pydantic model to dictionary
    customer_dict = customer.model_dump()
    
    try:
        # The underlying predictor accepts a dictionary and returns a dictionary with keys:
        # 'churn_prediction', 'churn_probability', 'risk_level'
        result = predictor.predict_churn(customer_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(customers: List[CustomerData]):
    """
    Predict churn risk for a batch of customers.
    """
    if not predictor or not predictor.best_model:
        raise HTTPException(status_code=503, detail="Model is not loaded or not trained yet.")
    
    try:
        # Convert list of pydantic models to list of dictionaries
        # predictor.predict_churn supports passing a pandas DataFrame, 
        # so we can pass the list directly to the original method logic.
        import pandas as pd
        
        customer_dicts = [c.model_dump() for c in customers]
        batch_df = pd.DataFrame(customer_dicts)
        
        results = predictor.predict_churn(batch_df)
        
        # Ensure it's a list even if there's only 1 item in the batch
        if isinstance(results, dict):
            return [results]
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
