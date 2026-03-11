from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import CustomerData, PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict churn risk for a single customer.
    """
    try:
        customer_dict = customer.model_dump()
        result = prediction_service.predict_single(customer_dict)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(customers: List[CustomerData]):
    """
    Predict churn risk for a batch of customers.
    """
    try:
        customer_dicts = [c.model_dump() for c in customers]
        results = prediction_service.predict_batch(customer_dicts)
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
