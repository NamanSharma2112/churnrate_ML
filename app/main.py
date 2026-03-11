from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.endpoints import predict
from app.services.prediction_service import prediction_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models on startup
    prediction_service.load_model()
    yield
    print("Shutting down model server...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict if a customer will churn based on their features.",
    version="1.1.0",
    lifespan=lifespan
)

# Include the prediction router
app.include_router(predict.router, tags=["Predictions"])

@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Customer Churn Prediction API. See /docs for the swagger playground."
    }
