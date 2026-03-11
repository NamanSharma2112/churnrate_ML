from pydantic import BaseModel, Field

class CustomerData(BaseModel):
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
