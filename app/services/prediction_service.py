import os
from churn_predictor import CustomerChurnPredictor

class PredictionService:
    def __init__(self):
        self.predictor = None

    def load_model(self) -> bool:
        """Loads the saved ML models into memory."""
        print("Loading Churn ML Model...")
        self.predictor = CustomerChurnPredictor()
        
        # We look in the root directory (where churn_predictor.py saves them)
        if not (os.path.exists("churn_model.pkl") and 
                os.path.exists("churn_scaler.pkl") and 
                os.path.exists("churn_encoders.pkl")):
            print("Model files not found! Please run 'python train_model.py' first.")
            return False
            
        success = self.predictor.load_model()
        if not success:
            print("Failed to load existing model. You might need to retrain.")
            return False
            
        print("Model loaded successfully.")
        return True

    def predict_single(self, customer_dict: dict) -> dict:
        """Predict churn for a single customer dictionary."""
        if not self.predictor or not self.predictor.best_model:
            raise RuntimeError("Model is not loaded or not trained yet.")
        
        return self.predictor.predict_churn(customer_dict)

    def predict_batch(self, customer_dicts: list) -> list:
        """Predict churn for a batch of customer dictionaries."""
        if not self.predictor or not self.predictor.best_model:
            raise RuntimeError("Model is not loaded or not trained yet.")
        
        import pandas as pd
        batch_df = pd.DataFrame(customer_dicts)
        results = self.predictor.predict_churn(batch_df)
        
        if isinstance(results, dict):
            return [results]
        return results

# Singleton instance to be used across the app
prediction_service = PredictionService()
