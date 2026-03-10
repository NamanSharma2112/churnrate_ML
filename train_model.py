from churn_predictor import CustomerChurnPredictor

def main():
    print("Initializing Predictor...")
    predictor = CustomerChurnPredictor()
    
    # Load or generate data
    print("Loading/generating data...")
    df = predictor.load_data()  # By default, generates sample data since we didn't provide a CSV path
    
    # Preprocess
    print("Preprocessing...")
    X, y = predictor.preprocess_data(df)
    
    # Train
    print("Training models...")
    predictor.train_models(X, y)
    
    # Evaluate best model
    predictor.evaluate_model()
    
    # Save the models to disk
    print("Saving best model...")
    predictor.save_model()

if __name__ == "__main__":
    main()
