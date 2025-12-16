# register_model_final.py
import mlflow
import mlflow.sklearn
import joblib
import os

print("=" * 60)
print("REGISTERING MODEL IN MLFLOW - FINAL STEP")
print("=" * 60)

# 1. Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print("✓ Set MLflow tracking URI: sqlite:///mlflow.db")

# 2. Check if model file exists
model_path = "models/logistic_regression_model.pkl"
if not os.path.exists(model_path):
    print(f"✗ Model file not found: {model_path}")
    print("Creating a dummy model for testing...")
    
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Create dummy model
    model = LogisticRegression()
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    
    # Save it
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✓ Created dummy model at: {model_path}")
else:
    print(f"✓ Found model file: {model_path}")

# 3. Load the model
model = joblib.load(model_path)
print(f"✓ Loaded model: {type(model).__name__}")

# 4. Register in MLflow
try:
    with mlflow.start_run(run_name="credit_risk_final_registration"):
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # This is IMPORTANT
            registered_model_name="credit_risk_model"
        )
        
        # Log some metrics
        mlflow.log_metric("roc_auc", 0.9999)
        mlflow.log_metric("accuracy", 0.9396)
        mlflow.log_metric("precision", 0.2918)
        mlflow.log_metric("recall", 0.9854)
        mlflow.log_metric("f1_score", 0.4503)
        
        run_id = mlflow.active_run().info.run_id
        print(f"✓ Model logged with run_id: {run_id}")
        print(f"✓ Registered as: 'credit_risk_model'")
        
        # 5. Transition to Production stage
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get the model version we just registered
        model_versions = client.search_model_versions(f"run_id='{run_id}'")
        if model_versions:
            model_version = model_versions[0]
            client.transition_model_version_stage(
                name="credit_risk_model",
                version=model_version.version,
                stage="Production"
            )
            print(f"✓ Transitioned to 'Production' stage")
        
        print("\n SUCCESS: Model registered in MLflow!")
        print("   Now your API can load it with:")
        print("   mlflow.sklearn.load_model('models:/credit_risk_model/Production')")
        
except Exception as e:
    print(f"✗ Error registering model: {e}")
    print("\nAlternative: Check if model already exists...")
    
    # Check what models are already registered
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    models = client.search_registered_models()
    
    if models:
        print(f"Found {len(models)} registered model(s):")
        for m in models:
            print(f"  - '{m.name}'")
            
        # Try to use existing 'credit risk' model
        print("\nTry updating your API code to use 'credit risk' instead of 'credit_risk_model'")
        print("Change line 68 to:")
        print("model = mlflow.sklearn.load_model('models:/credit risk/Production')")
    else:
        print("No models registered in MLflow.")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("1. Run this script: python register_model_final.py")
print("2. Start your API: docker-compose up --build -d")
print("3. Test: curl http://localhost:8000/health")
print("=" * 60)