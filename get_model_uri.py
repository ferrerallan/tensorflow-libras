# get_model_uri.py
import mlflow
import sys

def get_model_uri(model_name, stage="Production"):
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    
    model_versions = client.get_latest_versions(model_name, stages=[stage])
    if model_versions:
        model_version = model_versions[0]
        model_uri = f"models:/{model_name}/{stage}"
        print(f"Model URI: {model_uri}")
        return model_uri
    else:
        print(f"No model found with name {model_name} in stage {stage}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_model_uri.py MODEL_NAME [STAGE]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else "Production"
    get_model_uri(model_name, stage)