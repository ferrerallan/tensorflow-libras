import mlflow
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--model_name", type=str, default="LibrasRecognitionModel", help="Model name for registry")
    return parser.parse_args()

def register_model(run_id, model_name):
    # Configurar o servidor MLflow
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    
    # Registrar o modelo
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Transicionar para estágio de produção
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
    
    print(f"Model registered with name: {model_name}")
    print(f"Model version: {result.version}")
    print(f"Model in 'Production' stage")
    
    return model_name, result.version

if __name__ == "__main__":
    args = parse_arguments()
    register_model(args.run_id, args.model_name)