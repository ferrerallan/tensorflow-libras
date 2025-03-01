import argparse
import mlflow
import mlflow.sagemaker

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="LibrasRecognitionModel", help="Registered model name")
    parser.add_argument("--app_name", type=str, default="libras-recognition-app", help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--role_arn", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket for artifacts")
    parser.add_argument("--instance_type", type=str, default="ml.m5.large", help="Instance type")
    parser.add_argument("--instance_count", type=int, default=1, help="Number of instances")
    return parser.parse_args()

def deploy_to_sagemaker(args):
    # Configurar o servidor MLflow
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    
    # Definir o URI do modelo
    model_uri = f"models:/{args.model_name}/Production"
    
    print(f"Deploying model {model_uri} to SageMaker...")
    print(f"App name: {args.app_name}")
    print(f"Region: {args.region}")
    print(f"Role ARN: {args.role_arn}")
    print(f"Bucket: {args.bucket}")
    print(f"Instance type: {args.instance_type}")
    print(f"Instance count: {args.instance_count}")
    
    # Preparar configuração do deploy
    config = mlflow.sagemaker.SageMakerDeploymentClient.create_deployment(
        app_name=args.app_name,
        model_uri=model_uri,
        execution_role_arn=args.role_arn,
        region_name=args.region,
        mode="create",
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        bucket=args.bucket,
        synchronous=True
    )
    
    print(f"Model successfully deployed to SageMaker endpoint: {args.app_name}")
    print(f"You can now invoke this endpoint for predictions")
    return config

if __name__ == "__main__":
    args = parse_arguments()
    deploy_to_sagemaker(args)