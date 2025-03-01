import argparse
import boto3
import mlflow
import os
import tempfile
import shutil
import subprocess
import mlflow.tensorflow

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--app_name", type=str, default="libras-recognition-app", help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region")
    parser.add_argument("--role_arn", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket for artifacts")
    parser.add_argument("--instance_type", type=str, default="ml.m5.large", help="Instance type")
    return parser.parse_args()

def deploy_to_sagemaker(args):
    # Configurar o servidor MLflow
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    
    # Definir o URI do modelo usando o run_id
    model_uri = f"runs:/{args.run_id}/model"
    
    print(f"Loading model {model_uri} for SageMaker...")
    
    # Primeiro, carregar o modelo do MLflow
    loaded_model = mlflow.tensorflow.load_model(model_uri)
    
    # Criar diretório temporário para exportar o modelo
    tmp_dir = tempfile.mkdtemp()
    try:
        # Exportar o modelo para o formato esperado pelo SageMaker
        model_dir = os.path.join(tmp_dir, "model")
        print(f"Saving model to {model_dir}...")
        
        # Salvar o modelo carregado
        mlflow.tensorflow.save_model(
            tf_saved_model_dir=loaded_model,
            path=model_dir
        )
        
        # Comprimir o modelo em um arquivo tar.gz
        model_tar_path = os.path.join(tmp_dir, "model.tar.gz")
        print(f"Creating tar.gz archive at {model_tar_path}...")
        subprocess.run(["tar", "-czf", model_tar_path, "-C", model_dir, "."], check=True)
        
        # Fazer upload do arquivo tar.gz para o S3
        s3_client = boto3.client('s3', region_name=args.region)
        s3_model_path = f"models/{args.run_id}/model.tar.gz"
        model_s3_uri = f"s3://{args.bucket}/{s3_model_path}"
        
        print(f"Uploading model to {model_s3_uri}...")
        s3_client.upload_file(model_tar_path, args.bucket, s3_model_path)
        
        # Criar o endpoint SageMaker usando a AWS CLI
        print(f"Creating SageMaker endpoint {args.app_name}...")
        
        # Criar o modelo
        cmd = [
            "aws", "sagemaker", "create-model",
            "--model-name", f"{args.app_name}-model",
            "--primary-container", f'{{"Image": "763104351884.dkr.ecr.{args.region}.amazonaws.com/tensorflow-inference:2.12.0-cpu", "ModelDataUrl": "{model_s3_uri}"}}',
            "--execution-role-arn", args.role_arn,
            "--region", args.region
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Criar configuração de endpoint
        cmd = [
            "aws", "sagemaker", "create-endpoint-config",
            "--endpoint-config-name", f"{args.app_name}-config",
            "--production-variants", f'{{"VariantName": "variant-1", "ModelName": "{args.app_name}-model", "InstanceType": "{args.instance_type}", "InitialInstanceCount": 1}}',
            "--region", args.region
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Criar endpoint
        cmd = [
            "aws", "sagemaker", "create-endpoint",
            "--endpoint-name", args.app_name,
            "--endpoint-config-name", f"{args.app_name}-config",
            "--region", args.region
        ]