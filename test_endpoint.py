import argparse
import boto3
import json
import numpy as np
from PIL import Image
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_name", type=str, default="libras-recognition-app", 
                       help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--image_path", type=str, required=True, 
                       help="Path to the test image")
    return parser.parse_args()

def preprocess_image(image_path):
    # Abrir e redimensionar a imagem para 64x64 (mesmo tamanho usado no treinamento)
    img = Image.open(image_path)
    img = img.resize((64, 64))
    # Converter para array e normalizar
    img_array = np.array(img) / 255.0
    # Garantir que a imagem esteja no formato correto (64, 64, 3)
    if len(img_array.shape) == 2:  # Para imagens em escala de cinza
        img_array = np.stack([img_array] * 3, axis=-1)
    return img_array.tolist()

def invoke_endpoint(args):
    # Configurar cliente SageMaker
    runtime = boto3.client('sagemaker-runtime', region_name=args.region)
    
    # Verificar se a imagem existe
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    # Preparar imagem
    try:
        img_array = preprocess_image(args.image_path)
        print(f"Image preprocessed successfully: {args.image_path}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return
    
    # Preparar payload
    payload = {"instances": [img_array]}
    
    try:
        # Fazer a previsão
        print(f"Invoking endpoint {args.endpoint_name}...")
        response = runtime.invoke_endpoint(
            EndpointName=args.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Processar resultado
        result = json.loads(response['Body'].read().decode())
        print("\nPrediction result:")
        print(json.dumps(result, indent=2))
        
        # Se o resultado for um array de probabilidades, mostrar a classe predita
        if isinstance(result, dict) and "predictions" in result:
            predictions = result["predictions"]
            if isinstance(predictions, list) and len(predictions) > 0:
                pred_class = np.argmax(predictions[0])
                print(f"\nPredicted class index: {pred_class}")
                print(f"Confidence: {predictions[0][pred_class]:.4f}")
    
    except Exception as e:
        print(f"Error invoking endpoint: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    invoke_endpoint(args)