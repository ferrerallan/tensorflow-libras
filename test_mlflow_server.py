import requests
import numpy as np
from PIL import Image


URL = "http://127.0.0.1:5001/invocations"
IMAGE_PATH = "test_image.jpg"  # Path to the test image
IMAGE_SHAPE = (64, 64, 3)  # Model input dimensions (64x64 with 3 channels)
CLASSES = ["A", "B", "C"]  # Corresponding classes for letters

def load_and_preprocess_image(image_path, target_shape):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
    image = image.resize(target_shape[:2])  # Resize to target dimensions
    image_array = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    return image_array[np.newaxis, ...].tolist()  # Add batch dimension

def create_payload(input_data):
    return {"inputs": input_data}

def send_request(url, payload):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response

def process_response(response):
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
        predicted_class = CLASSES[predicted_class_index]  # Map index to class
        return predicted_class, dict(zip(CLASSES, predictions[0]))
    else:
        raise ValueError(f"Error {response.status_code}: {response.text}")

def main():
    try:
        input_data = load_and_preprocess_image(IMAGE_PATH, IMAGE_SHAPE)
        payload = create_payload(input_data)
        response = send_request(URL, payload)
        predicted_class, probabilities = process_response(response)

        print(f"The image was classified as: {predicted_class}")
        print(f"Probability details: {probabilities}")
    except FileNotFoundError:
        print(f"Error: File '{IMAGE_PATH}' not found.")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
