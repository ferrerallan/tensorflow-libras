# TensorFlow Libras

## Description

TensorFlow Libras is a machine learning project that uses a Convolutional Neural Network (CNN) to classify images of hand signs. This project is designed to train a TensorFlow model on a dataset of hand gestures (Libras dataset) and integrates MLflow for experiment tracking and model deployment. Once trained, the model can be deployed as a REST API for real-time inference.

- **Image Classification**: Classifies hand gestures into predefined classes using a TensorFlow CNN.
- **MLflow Integration**: Tracks experiments, logs model parameters, and saves trained models for deployment.
- **REST API Deployment**: Deploys the trained model as a REST API using MLflow's serving capabilities.
- **Preprocessing Pipeline**: Automatically preprocesses images for training and evaluation.
- **Test Script**: Includes a script for testing the deployed REST API with sample images.

## Requirements

### System Requirements

- Python 3.11 or higher
- MLflow tracking server
- Docker (optional for MLflow server)

### Python Dependencies

The project uses `Poetry` for dependency management. Key dependencies include:
- `tensorflow` (v2.18.0): For model building and training.
- `mlflow` (v2.19.0): For tracking and deploying machine learning models.

To install dependencies, run:
- poetry install

### MLflow Server

Set up an MLflow tracking server before running the project. You can either:
- Use the default local MLflow server at `http://localhost:5000`.
- Start a new MLflow server using Docker:
  - mlflow ui

### Environment Variables

To deploy the model, the `MLFLOW_TRACKING_URI` environment variable must point to the running MLflow server. Example:
- export MLFLOW_TRACKING_URI=http://localhost:5000

## Mode of Use

### Clone the Repository

Clone the repository and navigate to the project directory:
- git clone https://github.com/ferrerallan/tensorflow-libras.git
- cd tensorflow-libras

### Train the Model

Run the training script to train the model. Customize parameters like `epochs`, `batch_size`, and `learning_rate` using command-line arguments:
- python main.py --epochs 15 --batch_size 64 --learning_rate 0.0005

- **Training Data**: The script expects the dataset to be located in the `hands/` directory. The dataset should follow the structure:

hands/ ├── class_A/ │ ├── image1.jpg │ ├── image2.jpg ├── class_B/ │ ├── image3.jpg │ ├── image4.jpg

- **Output**: Trained models are saved in the `models/` directory and logged in MLflow.

### Deploy the Model

Once the model is trained, deploy it as a REST API using MLflow's serving capability:
- bash serve.sh

This command starts a REST API at `http://127.0.0.1:5001`.

### Test the Deployed Model

Use the `test_mlflow_server.py` script to test the REST API. Place a test image (e.g., `test_image.jpg`) in the project directory and run:
- python test_mlflow_server.py

Example Output:
The image was classified as: A Probability details: {'A': 0.85, 'B': 0.10, 'C': 0.05}



### Example API Request

You can also test the REST API with a direct `curl` request:
- curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[[...]]]}' http://127.0.0.1:5001/invocations

### Model Parameters

Default parameters can be adjusted via the command line:
- `--epochs`: Number of training epochs (default: 10).
- `--batch_size`: Batch size for training (default: 32).
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).

## Implementation Details

The project consists of several key components:

- **`main.py`**:
  - Prepares data using TensorFlow's `ImageDataGenerator` for image preprocessing and augmentation.
  - Defines and trains a CNN model using TensorFlow.
  - Logs experiments, parameters, and artifacts (trained models) to MLflow.

- **MLflow Integration**:
  - Tracks model training parameters, metrics, and artifacts.
  - Provides a simple mechanism to serve the trained model as a REST API.

- **`serve.sh`**:
  - Deploys the model using MLflow's `models serve` command.

- **`test_mlflow_server.py`**:
  - Loads and preprocesses a test image.
  - Sends a request to the deployed REST API and interprets the response.

- **Dataset**:
  - The `hands/` directory contains training and validation images structured by class.

## Example Workflow

- Train the Model:
  - python main.py --epochs 20 --batch_size 64 --learning_rate 0.0001
- Deploy the Model:
  - bash serve.sh
- Test the API:
  - python test_mlflow_server.py

