import warnings
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
from pathlib import Path

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()

def create_data_generators(data_dir, batch_size):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_data = datagen.flow_from_directory(
        data_dir, target_size=(64, 64), batch_size=batch_size, 
        class_mode="categorical", subset="training"
    )
    val_data = datagen.flow_from_directory(
        data_dir, target_size=(64, 64), batch_size=batch_size, 
        class_mode="categorical", subset="validation"
    )
    return train_data, val_data

def create_model(input_shape, num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def setup_mlflow():
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("tensorflow_libras_experiment")
    mlflow.keras.autolog()

def train_and_log_model(model, train_data, val_data, args):
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model": "CNN",
            "framework": "TensorFlow",
            "dataset": "libras/hands",
            "release.version": "1.0"
        })
        model.fit(train_data, epochs=args.epochs, validation_data=val_data)

        loss, accuracy = model.evaluate(val_data)
        print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}")

        model_path = "models/libras_model.h5"
        Path("models").mkdir(exist_ok=True)
        model.save(model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)

        print("Run ID:", run.info.run_id)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()

    data_dir = "hands/"
    train_data, val_data = create_data_generators(data_dir, args.batch_size)

    model = create_model(input_shape=(64, 64, 3), num_classes=train_data.num_classes, learning_rate=args.learning_rate)

    setup_mlflow()
    train_and_log_model(model, train_data, val_data, args)
