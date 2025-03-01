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

        # Log do modelo no MLflow de uma forma que permita registro
        mlflow.keras.log_model(model, "model")
        
        mlflow.log_artifact(model_path)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)

        print("Run ID:", run.info.run_id)
        
        # Registrar o modelo diretamente
        model_name = "LibrasRecognitionModel"
        registered_model = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        print(f"Modelo registrado como: {model_name} versão {registered_model.version}")
        
        return run.info.run_id, model_name