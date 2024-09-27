import os
import sys
import mlflow
import json
import shutil
import numpy as np
from ultralytics import YOLO
from isd.logger import logging
from isd.exception import isdException
from isd.constant.training_pipeline import *
from isd.entity.config_entity import ModelEvaluationConfig
from isd.entity.artifacts_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataIngestionArtifact


class ModelEvaluation:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise isdException(e, sys)

    def load_model(self):
        logging.info(f"Loading trained model from {self.model_trainer_artifact.trained_model_file_path}")

        try:
            # Load the trained YOLOv8 model
            model = YOLO(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"Loaded model from {self.model_trainer_artifact.trained_model_file_path}")
            return model
        except Exception as e:
            raise isdException(e, sys)

    def evaluate_model(self):
        logging.info("Starting model evaluation.")

        try:
            unzip_dir = "isd_dataset"
            self.model = self.load_model()
            
            # Evaluate model performance using YOLO's built-in validation method
            results = self.model.val(
                data=os.path.join(self.data_ingestion_artifact.feature_store_path, self.model_evaluation_config.data_yaml_path),
                imgsz=self.model_evaluation_config.image_size,
                batch=self.model_evaluation_config.batch_size
            )

            # Extract the metrics (updated to match the current YOLOv8 version structure)
            metrics = {
                'precision': results.box.p,   # Precision
                'recall': results.box.r,      # Recall
                'mAP_50': results.box.map50,  # mAP@0.50
                'mAP_50_95': results.box.map, # mAP@0.50 to 0.95
                'F1': results.box.f1,         # F1 score,
            }

            # Remove the entire 'isd_dataset' folder
            shutil.rmtree(unzip_dir)
            logging.info(f"Removed the {unzip_dir} directory")

            return metrics

        except Exception as e:
            raise isdException(e, sys)

    def initiate_model_evaluation(self):
        logging.info("Starting model evaluation...")
        try:
            mlflow.set_experiment("YOLOv8 Model Evaluation")

            with mlflow.start_run():
                # Evaluate on validation dataset
                logging.info("Evaluating on validation dataset")
                metrics = self.evaluate_model()

                # Log metrics to MLFlow
                logging.info("Logging metrics to MLFlow")
                for metric, value in metrics.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        value = float(np.mean(value))  # Convert array to mean float value
                    mlflow.log_metric(metric, value)

                # Convert all numpy arrays in metrics to lists or scalars for JSON serialization
                metrics_serializable = {
                    metric: value.tolist() if isinstance(value, np.ndarray) else value
                    for metric, value in metrics.items()
                }

                # Save metrics to a local JSON file
                evaluation_metrics_file_path = self.model_evaluation_config.metrics_file_path
                evaluation_metrics_dir = os.path.dirname(evaluation_metrics_file_path)
                os.makedirs(evaluation_metrics_dir, exist_ok=True)

                with open(evaluation_metrics_file_path, 'w') as f:
                    json.dump(metrics_serializable, f, indent=4)

                with open("metrics.json", 'w') as f:
                    json.dump(metrics_serializable, f, indent=4)
                
                logging.info(f"Metrics saved to {evaluation_metrics_file_path}")

                logging.info("Model evaluation completed and logged in MLFlow.")

                # Create ModelEvaluationArtifact
                model_evaluation_artifact = ModelEvaluationArtifact(
                    evaluated_model_metrics=metrics_serializable,
                    evaluation_metrics_file_path=self.model_evaluation_config.metrics_file_path
                )

                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                logging.info("Exited initiate_model_evaluation method of ModelEvaluation class")

                return model_evaluation_artifact

        except Exception as e:
            raise isdException(e, sys)