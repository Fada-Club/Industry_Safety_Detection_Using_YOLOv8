import os
import sys
import yaml
from isd.utils.main_utils import read_yaml_file
from six.moves import urllib
from isd.logger import logging
from isd.exception import isdException
from isd.constant.training_pipeline import *
from isd.entity.config_entity import ModelTrainerConfig
from isd.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO
import shutil

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Create directory for unzipped data
            unzip_dir = "isd_dataset"
            if not os.path.exists(unzip_dir):
                os.makedirs(unzip_dir)
                logging.info(f"Created directory {unzip_dir} for unzipped data")

            # Unzipping data into the specified folder
            logging.info("Unzipping data")
            unzip_command = f"unzip {DATA_INGESTION_S3_DATA_NAME} -d {unzip_dir}"
            os.system(unzip_command)
            logging.info(f"Data unzipped to {unzip_dir}")

            # Optionally, remove the zip file if no longer needed
            os.system(f"rm {DATA_INGESTION_S3_DATA_NAME}")
            logging.info(f"Removed {DATA_INGESTION_S3_DATA_NAME} after unzipping")

            # YOLOv8 Training setup
            logging.info("Starting YOLOv8 training")

            model = YOLO(self.model_trainer_config.model_name)  # Load YOLOv8 model
            data_yaml_path = os.path.join(unzip_dir, "data.yaml")  # Path to data.yaml

            # Custom directory for saving weights and yolov8x.pt inside `isd_dataset`
            weights_dir = os.path.join(unzip_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            # Start training the YOLOv8 model with correct arguments
            model.train(
                data=data_yaml_path,
                imgsz=self.model_trainer_config.image_size,
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                project=self.model_trainer_config.model_trainer_dir,
                name="yolov8_training",
                workers=0
            )

            logging.info("YOLOv8 training completed")

            # Path to the best trained model
            trained_model_path = os.path.join(
                self.model_trainer_config.model_trainer_dir, "yolov8_training", "weights", "best.pt"
            )

            # Create "model" directory if it doesn't exist and copy the best.pt there
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy(trained_model_path, os.path.join(model_dir, "best.pt"))
            logging.info(f"Copied best.pt to {model_dir}")

            # Remove `yolov8x.pt` if downloaded inside the `isd_dataset` folder
            yolov8_weights_path = "yolov8x.pt"
            if os.path.exists(yolov8_weights_path):
                os.remove(yolov8_weights_path)
                logging.info(f"Removed {yolov8_weights_path}")

            # Remove the entire 'isd_dataset' folder
            # shutil.rmtree(unzip_dir)
            # logging.info(f"Removed the {unzip_dir} directory")

            # Create the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=os.path.join(model_dir, "best.pt")
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise isdException(e, sys)