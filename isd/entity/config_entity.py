import os
from dataclasses import dataclass
from datetime import datetime
from isd.constant.training_pipeline import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR,TIMESTAMP)


training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig() 


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )

    S3_DATA_NAME = DATA_INGESTION_S3_DATA_NAME



@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATUS_FILE)

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES




@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME
    )

    model_name = MODEL_TRAINER_NAME

    no_epochs = MODEL_TRAINER_NO_EPOCHS

    batch_size = MODEL_TRAINER_BATCH_SIZE

    image_size = MODEL_TRAINER_IMAGE_SIZE



@dataclass
class ModelPusherConfig:
    MODEL_BUCKET_NAME: str = MODEL_BUCKET_NAME
    S3_MODEL_KEY_PATH: str = S3_MODEL_NAME



@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_EVALUATION_DIR_NAME
    )

    metrics_file_path: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_METRICS_FILE_NAME)

    batch_size: int = MODEL_EVALUATION_BATCH_SIZE

    image_size: int = MODEL_EVALUATION_IMAGE_SIZE

    data_yaml_path: str = MODEL_EVALUATION_DATA_YAML