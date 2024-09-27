ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_S3_DATA_NAME: str = "file.zip"

DATA_BUCKET_NAME = "isd-complete"



"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ["test", "train", "valid", "data.yaml"]



"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_NAME: str = "yolov8x.pt"

MODEL_TRAINER_NO_EPOCHS: int = 1

MODEL_TRAINER_BATCH_SIZE: int = 8

MODEL_TRAINER_IMAGE_SIZE: int = 224




"""
MODEL PUSHER related constant start with MODEL_PUSHER var name
"""
MODEL_BUCKET_NAME = "isd-complete"
S3_MODEL_NAME = "best.pt"


"""
Model Evaluation related constant start with MODEL_EVALUATION var name
"""

MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"

MODEL_EVALUATION_METRICS_FILE_NAME: str = "metrics.json"

MODEL_EVALUATION_BATCH_SIZE: int = 8

MODEL_EVALUATION_IMAGE_SIZE: int = 640

MODEL_EVALUATION_DATA_YAML = "data.yaml"