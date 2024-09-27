import sys, os
from isd.logger import logging
from isd.exception import isdException
from isd.configuration.s3_operations import S3Operation
from isd.components.data_ingestion import DataIngestion
from isd.components.data_validation import DataValidation
from isd.components.model_trainer import ModelTrainer
from isd.components.model_evaluation import ModelEvaluation
from isd.components.model_pusher import ModelPusher


from isd.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig,
                                      ModelPusherConfig)


from isd.entity.artifacts_entity import (DataIngestionArtifact,
                                         DataValidationArtifact,
                                         ModelTrainerArtifact,
                                         ModelEvaluationArtifact,
                                         ModelPusherArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig() 
        self.model_pusher_config = ModelPusherConfig()
        self.s3_operations = S3Operation()


    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try: 
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise isdException(e, sys)
        

    

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise isdException(e, sys) from e
        

    
    def start_model_trainer(self
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise isdException(e, sys)
        

    def start_model_evaluation(
        self, model_trainer_artifact: ModelTrainerArtifact, data_ingestion_artifact: DataIngestionArtifact
    ) -> ModelEvaluationArtifact:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info("Performed the model evaluation operation")
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifact

        except Exception as e:
            raise isdException(e, sys)


    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact, s3: S3Operation):

        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact= model_trainer_artifact,
                s3=s3
                
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise isdException(e, sys)
        

    

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            if data_validation_artifact.validation_status == True:
                model_trainer_artifact = self.start_model_trainer()
                model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,data_ingestion_artifact=data_ingestion_artifact)
                model_pusher_artifact = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact,s3=self.s3_operations)
            
            else:
                raise Exception("Your data is not in correct format")


        except Exception as e:
            raise isdException(e, sys)