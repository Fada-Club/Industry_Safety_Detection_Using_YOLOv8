import os
import sys
import shutil
from isd.logger import logging
from isd.exception import isdException
from isd.entity.config_entity import DataValidationConfig
from isd.entity.artifacts_entity import (DataIngestionArtifact, DataValidationArtifact)


class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

        except Exception as e:
            raise isdException(e, sys)

    def validate_all_files_exist(self) -> bool:
        try:
            logging.info("Checking if all required files exist in the extracted dataset.")
            
            # Get the list of files in the extracted feature store path
            all_files_in_feature_store = os.listdir(self.data_ingestion_artifact.feature_store_path)

            # Ensure all required files (train, test, valid, data.yaml) are present
            missing_files = []
            for required_file in self.data_validation_config.required_file_list:
                if required_file not in all_files_in_feature_store:
                    missing_files.append(required_file)

            # Validation status for required files: True if no files are missing
            validation_status = len(missing_files) == 0

            # If any files are missing, log and return failure status
            if not validation_status:
                os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
                status_file_path = self.data_validation_config.valid_status_file_dir
                with open(status_file_path, 'w') as f:
                    f.write("Validation status: False\n")
                    f.write(f"Missing files: {', '.join(missing_files)}\n")
                logging.error(f"Missing required files: {', '.join(missing_files)}")
                return False

            # Step 2: Check if train, test, valid contain equal number of images and labels
            for dataset_type in ["train", "test", "valid"]:
                dataset_path = os.path.join(self.data_ingestion_artifact.feature_store_path, dataset_type)
                
                # Check if "images" and "labels" folders exist
                images_folder = os.path.join(dataset_path, "images")
                labels_folder = os.path.join(dataset_path, "labels")
                
                if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
                    logging.error(f"'images' or 'labels' folder missing in {dataset_type}.")
                    validation_status = False
                    break

                # Count the number of files in "images" and "labels"
                num_images = len(os.listdir(images_folder))
                num_labels = len(os.listdir(labels_folder))

                if num_images != num_labels:
                    logging.error(f"Number of images and labels do not match in {dataset_type}.")
                    logging.error(f"Images: {num_images}, Labels: {num_labels}")
                    validation_status = False
                    break

            # Write final validation status
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            status_file_path = self.data_validation_config.valid_status_file_dir

            with open(status_file_path, 'w') as f:
                if validation_status:
                    f.write("Validation status: True\n")
                    logging.info("All required files and folder structure are valid.")
                else:
                    f.write("Validation status: False\n")
                    logging.info("Validation failed due to missing files or mismatched counts.")

            return validation_status

        except Exception as e:
            raise isdException(e, sys)



    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            # Validate the presence of all required files
            status = self.validate_all_files_exist()

            # Create DataValidationArtifact with validation status
            data_validation_artifact = DataValidationArtifact(validation_status=status)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            # Optionally copy the zip file if validation is successful
            if status:
                shutil.copy(self.data_ingestion_artifact.data_zip_file_path, os.getcwd())

            return data_validation_artifact

        except Exception as e:
            raise isdException(e, sys)