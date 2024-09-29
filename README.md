# Industry Safety Detection Using YOLOv8

# Introduction

This project aims to enhance worker safety in industrial environments by developing an end-to-end MLOps pipeline for Industry Safety Detection using the YOLOv8 model. The model is trained to detect 10 different classes, including critical safety-related objects such as 'Hardhat', 'Mask', and 'Safety Vest'. By leveraging real-time object detection, this system provides an automated solution for identifying safety breaches on industrial sites, reducing the risk of workplace accidents.

The pipeline is designed to streamline the entire machine learning workflow, consisting of key stages like data ingestion, data validation, model training, and model evaluation. Data is automatically ingested and validated to ensure its quality before training the YOLOv8 model. The model evaluation is managed using MLflow, which tracks important metrics such as mean average precision (mAP) and facilitates comparison between model versions.

To ensure seamless deployment and scalability, the project incorporates a CI/CD pipeline using Docker images. These Docker containers encapsulate the entire model environment, making it easy to deploy the trained models on AWS. Specifically, we use AWS Elastic Container Registry (ECR) to store and manage Docker images, and the CI/CD pipeline automates the deployment process, ensuring that updates are continuously integrated and tested. This approach enables the project to maintain robust, real-time deployment capabilities in a cloud environment, supporting ongoing improvements and model updates.

## Tech Stack Used
1) Python
2) Flask
3) YOLOv8
4) Docker
5) MLFlow
6) PyTorch
7) OpenCV

## Infrastructure
1) DockerHub
2) AWS Elastic Container Registry (ECR)
3) GitHub
4) CI/CD pipeline

## System Design
![image](./assets/SystemDesign.png)

## Dataset

Dataset for this Project is taken from Kaggle. Here is the Dataset [Link](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow). It contains images of construction sites with various safety-related classes, making it suitable for training object detection models to identify potential safety hazards.

## Dataset Information

* Images are not in dcm format, the images are in jpg to fit the model.
* Data contain 3 folders which are train, test and valid.
* There are 10 classes to detect from the dataset:


'**Hardhat**', '**Mask**', '**NO-Hardhat**', '**NO-Mask**', '**NO-Safety Vest**', '**Person**', '**Safety Cone**', '**Safety Vest**', '**machinery**', '**vehicle**'

* test represent testing set
* train represent training set
* valid represent validation set
* training set is 2605 images
* testing set is 82 images
* validation set is 114 images


#### Dataset Details<a id='dataset-details'></a>
<pre>
Dataset Name            : Construction Site Safety Image Dataset Roboflow
Number of Class         : 10
Number/Size of Images   : Total      : 2801 (311 MB)
                          Training   : 2605
                          Testing    : 82
                          Validation : 114 

</pre>
## Results<a id='results-'></a>
We have achieved following results with YOLOv8x model for detection of the 10 clasess like ,'**Mask**', '**machinery**', '**Safety Vest**' and others from Construction Site Safety Images.

<pre>
<b>Performance Metrics </b>
mAP_50 Score                                     : 88.90%
mAP_50_95 Score                                  : 65.30%
</pre>

## Installation
    
The Code is written in Python 3.8.19. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

## Run Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/kdot313/Industry.git
```
### Step 2- Create a conda environment after opening the repository
```bash
conda create -p env python=3.8 -y
```
```bash
source activate ./env
```
### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Set Environment variables for MLFlow
```bash
export MLFLOW_TRACKING_URI=ttps://dagshub.com/kdot313/Industry.mlflow

export MLFLOW_TRACKING_USERNAME=kdot313

export MLFLOW_TRACKING_PASSWORD=d91b06fbd9b355c4da3eb05a4b538f21602d1421
```

### Step 5 - Create IAM user with following Permissions Enabled

* **AmazonEC2ContainerRegistryFullAccess**
* **AmazonEC2FullAccess**


### Step 6 - Configure your AWS
```bash
aws configure
```

### Step 7 - Enter your AWS Credentials of IAM User
```bash
AWS_SECRET_ACCESS_KEY = ""
AWS_ACCESS_KEY_ID = ""
AWS_REGION = "us-east-1"
AWS_FOLDER = Press Enter and move on
```

### Step 8 - Prepare your Dataset zip file named isd_data.zip
Your Zip file should contain following folders and files in this order:
```bash
isd_data.zip
│
├── train
│   ├── images
│   └── labels
│
├── test
│   ├── images
│   └── labels
│
├── valid
│   ├── images
│   └── labels
│
└── data.yaml

```

* **Ensure that the train, test, and valid directories contain their respective images and labels subfolders.**
* **Update the data.yaml file with the correct paths for train, test, and valid directories based on your system's file paths.**

### Step 8 - Upload the Dataset zip file in your S3 Bucket
```bash
AWS_SECRET_ACCESS_KEY = ""
AWS_ACCESS_KEY_ID = ""
AWS_REGION = "us-east-1"
AWS_FOLDER = Press Enter and move on
```

aws s3 cp path/to/your/file.zip s3://your-bucket-name/file.zip

aws s3 cp /workspaces/Industry/best.pt s3://isd-complete/best.pt