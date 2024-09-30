# clinical_summary
Complete and accurate clinical documentation is crucial for monitoring patient care. Physicians are tasked with documentation of clinic visit, but the traditional methods of a physician manually writing a report have led to their increased workload, reduced interaction time with patients, and a diminished work-life balance.

Therefore, this project employs Huggingface's Falconsai T5 transformer model to summarize clinical patient reports. 
Evaluation is done based on Rouge scores: https://clementbm.github.io/theory/2021/12/23/rouge-bleu-scores.html 

### Dataset
This model is trained on the Huggingface dataset: https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts.

### Scores
Rouge scores are low as shown below:
|index|rouge1|rouge2|rougeL|rougeLsum|
|---|---|---|---|---|
|falconsai-finetuned|0\.0020408163265306124|0\.0|0\.0020408163265306124|0\.0020408163265306124|

But the summary outputs look correct when evaluated manually. Here are some examples:
#### sample 0
Reference Summary:
{'conditions': 'Psoriatic Arthritis', 'interventions': 'Drug: Adalimumab'}

Model Summary:
'conditions': 'Arthritis, Psoriatic', 'interventions'; 'Drug: Adalimumab'

#### sample 5
Reference Summary:
{'conditions': 'Suicide', 'interventions': 'Behavioral: Attachment-Based Family Therapy|Behavioral: Enhanced Usual Care'}

Model Summary:
'conditions': 'Suicidal Ideation|Depression', 'interventions'; 'Behavioral: Attachment Based Family Therapy (ABFT)'

#### sample 10
Reference Summary:
{'conditions': 'Acute Coronary Syndrome', 'interventions': 'Drug: Iodofiltic acid I 123'}

Model Summary:
'conditions': 'Coronary Artery Syndrome|Acute Coronary Syndrome|Chronic Insufficiency', 'interventions'. 'Drug: Iodofiltic Acid I 123'

## Project Organization
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- MIT Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               <- Contains sub-folders for data from each step of the model training pipeline.
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── logs               <- contains logs saved after running various scripts.
├── pyproject.toml     <- Project configuration file with package metadata for clinical_summary
│                         and configuration for tools like black
│
├── app.py             <- Developing a simple UI for users to input text and get summaries via a POST API.
│
├── main.py            <- Python file to train the model on training dataset.
│
├── config.yaml        <- It serves as a central configuration file for the project to ensure that all necessary
|                          paths and URLs are easily accessible and configurable. A YAML file is used to create data 
|                          configuration files. It's recommended that configuration files be written in YAML rather 
|                         than JSON, even though they can be used interchangeably in
|                         most cases, because YAML has better readability and is more user-friendly.
│
├── params.yaml        <- stores easily accessible model tuning parameters.
|
├── .github/workflows/main.yaml  <- CI/CD pipeline
│
├── Dockerfile         <- A text document that contains all the instructions to assemble a Docker image.
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── templates/index.html<- API for interacting with the model
|
├── static/style.css   <- stores CSS styling specifications
│
└── clinical_summary   <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes clinical_summary a Python module and defines some commonly used functions and classes
    │
    ├── config.py      <- Defines all necessary paths and URLs as per config.yaml and params.yaml file.
    └── pipeline       <- Scripts to run the 5 stage model training pipeline and model prediction.
        |
        ├── __init__.py    <- Makes pipeline a Python module.
        ├── stage_01_data_ingestion.py      <- script for downloading input dataset.
        ├── stage_02_data_validation.py     <- script for validating if input dataset has correct divisions for train, test and validation.
        ├── stage_03_data_transformation.py <- script for preparing the dataset into 3 divisions and encoding them in token forms.
        ├── stage_04_model_training.py      <- script for training the model.
        ├── stage_05_model_evaluation.py    <- script for evaluating the model performance. 
        └── pipeline.py                     <- script for running the model on test dataset. 

 

```
## Setting up the project environment
#### Create a new virtual environment
python -m venv venv

#### Activate the new virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

#### Install the required packages
pip install -r requirements.txt

## AWS deployment
### 1. Login to the AWS account

### 2. Create an IAM user for deployment
1. Build a Docker image of the source code
2. Push the Docker image to ECR to save it on AWS
3. Launch an EC2 virtual machine
4. Pull the Docker image from ECR to EC2
5. Launch the Docker image on EC2

Policies required for an IAM user:
1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess

### 3. Create ECR repo to store/save docker image
URI for ECR repository to save Docker image:
405074955098.dkr.ecr.ap-south-1.amazonaws.com/clinical_summary

### 4. Launch an EC2 Ubuntu machine instance

### 5. Start EC2 machine and install Docker
```
# optional
sudo apt-get update -y

sudo apt-get upgrade

# required
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

### 6. Configure the EC2 machine as a self-hosted runner
On Github repository settings page > actions > runner > new self hosted runner > choose OS as Ubuntu > then run commands listed on the runner webpage line by line in the EC2 machine console

If restarted the machine, just change directory to 'actions-runner' and run the GitHub Actions self-hosted runner using the command './run.sh'. To stop the runner execution, press Ctrl+C. Whenever you push some changes in your GitHub repo, the actions runner will deploy the changes in AWS machine.

### 7. Save Github secrets
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = ap-south-1

AWS_ECR_LOGIN_URI = 405074955098.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = clinical_summary