# clinical_summary
Complete and accurate clinical documentation is crucial for monitoring patient care. Physicians are tasked with documentation of clinic visit, but the traditional methods of a physician manually writing a report have led to their increased workload, reduced interaction time with patients, and a diminished work-life balance.

Therefore, this project employs Huggingface's Falconsai T5 transformer model to summarize clinical patient reports.

### Dataset
This model is trained on the Huggingface dataset: https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts.

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
├── config.yaml        <- It serves as a central configuration file for the project to ensure that all necessary paths and URLs are
|                         easily accessible and configurable. A YAML file is used to create data configuration files. It's recommended 
|                         that configuration files be written in YAML rather than JSON, even though they can be used interchangeably in
|                         most cases, because YAML has better readability and is more user-friendly.
│
├── params.yaml        <- stores easily accessible model tuning parameters.
│
├── Dockerfile         <- A text document that contains all the instructions to assemble a Docker image.
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── clinical_summary   <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes clinical_summary a Python module and defines some commonly used functions
    │
    ├── config.py      <- Defines all necessary paths and URLs.
    └── pipeline       <- Scripts to run the 5 stage model pipeline.
        |
        ├── __init__.py    <- Makes pipeline a Python module.
        ├── stage_01_data_ingestion.py      <- script for downloading input dataset.
        ├── stage_02_data_validation.py     <- script for validating if input dataset has correct divisions for train, test and validation.
        ├── stage_03_data_transformation.py <- script for preparing the dataset into 3 divisions and encoding them in token forms.
        ├── stage_04_model_training.py      <- script for training the model.
        ├── stage_05_model_evaluation.py    <- script for evaluating the model performance. 
        └── pipeline.py                     <- script for running the model on test dataset. 

 

```
## Setting up a virtual environment
#### Create a new virtual environment
python -m venv venv

#### Activate the new virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

#### Install the required packages
pip install -r requirements.txt


## Reference
I have learnt about the model training pipeline and GitHub actions deployment concepts from the tutorial by Bappy Ahmed: https://www.youtube.com/watch?v=p7V4Aa7qEpw.
--------

