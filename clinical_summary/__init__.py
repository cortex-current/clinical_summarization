from dataclasses import dataclass
from pathlib import Path
import os, sys
from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
import logging
from box import ConfigBox
from typing import Any 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_name: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
    source_data_name: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path


# common libraries to import commonly throughout other code files
# ensuring that correct type annotations are present for the function arguments and return values, otherwise throws error
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type (can easily read python config values into python types)
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to false.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
            
@ensure_annotations
def get_size(path: Path) -> str:
    """
    get size in kB
    Args:
        path(Path): path of the file

    Returns:
        str: size in KB (1024 bytes, not kB as in 1000)
        A kilobyte (KB) is 1,024 bytes, not one thousand bytes as might be expected,
          because computers use binary (base two) math, instead of a decimal (base ten) system.
    """
    size_in_KB = round(os.path.getsize(path)/1024)
    return f"~ (size_in_KB) KB"

### Logging ==========================================================

# it will show message logs from the module file you are running
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers=[
        logging.FileHandler(log_filepath), #log in a file
        logging.StreamHandler(sys.stdout) # log in the terminal
    ]
)

logger = logging.getLogger("text_summarization_logger")