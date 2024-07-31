from clinical_summary.config import ConfigurationManager
from clinical_summary import get_size, logger, DataIngestionConfig
import os
import urllib.request as request
from datasets import load_dataset
import zipfile
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            dataset = load_dataset(self.config.source_data_name)
            dataset.save_to_disk(self.config.local_data_file)
            logger.info(f"Dataset {self.config.source_data_name} saved to {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

        
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()