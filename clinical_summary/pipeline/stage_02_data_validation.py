from clinical_summary.config import ConfigurationManager
from clinical_summary import logger, DataValidationConfig
import os

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None
            all_files = os.listdir(os.path.join("data","data_ingestion","dataset"))
            print(all_files)

            # Ensure all required files are present
            missing_files = [file for file in self.config.ALL_REQUIRED_FILES if file not in all_files]
            print(missing_files)
            if missing_files:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation failed. Missing files: {', '.join(missing_files)}")
            else:
                validation_status = True
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write("Validation successful. All required files are present.")
            
            return validation_status
        
        except Exception as e:
            raise e

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()