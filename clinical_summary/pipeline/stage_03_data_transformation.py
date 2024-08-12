from clinical_summary.config import ConfigurationManager
from clinical_summary import logger, DataTransformationConfig
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def split_data(self,dataset):
        # Convert the dataset to a pandas DataFrame
        df = dataset['train'].to_pandas()

        # Function to extract text between 'Input:' and 'Output:'
        def extract_text_between(input_string):
            try:
                # Find the start of 'Input:' and end of 'Output:'
                start = input_string.index('Input:') + len('Input:')
                end = input_string.index('Output:')
                return input_string[start:end].strip()  # Extract and trim whitespace
            except ValueError:
                return None  # Return None if 'Input:' or 'Output:' is not found

        # Apply the function to the DataFrame column
        df['Question'] = df['Question'].apply(extract_text_between)


        # Split the data into train and temp sets (80% train, 20% temp)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

        # Further split the temp set into validation and test sets (50% of temp each, i.e., 10% of the original data each)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Convert DataFrames to Dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Combine into a DatasetDict
        split_dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        print(f"Data is split into train, test and validation subsets.")
        
        return split_dataset
    
    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['Question'], truncation = False, padding=True ) #max_length = 1024,
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['Answer'], max_length = 256,  truncation = False, padding=True) 
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        dataset = load_dataset(self.config.source_data_name)
        dataset = self.split_data(dataset)
        dataset_pt = dataset.map(self.convert_examples_to_features, batched = True)
        dataset_pt.save_to_disk(os.path.join(self.config.root_dir,"dataset"))




class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()