from clinical_summary.config import ConfigurationManager
from clinical_summary import logger, ModelTrainingConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
from datasets import load_dataset, load_from_disk
import torch
import os

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_T5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=4,   # Rank of the adapter matrices
            lora_alpha=16,  # Scaling factor
            target_modules=["q", "v"],  # Targeting the query and value projection matrices of attention layers
            lora_dropout=0.1,  # Dropout for the LoRA layers
            bias="none"  # Do not apply LoRA to bias parameters
        )
        model_T5 = get_peft_model(model_T5, lora_config)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_T5)
        
        #loading data 
        dataset_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        ) 
        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
        #     per_device_train_batch_size=1, per_device_eval_batch_size=1,
        #     weight_decay=0.01, logging_steps=10,
        #     evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
        #     gradient_accumulation_steps=16
        # ) 

        trainer = Trainer(model=model_T5, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_pt["train"], 
                  eval_dataset=dataset_pt["validation"])
        
        trainer.train()

        ## Save model
        model_T5.save_pretrained(os.path.join(self.config.root_dir,"flan-T5-finetuned"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


class ModelTrainTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train()

        

        output_dir=self.config.root_dir
    