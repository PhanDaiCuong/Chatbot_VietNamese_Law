import os
from getpass import getpass
import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
import torch
from finetune import 
    
class TrainerClass:
    def __init__(self, base_model_id, output_dir, data_id, max_seq_length, dtype = None, ):
        # Initialize paths and directories
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.data_id = data_id
        self.max_seq_length = max_seq_length
        self.dtype = dtype, 
        # Define system message template
        self.system_message = """Bạn là một trợ lý thông minh, bạn được giao một vấn đề. Hãy suy nghĩ về vấn đề và đưa ra câu trả lời câu hỏi hiện tại của user. 
                            Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính. 
    
        """

    def create_conversation(self, examples, EOS_TOKEN):
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": examples["question"]},
                {"role": "assistant", "content": examples["context"] + EOS_TOKEN}
            ]
        }

    def load_datasets(self):
        dataset = load_dataset(self.data_id)
        dataset = dataset.map(
            self.create_conversation,
            batched=True,
        )
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]
        return train_dataset, test_dataset


    def config_model_lora(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=self.base_model_id,
        max_seq_length=self.max_seq_length,
        dtype=True,
        load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        return model, tokenizer

     
    def setup_training_args(self):
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,                 # directory to save model and logs
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=5,  # Set this for 1 full training run, while commenting out 'max_steps'.
            # max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="comet_ml" if enable_comet else "none",                                     # report metrics to tensorboard
        )
        return training_args

    def train_model(self):
        # Load datasets
        train_dataset, test_dataset = self.load_datasets()


        # Load model lora config
        self.model, self.tokenizer = self.config_model_lora()

        # Set up training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset, 
            eval_dataset=test_dataset,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            dataset_num_proc=2,
            packing=True,
        )

        # Start training
        trainer.train()

if __name__ == "__main__":
    base_model_id, output_dir, data_id, max_seq_length, dtype = None, 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="1TuanPham/T-VisStar-7B-v0.1")
    parser.add_argument("--output_dir", type=str, default="output_ckp")
    parser.add_argument("--data_id", type = str, default = "PhanDai/luat-viet-nam-qa")
    parser.add_argument("--max_seq_length", type = int, default = 4096)
    args = parser.parse_args()

    trainer_class = TrainerClass(
        args.base_model_id,
        args.output_dir,
        args.data_id,
        args.max_seq_length,
    )
    trainer_class.train_model()
    