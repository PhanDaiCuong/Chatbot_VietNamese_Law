import os
from getpass import getpass
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch 
    
class TrainerClass:
    def __init__(self, base_model_id, output_dir, data_id, max_seq_length, batch_train, batch_eval, num_step, dtype = None):
        # Initialize paths and directories
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.data_id = data_id
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.batch_train = batch_train
        self.batch_eval = batch_eval
        self.num_steps = num_step
        # Define system message template
        self.alpaca_prompt = """Dưới đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với thông tin đầu vào cung cấp thêm ngữ cảnh. Hãy viết phản hồi hoàn thành yêu cầu một cách phù hợp.
        ### Hướng dẫn:
        Bạn là một trợ lý thông minh, hãy trả lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan. Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
        ### Câu hỏi:
        {}

        ### Trả lời:
        {}"""

    def formatting_prompts_func(self, examples, EOS_TOKEN):
        inputs = examples["question"]              # list[str]
        outputs = examples["context"]              # list[str] hoặc list[list[str]]

        texts = []
        for input_text, output_text in zip(inputs, outputs):
            if isinstance(output_text, list):
                output_text = " ".join(output_text)  # nối lại nếu là list

            formatted_text = self.alpaca_prompt.format(input_text.strip(), output_text.strip()) + EOS_TOKEN
            texts.append(formatted_text)

        return {
            "text": texts  
        }

    def load_datasets(self, EOS_TOKEN):
        dataset = load_dataset(self.data_id)
        dataset = dataset.map(lambda x: self.formatting_prompts_func(x, EOS_TOKEN), batched=True,)
        
        return dataset


    def config_model_lora(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,
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
            per_device_train_batch_size=self.batch_train,
            per_device_eval_batch_size=self.batch_eval,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs=5,  # Set this for 1 full training run, while commenting out 'max_steps'.
            max_steps= self.num_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            # report_to="comet_ml" if enable_comet else "none",                                     # report metrics to tensorboard
        )
        return training_args

    def train_model(self):
    
        # Load model lora config
        model, tokenizer = self.config_model_lora()

        EOS_TOKEN = tokenizer.eos_token
        # Load datasets
        dataset = self.load_datasets(EOS_TOKEN)
        # Set up training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"], 
            eval_dataset=dataset["validation"],
            dataset_text_field="text",
            # max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=training_args,
        )

        # Start training
        print("Starting Train...")
        trainer.train()
        print("Ending Train ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--output_dir", type=str, default="output_ckp")
    parser.add_argument("--data_id", type = str, default = "PhanDai/luat-viet-nam-qa_small")
    parser.add_argument("--max_seq_length", type = int, default = 4096)
    parser.add_argument("--batch_train", type = int, default = 8)
    parser.add_argument("--batch_eval", type = int, default = 8)
    parser.add_argument("--num_steps", type = int, default = 25)
    
    args = parser.parse_args()

    trainer_class = TrainerClass(
        args.base_model_id,
        args.output_dir,
        args.data_id,
        args.max_seq_length,
        args.batch_train, 
        args.batch_eval, 
        args.num_steps
    )
    trainer_class.train_model()
    