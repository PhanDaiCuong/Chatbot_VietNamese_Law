#!/usr/bin/env python
# merge_model.py
# Script to merge a fine-tuned model with the base model using merged_16bit saving method

import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a fine-tuned model checkpoint into a single merged-16bit model"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="Pretrained base model identifier or path"
    )
    parser.add_argument(
        "--tuned_checkpoint", type=str, required=True,
        help="Directory of the fine-tuned checkpoint (output folder)"
    )
    parser.add_argument(
        "--merged_model_name", type=str, required=True,
        help="Name for the merged model (local directory)"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None,
        help="Tokenizer identifier or path (defaults to base_model)"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face token for push_to_hub (optional)"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Whether to push merged model to the Hub"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine tokenizer source
    tokenizer_source = args.tokenizer_name if args.tokenizer_name else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    # Load the fine-tuned model checkpoint (assumes PeftModel or custom)
    # For many PEFT setups, you would load as PeftModel.from_pretrained(base, tuned_checkpoint)
    # Here, we assume direct load:
    model = AutoModelForCausalLM.from_pretrained(
        args.tuned_checkpoint,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Create local directory for merged model
    os.makedirs(args.merged_model_name, exist_ok=True)

    # Save merged weights in 16-bit
    model.save_pretrained_merged(
        args.merged_model_name,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved locally at: {args.merged_model_name}")

    # Optionally push to Hugging Face Hub
    if args.push_to_hub and args.hf_token:
        api = HfApi()
        user_info = api.whoami(token=args.hf_token)
        hf_user = user_info.get("name")
        repo_id = f"{hf_user}/{args.merged_model_name}"
        model.push_to_hub_merged(
            repo_id,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=args.hf_token,
        )
        print(f"Merged model pushed to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
