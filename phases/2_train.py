#!/usr/bin/env python3
"""
Phase 2: LoRA Fine-tuning
Fine-tunes Qwen3-8B using different optimizers (AdamW, SGD+Momentum, AdaBound).
"""

import argparse
import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.optimization import AdamW
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import evaluate
import bitsandbytes as bnb
from adabound import AdaBound

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SGDMomentum(torch.optim.SGD):
    """SGD with momentum wrapper for consistency."""
    def __init__(self, params, lr=1e-3, momentum=0.9, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)

def get_optimizer_class(optimizer_name: str):
    """Get optimizer class by name."""
    optimizers = {
        "adamw": AdamW,
        "sgd": SGDMomentum,
        "adabound": AdaBound
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")
    
    return optimizers[optimizer_name]

def create_prompt(example: Dict[str, Any]) -> str:
    """Create a formatted prompt from a CommonsenseQA example."""
    question = example["question"]["stem"]
    choices = example["question"]["choices"]
    
    # Format choices as A, B, C, D, E
    choice_text = "\n".join([
        f"{chr(65 + i)}. {choice['text']}" 
        for i, choice in enumerate(choices)
    ])
    
    prompt = f"{question}\n{choice_text}\nAnswer:"
    return prompt

def preprocess_dataset(dataset, tokenizer, max_length=1024):
    """Preprocess the dataset for training."""
    logger.info("Preprocessing dataset...")
    
    def preprocess_function(examples):
        # Create prompts and targets
        prompts = []
        targets = []
        
        for i in range(len(examples["id"])):
            example = {
                "question": examples["question"][i],
                "answerKey": examples["answerKey"][i]
            }
            
            prompt = create_prompt(example)
            target = example["answerKey"]
            
            # Combine prompt and target
            full_text = f"{prompt} {target}"
            prompts.append(full_text)
            targets.append(target)
        
        # Tokenize
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # Set labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Process dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info(f"Processed {len(processed_dataset)} examples")
    return processed_dataset

def setup_model_and_tokenizer(model_path: str):
    """Setup model and tokenizer with quantization."""
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=8,                              # Rank
        lora_alpha=32,                   # Alpha parameter
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],  # Target modules for Qwen
        lora_dropout=0.05,              # Dropout
        bias="none",                     # Bias type
        task_type=TaskType.CAUSAL_LM    # Task type
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Get the predicted tokens (last position logits)
    predictions = np.argmax(predictions, axis=-1)
    
    # Simple accuracy computation (this is a simplified version)
    # In practice, you'd want more sophisticated evaluation
    accuracy = np.mean(predictions.flatten() == labels.flatten())
    
    return {"accuracy": accuracy}

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    optimizer_name: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    seed: int = 42
):
    """Train the model with specified optimizer."""
    logger.info(f"Starting training with {optimizer_name} optimizer...")
    
    # Set seeds
    set_seed(seed)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=20,
        logging_dir=f"{output_dir}/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Get optimizer class
    optimizer_class = get_optimizer_class(optimizer_name)
    
    # Custom optimizer instantiation
    def optimizer_init():
        optimizer_kwargs = {"lr": learning_rate}
        
        if optimizer_name == "sgd":
            optimizer_kwargs["momentum"] = 0.9
        elif optimizer_name == "adabound":
            optimizer_kwargs["final_lr"] = 0.1
            optimizer_kwargs["gamma"] = 1e-3
        
        return optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer_init(), None)  # (optimizer, scheduler)
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save final model
    trainer.save_model(f"{output_dir}/final")
    
    # Save training metrics
    metrics = {
        "optimizer": optimizer_name,
        "training_time": training_time,
        "final_loss": trainer.state.log_history[-1].get("train_loss", None),
        "best_eval_accuracy": trainer.state.best_metric,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best evaluation accuracy: {trainer.state.best_metric:.4f}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model with different optimizers")
    parser.add_argument("--optimizer", 
                       choices=["adamw", "sgd", "adabound"],
                       required=True,
                       help="Optimizer to use")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_path", default="./models/qwen3_8b", help="Path to base model")
    parser.add_argument("--data_path", default="./data/commonsense_qa", help="Path to dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                       help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Phase 2: LoRA Fine-tuning with {args.optimizer}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    
    # Preprocess dataset
    train_dataset = preprocess_dataset(dataset["train"], tokenizer, args.max_length)
    eval_dataset = preprocess_dataset(dataset["validation"], tokenizer, args.max_length)
    
    # Train model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer_name=args.optimizer,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )
    
    # Save LoRA adapter
    adapter_path = f"{args.output_dir}/adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    logger.info(f"✓ Training completed successfully")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"LoRA adapter saved to: {adapter_path}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 