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
from torch.optim import AdamW
import numpy as np
import psutil
import GPUtil
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import evaluate
import bitsandbytes as bnb
from adabound import AdaBound

# Import hybrid optimizer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.hybrid_adam_sgd import AdamSGDHybrid

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
        "adabound": AdaBound,
        "hybrid": AdamSGDHybrid
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")

    return optimizers[optimizer_name]


def get_device():
    """Detect and return the appropriate device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda", "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "MPS (Apple Silicon)"
    else:
        return "cpu", "CPU"


def get_gpu_metrics():
    """Get current GPU/accelerator metrics."""
    metrics = {}

    try:
        # Try CUDA metrics first (NVIDIA GPUs)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            metrics = {
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal,
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_utilization_percent": gpu.load * 100,
                "gpu_temperature": gpu.temperature
            }
            return metrics
    except Exception:
        pass

    # For MPS (Apple Silicon), use torch memory stats
    device, device_name = get_device()
    if device == "mps":
        try:
            # MPS doesn't provide detailed metrics, but we can track allocated memory
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory() / (1024**2)  # Convert to MB
                metrics = {
                    "gpu_memory_used_mb": allocated,
                    "gpu_memory_total_mb": 24576,  # M4 Pro unified memory (approximate)
                    "gpu_memory_percent": (allocated / 24576) * 100,
                    "device": "MPS"
                }
        except Exception as e:
            logger.warning(f"Failed to get MPS metrics: {e}")

    return metrics


def get_system_metrics():
    """Get current system metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": memory.used / (1024**3),
            "ram_percent": memory.percent
        }
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
    return {}


class MetricsTrackingCallback(TrainerCallback):
    """Callback to track GPU and system metrics during training."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.metrics_log = []
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        current_time = time.time() - self.start_time

        # Collect metrics
        metrics = {
            "timestamp": current_time,
            "step": state.global_step,
            "epoch": state.epoch,
        }

        # Add training logs
        if logs:
            metrics.update(logs)

        # Add GPU metrics
        gpu_metrics = get_gpu_metrics()
        metrics.update(gpu_metrics)

        # Add system metrics
        system_metrics = get_system_metrics()
        metrics.update(system_metrics)

        self.metrics_log.append(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Save metrics log
        metrics_path = Path(self.output_dir) / "gpu_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info(f"GPU metrics saved to: {metrics_path}")


def create_prompt(example: Dict[str, Any]) -> str:
    """Create a formatted prompt from a CommonsenseQA example."""
    # Handle different data structures
    question_data = example.get("question", "")

    # If question is a dict with 'stem' and 'choices'
    if isinstance(question_data, dict):
        question = question_data["stem"]
        choices = question_data["choices"]

        # Format choices as A, B, C, D, E
        choice_text = "\n".join([
            f"{chr(65 + i)}. {choice['text']}"
            for i, choice in enumerate(choices)
        ])
    else:
        # If question is already a string, use it directly
        # Also get choices from top level if available
        question = question_data
        choices_data = example.get("choices", {})

        if isinstance(choices_data, dict) and "text" in choices_data:
            # Choices is a dict with 'text' and 'label' lists
            choice_text = "\n".join([
                f"{choices_data['label'][i]}. {choices_data['text'][i]}"
                for i in range(len(choices_data["text"]))
            ])
        else:
            # No structured choices available, just use question
            choice_text = ""

    if choice_text:
        prompt = f"{question}\n{choice_text}\nAnswer:"
    else:
        prompt = f"{question}\nAnswer:"

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

            # Add choices if available
            if "choices" in examples:
                example["choices"] = examples["choices"][i]

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
    """Setup model and tokenizer with device-appropriate configuration."""
    logger.info("Loading model and tokenizer...")

    # Detect device
    device, device_name = get_device()
    logger.info(f"Using device: {device_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device-specific model loading
    if device == "cuda":
        # CUDA: Use 8-bit quantization for memory efficiency
        logger.info("Loading with 8-bit quantization for CUDA...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)

    elif device == "mps":
        # MPS (Apple Silicon): 8-bit quantization not supported, use float16
        logger.info("Loading for MPS (Apple Silicon) without quantization...")
        logger.warning("8-bit quantization not available on MPS. Using float16 instead.")
        logger.warning("This will use more memory (~14GB for Qwen2.5-7B).")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,  # Use float16 for MPS
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    else:
        # CPU: Load in float32 (slower but compatible)
        logger.warning("Loading on CPU. This will be very slow.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float32,
        )

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
        eval_strategy="epoch",  # Updated from evaluation_strategy
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
        elif optimizer_name == "hybrid":
            optimizer_kwargs["beta1"] = 0.9
            optimizer_kwargs["beta2"] = 0.999
            optimizer_kwargs["momentum"] = 0.9
            optimizer_kwargs["weight_decay"] = 0.01
            optimizer_kwargs["transition_steps"] = 1000
            optimizer_kwargs["final_ratio"] = 0.1

        return optimizer_class(model.parameters(), **optimizer_kwargs)

    # Create metrics tracking callback
    metrics_callback = MetricsTrackingCallback(output_dir)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer_init(), None),  # (optimizer, scheduler)
        callbacks=[metrics_callback]
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save final model
    trainer.save_model(f"{output_dir}/final")
    
    # Get final GPU metrics
    final_gpu_metrics = get_gpu_metrics()
    final_system_metrics = get_system_metrics()

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
        "peak_gpu_memory_mb": final_gpu_metrics.get("gpu_memory_used_mb", None),
        "peak_gpu_memory_percent": final_gpu_metrics.get("gpu_memory_percent", None),
        "final_gpu_utilization": final_gpu_metrics.get("gpu_utilization_percent", None),
    }

    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best evaluation accuracy: {trainer.state.best_metric:.4f}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model with different optimizers")
    parser.add_argument("--optimizer",
                       choices=["adamw", "sgd", "adabound", "hybrid"],
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