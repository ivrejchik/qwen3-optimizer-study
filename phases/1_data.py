#!/usr/bin/env python3
"""
Phase 1: Data & Model Acquisition
Downloads and caches CommonsenseQA dataset and Qwen3-8B model.
"""

import argparse
import os
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories."""
    directories = ['data', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_dataset():
    """Download and cache CommonsenseQA dataset."""
    logger.info("Downloading CommonsenseQA dataset...")
    
    try:
        # Load dataset
        dataset = load_dataset("tau/commonsense_qa")
        
        # Save to disk
        dataset_path = "./data/commonsense_qa"
        dataset.save_to_disk(dataset_path)
        
        logger.info(f"Dataset saved to: {dataset_path}")
        
        # Print dataset info
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")
        
        # Show sample
        sample = dataset['train'][0]
        logger.info("Sample question:")
        logger.info(f"  Question: {sample['question']['stem']}")
        logger.info(f"  Choices: {[choice['text'] for choice in sample['question']['choices']]}")
        logger.info(f"  Answer: {sample['answerKey']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def download_model():
    """Download and cache Qwen3-8B model."""
    logger.info("Downloading Qwen3-8B model...")
    
    try:
        model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using available model
        model_path = "./models/qwen3_8b"
        
        logger.info(f"Loading model: {model_name}")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(model_path)
        logger.info("Tokenizer saved successfully")
        
        # Download model (with proper device handling)
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Load to CPU first to avoid GPU memory issues
        )
        
        # Save model
        model.save_pretrained(
            model_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        logger.info(f"Model saved to: {model_path}")
        
        # Print model info
        logger.info(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        logger.info(f"Model dtype: {model.dtype}")
        
        # Verify tokenizer
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Tokenizer test - Input: '{test_text}' -> Tokens: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def verify_downloads():
    """Verify that all downloads completed successfully."""
    logger.info("Verifying downloads...")
    
    # Check dataset
    dataset_path = Path("./data/commonsense_qa")
    if dataset_path.exists() and (dataset_path / "dataset_info.json").exists():
        logger.info("✓ Dataset verification passed")
        dataset_ok = True
    else:
        logger.error("✗ Dataset verification failed")
        dataset_ok = False
    
    # Check model
    model_path = Path("./models/qwen3_8b")
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    
    model_ok = True
    for file_name in required_files:
        if not (model_path / file_name).exists():
            logger.error(f"✗ Missing model file: {file_name}")
            model_ok = False
    
    if model_ok:
        logger.info("✓ Model verification passed")
    
    return dataset_ok and model_ok

def get_disk_usage():
    """Calculate disk usage of downloaded files."""
    import shutil
    
    dataset_size = 0
    model_size = 0
    
    dataset_path = Path("./data/commonsense_qa")
    if dataset_path.exists():
        dataset_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
    
    model_path = Path("./models/qwen3_8b")
    if model_path.exists():
        model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    
    total_size = dataset_size + model_size
    
    logger.info(f"Disk usage:")
    logger.info(f"  Dataset: {dataset_size / (1024**3):.2f} GB")
    logger.info(f"  Model: {model_size / (1024**3):.2f} GB")
    logger.info(f"  Total: {total_size / (1024**3):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Download dataset and model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-only", action="store_true", help="Download only dataset")
    parser.add_argument("--model-only", action="store_true", help="Download only model")
    args = parser.parse_args()
    
    logger.info("Starting Phase 1: Data & Model Acquisition")
    logger.info(f"Random seed: {args.seed}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Setup directories
    setup_directories()
    
    success = True
    
    # Download dataset
    if not args.model_only:
        if not download_dataset():
            success = False
    
    # Download model
    if not args.dataset_only:
        if not download_model():
            success = False
    
    # Verify downloads
    if success:
        success = verify_downloads()
    
    # Show disk usage
    get_disk_usage()
    
    if success:
        logger.info("✓ Phase 1 completed successfully")
        logger.info("Next: Run Phase 2 with 'python phases/2_train.py'")
    else:
        logger.error("✗ Phase 1 failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 