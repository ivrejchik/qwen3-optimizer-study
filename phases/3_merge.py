#!/usr/bin/env python3
"""
Phase 3: Adapter Merging
Merges LoRA adapters back into full models for inference.
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_adapter(base_model_path: str, adapter_path: str, output_path: str, optimizer_name: str):
    """Merge a LoRA adapter into the base model."""
    logger.info(f"Merging {optimizer_name} adapter...")
    
    try:
        # Load base model
        logger.info(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load and merge adapter
        logger.info(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge adapter weights into base model
        logger.info("Merging adapter weights...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        
        # Get model size info
        param_count = sum(p.numel() for p in merged_model.parameters())
        model_size_gb = sum(
            f.stat().st_size for f in Path(output_path).rglob('*.safetensors')
        ) / (1024**3)
        
        logger.info(f"✓ {optimizer_name} merge completed")
        logger.info(f"  Parameters: {param_count / 1e9:.1f}B")
        logger.info(f"  Model size: {model_size_gb:.2f} GB")
        
        # Clean up GPU memory
        del merged_model, base_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge {optimizer_name} adapter: {e}")
        return False

def verify_merged_model(model_path: str, optimizer_name: str):
    """Verify that the merged model can be loaded and used."""
    logger.info(f"Verifying {optimizer_name} merged model...")
    
    try:
        # Test loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Test tokenization
        test_text = "What is the capital of France?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
        logger.info(f"✓ {optimizer_name} model verification passed")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"✗ {optimizer_name} model verification failed: {e}")
        return False

def create_model_info(model_path: str, optimizer_name: str):
    """Create a model info file with metadata."""
    info = {
        "optimizer": optimizer_name,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adaptation_method": "LoRA",
        "task": "CommonsenseQA",
        "merged": True,
        "framework": "transformers + peft"
    }
    
    info_path = Path(model_path) / "model_info.json"
    import json
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model info saved to: {info_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into full models")
    parser.add_argument("--base_model", default="./models/qwen3_8b", 
                       help="Path to base model")
    parser.add_argument("--experiments_dir", default="./experiments", 
                       help="Directory containing experiment results")
    parser.add_argument("--optimizers", nargs="+", 
                       default=["adamw", "sgd", "adabound"],
                       help="Optimizers to merge")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify merged models after creation")
    
    args = parser.parse_args()
    
    logger.info("Starting Phase 3: Adapter Merging")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Experiments directory: {args.experiments_dir}")
    logger.info(f"Optimizers: {args.optimizers}")
    
    # Check if base model exists
    if not Path(args.base_model).exists():
        logger.error(f"Base model not found: {args.base_model}")
        return 1
    
    success_count = 0
    total_optimizers = len(args.optimizers)
    
    # Merge each optimizer's adapter
    for optimizer in args.optimizers:
        adapter_path = f"{args.experiments_dir}/{optimizer}/adapter"
        output_path = f"{args.experiments_dir}/{optimizer}/merged"
        
        # Check if adapter exists
        if not Path(adapter_path).exists():
            logger.warning(f"Adapter not found for {optimizer}: {adapter_path}")
            continue
        
        # Skip if already merged
        if Path(output_path).exists():
            logger.warning(f"Merged model already exists for {optimizer}, skipping")
            continue
        
        # Merge adapter
        if merge_adapter(args.base_model, adapter_path, output_path, optimizer):
            # Create model info
            create_model_info(output_path, optimizer)
            
            # Verify if requested
            if args.verify:
                if verify_merged_model(output_path, optimizer):
                    success_count += 1
            else:
                success_count += 1
        
        logger.info(f"Progress: {success_count}/{total_optimizers} completed")
    
    # Summary
    logger.info("Phase 3 Summary:")
    logger.info(f"  Successfully merged: {success_count}/{total_optimizers}")
    
    if success_count == total_optimizers:
        logger.info("✓ All adapters merged successfully")
        logger.info("Next: Run Phase 4 with 'python phases/4_eval.py'")
        return 0
    else:
        logger.warning(f"Only {success_count}/{total_optimizers} adapters merged")
        return 1

if __name__ == "__main__":
    exit(main()) 