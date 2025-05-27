#!/usr/bin/env python3
"""
Phase 4: Model Evaluation
Evaluates all models (baseline + fine-tuned) on CommonsenseQA validation set.
"""

import argparse
import logging
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import evaluate
from tqdm import tqdm
import psutil
import GPUtil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        try:
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                return gpu.memoryUsed, gpu.memoryTotal
        except:
            pass
    return 0, 0

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

def evaluate_model(
    model_path: str,
    model_name: str,
    dataset,
    tokenizer_path: str = None,
    max_new_tokens: int = 1,
    batch_size: int = 1
) -> Dict[str, Any]:
    """Evaluate a single model on the dataset."""
    logger.info(f"Evaluating {model_name} model...")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    # Load model and tokenizer
    logger.info(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    # Ensure we have padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Evaluation metrics
    start_time = time.time()
    predictions = []
    true_labels = []
    total_examples = len(dataset)
    
    # Track resource usage
    gpu_memory_start, gpu_memory_total = get_gpu_memory()
    cpu_percent_start = psutil.cpu_percent(interval=1)
    
    logger.info(f"Processing {total_examples} examples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
            # Create prompt
            prompt = create_prompt(example)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(model.device)
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0
            )
            
            # Extract generated token
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Get prediction (first character should be A, B, C, D, or E)
            predicted_answer = generated_text[0].upper() if generated_text else "A"
            
            # Ensure prediction is valid
            if predicted_answer not in ["A", "B", "C", "D", "E"]:
                predicted_answer = "A"  # Default fallback
            
            predictions.append(predicted_answer)
            true_labels.append(example["answerKey"])
            
            # Log progress every 100 examples
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total_examples}")
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(predictions, true_labels)) / len(predictions)
    items_per_second = len(dataset) / evaluation_time
    
    # Resource usage
    gpu_memory_end, _ = get_gpu_memory()
    cpu_percent_end = psutil.cpu_percent(interval=1)
    
    # Calculate per-choice accuracy
    choice_accuracy = {}
    for choice in ["A", "B", "C", "D", "E"]:
        choice_preds = [p for p, l in zip(predictions, true_labels) if l == choice]
        if choice_preds:
            choice_accuracy[f"accuracy_{choice}"] = sum(p == choice for p in choice_preds) / len(choice_preds)
        else:
            choice_accuracy[f"accuracy_{choice}"] = 0.0
    
    # Prepare results
    results = {
        "model": model_name,
        "model_path": model_path,
        "accuracy": accuracy,
        "total_examples": total_examples,
        "correct_predictions": sum(p == l for p, l in zip(predictions, true_labels)),
        "evaluation_time_seconds": evaluation_time,
        "items_per_second": items_per_second,
        "gpu_memory_used_mb": gpu_memory_end,
        "gpu_memory_total_mb": gpu_memory_total,
        "cpu_percent_avg": (cpu_percent_start + cpu_percent_end) / 2,
        **choice_accuracy
    }
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Evaluation time: {evaluation_time:.2f}s")
    logger.info(f"  Speed: {items_per_second:.2f} items/sec")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def save_detailed_results(results: List[Dict[str, Any]], output_dir: str):
    """Save detailed results to JSON and CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    json_path = output_path / "detailed_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to: {json_path}")
    
    # Save to CSV
    csv_path = output_path / "results.csv"
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results CSV saved to: {csv_path}")
    
    return str(csv_path)

def create_summary_report(results: List[Dict[str, Any]], output_dir: str):
    """Create a summary report of the evaluation."""
    if not results:
        return
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    report = []
    report.append("# Qwen3 Optimizer Comparison - Evaluation Summary")
    report.append("")
    report.append(f"**Total Models Evaluated:** {len(results)}")
    report.append(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Best performing model
    best_model = sorted_results[0]
    report.append("## Best Performing Model")
    report.append(f"- **Model:** {best_model['model']}")
    report.append(f"- **Accuracy:** {best_model['accuracy']:.4f}")
    report.append(f"- **Speed:** {best_model['items_per_second']:.2f} items/sec")
    report.append("")
    
    # Results table
    report.append("## Results Summary")
    report.append("")
    report.append("| Model | Accuracy | Speed (items/s) | Eval Time (s) | GPU Memory (MB) |")
    report.append("|-------|----------|-----------------|---------------|-----------------|")
    
    for result in sorted_results:
        report.append(
            f"| {result['model']} | {result['accuracy']:.4f} | "
            f"{result['items_per_second']:.2f} | {result['evaluation_time_seconds']:.1f} | "
            f"{result['gpu_memory_used_mb']:.0f} |"
        )
    
    report.append("")
    
    # Performance analysis
    report.append("## Performance Analysis")
    
    if len(results) > 1:
        baseline = next((r for r in results if r["model"] == "baseline"), None)
        if baseline:
            report.append(f"### Improvement over Baseline")
            for result in sorted_results:
                if result["model"] != "baseline":
                    improvement = (result["accuracy"] - baseline["accuracy"]) * 100
                    report.append(f"- **{result['model']}:** {improvement:+.2f}% accuracy improvement")
        
        # Speed comparison
        fastest = max(results, key=lambda x: x["items_per_second"])
        slowest = min(results, key=lambda x: x["items_per_second"])
        speedup = fastest["items_per_second"] / slowest["items_per_second"]
        report.append(f"### Speed")
        report.append(f"- **Fastest:** {fastest['model']} ({fastest['items_per_second']:.2f} items/s)")
        report.append(f"- **Slowest:** {slowest['model']} ({slowest['items_per_second']:.2f} items/s)")
        report.append(f"- **Speedup:** {speedup:.2f}x")
    
    # Save report
    report_path = Path(output_dir) / "evaluation_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on CommonsenseQA")
    parser.add_argument("--data_path", default="./data/commonsense_qa", 
                       help="Path to dataset")
    parser.add_argument("--base_model", default="./models/qwen3_8b", 
                       help="Path to base model")
    parser.add_argument("--experiments_dir", default="./experiments", 
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", default="./results", 
                       help="Output directory for results")
    parser.add_argument("--optimizers", nargs="+", 
                       default=["adamw", "sgd", "adabound"],
                       help="Optimizers to evaluate")
    parser.add_argument("--include_baseline", action="store_true", 
                       help="Include baseline model evaluation")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    logger.info("Starting Phase 4: Model Evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.data_path)["validation"]
    
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} examples for testing")
    
    # Prepare models to evaluate
    models_to_evaluate = []
    
    # Add baseline model if requested
    if args.include_baseline:
        if Path(args.base_model).exists():
            models_to_evaluate.append(("baseline", args.base_model))
        else:
            logger.warning(f"Baseline model not found: {args.base_model}")
    
    # Add fine-tuned models
    for optimizer in args.optimizers:
        merged_path = f"{args.experiments_dir}/{optimizer}/merged"
        if Path(merged_path).exists():
            models_to_evaluate.append((optimizer, merged_path))
        else:
            logger.warning(f"Merged model not found for {optimizer}: {merged_path}")
    
    if not models_to_evaluate:
        logger.error("No models found to evaluate!")
        return 1
    
    logger.info(f"Evaluating {len(models_to_evaluate)} models: {[m[0] for m in models_to_evaluate]}")
    
    # Evaluate each model
    all_results = []
    
    for model_name, model_path in models_to_evaluate:
        try:
            results = evaluate_model(
                model_path=model_path,
                model_name=model_name,
                dataset=dataset,
                max_new_tokens=1
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    if not all_results:
        logger.error("No models were successfully evaluated!")
        return 1
    
    # Save results
    csv_path = save_detailed_results(all_results, args.output_dir)
    create_summary_report(all_results, args.output_dir)
    
    # Print summary
    logger.info("Evaluation Summary:")
    sorted_results = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        logger.info(f"  {i}. {result['model']}: {result['accuracy']:.4f} accuracy")
    
    best_model = sorted_results[0]
    logger.info(f"Best performing model: {best_model['model']} ({best_model['accuracy']:.4f})")
    
    logger.info("✓ Phase 4 completed successfully")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Next: Review results and run analysis")
    
    return 0

if __name__ == "__main__":
    exit(main()) 