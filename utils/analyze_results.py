#!/usr/bin/env python3
"""
Results Analysis and Visualization
Generates comprehensive analysis and plots from evaluation results.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    try:
        df = pd.read_csv(results_path)
        logger.info(f"Loaded {len(df)} results from {results_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return None

def create_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Create accuracy comparison plot."""
    plt.figure(figsize=(10, 6))
    
    # Sort by accuracy for better visualization
    df_sorted = df.sort_values('accuracy', ascending=True)
    
    # Create bar plot
    bars = plt.barh(df_sorted['model'], df_sorted['accuracy'])
    
    # Color bars
    colors = ['#ff7f0e' if model == 'baseline' else '#1f77b4' for model in df_sorted['model']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(df_sorted['model'], df_sorted['accuracy'])):
        plt.text(acc + 0.001, i, f'{acc:.3f}', va='center', fontweight='bold')
    
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison on CommonsenseQA')
    plt.xlim(0, max(df['accuracy']) * 1.1)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Accuracy comparison plot saved to {output_path}")

def create_performance_metrics(df: pd.DataFrame, output_dir: str):
    """Create comprehensive performance metrics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy plot
    axes[0, 0].bar(df['model'], df['accuracy'])
    axes[0, 0].set_title('Accuracy by Optimizer')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Speed plot
    axes[0, 1].bar(df['model'], df['items_per_second'])
    axes[0, 1].set_title('Inference Speed by Optimizer')
    axes[0, 1].set_ylabel('Items per Second')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Evaluation time
    axes[1, 0].bar(df['model'], df['evaluation_time_seconds'])
    axes[1, 0].set_title('Evaluation Time by Optimizer')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # GPU Memory usage
    if 'gpu_memory_used_mb' in df.columns:
        axes[1, 1].bar(df['model'], df['gpu_memory_used_mb'])
        axes[1, 1].set_title('GPU Memory Usage by Optimizer')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'GPU Memory\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('GPU Memory Usage')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'performance_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Performance metrics plot saved to {output_path}")

def create_radar_chart(df: pd.DataFrame, output_dir: str):
    """Create radar chart comparing models across multiple metrics."""
    # Normalize metrics for radar chart
    metrics = ['accuracy', 'items_per_second']
    
    # Add normalized columns
    df_norm = df.copy()
    for metric in metrics:
        df_norm[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[f'{metric}_norm'] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Speed'])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison\n(Normalized Metrics)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Save plot
    output_path = Path(output_dir) / 'radar_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Radar chart saved to {output_path}")

def create_improvement_analysis(df: pd.DataFrame, output_dir: str):
    """Create improvement analysis relative to baseline."""
    baseline_row = df[df['model'] == 'baseline']
    
    if baseline_row.empty:
        logger.warning("No baseline model found for improvement analysis")
        return
    
    baseline_acc = baseline_row['accuracy'].iloc[0]
    
    # Calculate improvements
    improvements = []
    for _, row in df.iterrows():
        if row['model'] != 'baseline':
            improvement = (row['accuracy'] - baseline_acc) * 100
            improvements.append({
                'model': row['model'],
                'improvement_pct': improvement,
                'accuracy': row['accuracy']
            })
    
    if not improvements:
        logger.warning("No fine-tuned models found for improvement analysis")
        return
    
    imp_df = pd.DataFrame(improvements)
    
    # Create improvement plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(imp_df['model'], imp_df['improvement_pct'])
    
    # Color positive improvements green, negative red
    for bar, imp in zip(bars, imp_df['improvement_pct']):
        bar.set_color('#2ca02c' if imp > 0 else '#d62728')
    
    # Add value labels
    for i, (model, imp) in enumerate(zip(imp_df['model'], imp_df['improvement_pct'])):
        plt.text(i, imp + 0.1 if imp > 0 else imp - 0.1, f'{imp:+.2f}%', 
                ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Optimizer')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title(f'Accuracy Improvement over Baseline\n(Baseline: {baseline_acc:.3f})')
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'improvement_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Improvement analysis plot saved to {output_path}")

def create_efficiency_analysis(df: pd.DataFrame, output_dir: str):
    """Create efficiency analysis (accuracy vs speed)."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    colors = ['#ff7f0e' if model == 'baseline' else '#1f77b4' for model in df['model']]
    plt.scatter(df['items_per_second'], df['accuracy'], s=100, c=colors, alpha=0.7, edgecolors='black')
    
    # Add labels for each point
    for _, row in df.iterrows():
        plt.annotate(row['model'], 
                    (row['items_per_second'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xlabel('Inference Speed (items/second)')
    plt.ylabel('Accuracy')
    plt.title('Efficiency Analysis: Accuracy vs Speed')
    plt.grid(True, alpha=0.3)
    
    # Add quadrant lines
    plt.axhline(y=df['accuracy'].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=df['items_per_second'].mean(), color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    plt.text(0.95, 0.95, 'High Accuracy\nHigh Speed', transform=plt.gca().transAxes, 
             ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'efficiency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Efficiency analysis plot saved to {output_path}")

def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create a formatted summary table."""
    # Select key metrics
    summary_cols = ['model', 'accuracy', 'items_per_second', 'evaluation_time_seconds']
    if 'gpu_memory_used_mb' in df.columns:
        summary_cols.append('gpu_memory_used_mb')
    
    summary_df = df[summary_cols].copy()
    
    # Round numerical values
    summary_df['accuracy'] = summary_df['accuracy'].round(4)
    summary_df['items_per_second'] = summary_df['items_per_second'].round(2)
    summary_df['evaluation_time_seconds'] = summary_df['evaluation_time_seconds'].round(1)
    if 'gpu_memory_used_mb' in summary_df.columns:
        summary_df['gpu_memory_used_mb'] = summary_df['gpu_memory_used_mb'].round(0)
    
    # Sort by accuracy (descending)
    summary_df = summary_df.sort_values('accuracy', ascending=False)
    
    # Save as CSV
    output_path = Path(output_dir) / 'summary_table.csv'
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary table saved to {output_path}")
    
    # Create formatted markdown table
    md_table = summary_df.to_markdown(index=False, floatfmt='.4f')
    
    md_path = Path(output_dir) / 'summary_table.md'
    with open(md_path, 'w') as f:
        f.write("# Results Summary Table\n\n")
        f.write(md_table)
    
    logger.info(f"Markdown table saved to {md_path}")

def create_gpu_timeline(experiments_dir: str, output_dir: str, optimizers: List[str]):
    """Create GPU utilization timeline from training metrics."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('GPU Memory Usage Over Time', 'GPU Utilization Over Time'),
        vertical_spacing=0.15
    )

    colors = px.colors.qualitative.Set2

    for i, optimizer in enumerate(optimizers):
        gpu_metrics_path = Path(experiments_dir) / optimizer / "gpu_metrics.json"

        if not gpu_metrics_path.exists():
            logger.warning(f"GPU metrics not found for {optimizer}: {gpu_metrics_path}")
            continue

        try:
            with open(gpu_metrics_path, 'r') as f:
                metrics = json.load(f)

            if not metrics:
                continue

            timestamps = [m['timestamp'] for m in metrics if 'timestamp' in m]
            gpu_memory = [m.get('gpu_memory_used_mb', 0) for m in metrics]
            gpu_util = [m.get('gpu_utilization_percent', 0) for m in metrics]

            # Add memory trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=gpu_memory,
                    mode='lines',
                    name=f'{optimizer} (Memory)',
                    line=dict(color=colors[i % len(colors)], width=2),
                    legendgroup=optimizer,
                ),
                row=1, col=1
            )

            # Add utilization trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=gpu_util,
                    mode='lines',
                    name=f'{optimizer} (Util)',
                    line=dict(color=colors[i % len(colors)], width=2),
                    legendgroup=optimizer,
                    showlegend=False
                ),
                row=2, col=1
            )

        except Exception as e:
            logger.error(f"Failed to load GPU metrics for {optimizer}: {e}")

    # Update layout
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="GPU Metrics Timeline During Training",
        showlegend=True,
        hovermode='x unified'
    )

    # Save interactive HTML
    output_path = Path(output_dir) / 'gpu_timeline.html'
    fig.write_html(str(output_path))
    logger.info(f"GPU timeline saved to {output_path}")

    # Also save as PNG
    try:
        png_path = Path(output_dir) / 'gpu_timeline.png'
        fig.write_image(str(png_path), width=1200, height=800)
        logger.info(f"GPU timeline PNG saved to {png_path}")
    except Exception as e:
        logger.warning(f"Could not save PNG (requires kaleido): {e}")


def create_interactive_radar(df: pd.DataFrame, output_dir: str):
    """Create interactive radar chart with plotly."""
    # Normalize metrics for radar chart
    metrics = ['accuracy', 'items_per_second']

    # Add GPU memory if available (inverse normalization - less is better)
    if 'gpu_memory_used_mb' in df.columns:
        metrics.append('gpu_memory_used_mb')

    # Normalize all metrics
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if metric == 'gpu_memory_used_mb':
            # Invert for memory (less is better)
            df_norm[f'{metric}_norm'] = 1 - ((df[metric] - min_val) / (max_val - min_val + 1e-10))
        else:
            df_norm[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val + 1e-10)

    # Create plotly radar chart
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[f'{metric}_norm'] for metric in metrics]
        values.append(values[0])  # Close the radar

        categories = ['Accuracy', 'Speed']
        if 'gpu_memory_used_mb' in metrics:
            categories.append('Memory Efficiency')
        categories.append(categories[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['model'],
            line_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Interactive Multi-Metric Comparison (Normalized)",
        height=600
    )

    # Save interactive HTML
    output_path = Path(output_dir) / 'interactive_radar.html'
    fig.write_html(str(output_path))
    logger.info(f"Interactive radar chart saved to {output_path}")


def create_training_comparison(experiments_dir: str, output_dir: str, optimizers: List[str]):
    """Create training loss comparison from training metrics."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, optimizer in enumerate(optimizers):
        metrics_path = Path(experiments_dir) / optimizer / "gpu_metrics.json"

        if not metrics_path.exists():
            logger.warning(f"Training metrics not found for {optimizer}")
            continue

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            if not metrics:
                continue

            # Extract loss values
            steps = []
            losses = []
            for m in metrics:
                if 'step' in m and 'loss' in m:
                    steps.append(m['step'])
                    losses.append(m['loss'])

            if steps and losses:
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=losses,
                    mode='lines+markers',
                    name=optimizer,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))

        except Exception as e:
            logger.error(f"Failed to load training metrics for {optimizer}: {e}")

    fig.update_layout(
        title="Training Loss Comparison",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        hovermode='x unified',
        height=500
    )

    # Save interactive HTML
    output_path = Path(output_dir) / 'training_loss_comparison.html'
    fig.write_html(str(output_path))
    logger.info(f"Training loss comparison saved to {output_path}")


def generate_insights(df: pd.DataFrame, output_dir: str):
    """Generate insights and recommendations."""
    insights = []
    
    # Best performing model
    best_model = df.loc[df['accuracy'].idxmax()]
    insights.append(f"🏆 **Best Performing Model**: {best_model['model']} with {best_model['accuracy']:.4f} accuracy")
    
    # Fastest model
    fastest_model = df.loc[df['items_per_second'].idxmax()]
    insights.append(f"⚡ **Fastest Model**: {fastest_model['model']} at {fastest_model['items_per_second']:.2f} items/sec")
    
    # Baseline comparison
    baseline = df[df['model'] == 'baseline']
    if not baseline.empty:
        baseline_acc = baseline['accuracy'].iloc[0]
        improved_models = df[df['accuracy'] > baseline_acc]
        if len(improved_models) > 1:  # Excluding baseline itself
            avg_improvement = ((improved_models['accuracy'].mean() - baseline_acc) * 100)
            insights.append(f"📈 **Average Improvement**: {avg_improvement:.2f}% over baseline")
    
    # Speed vs accuracy trade-off
    speed_acc_corr = df['items_per_second'].corr(df['accuracy'])
    if abs(speed_acc_corr) > 0.5:
        trend = "positive" if speed_acc_corr > 0 else "negative"
        insights.append(f"🔄 **Speed-Accuracy Trade-off**: {trend} correlation ({speed_acc_corr:.2f})")
    
    # Variability analysis
    acc_std = df['accuracy'].std()
    if acc_std < 0.01:
        insights.append("📊 **Low Variability**: All models perform similarly (std < 0.01)")
    elif acc_std > 0.05:
        insights.append("📊 **High Variability**: Significant performance differences between optimizers")
    
    # Save insights
    insights_path = Path(output_dir) / 'insights.md'
    with open(insights_path, 'w') as f:
        f.write("# Key Insights\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Generate recommendations based on insights
        if best_model['model'] != 'baseline':
            f.write(f"1. **For highest accuracy**: Use {best_model['model']} optimizer\n")
        
        if fastest_model['model'] != best_model['model']:
            f.write(f"2. **For fastest inference**: Use {fastest_model['model']} optimizer\n")
        
        f.write("3. **For production**: Consider the accuracy-speed trade-off based on your requirements\n")
        f.write("4. **For further experiments**: Try ensemble methods or different learning rates\n")
    
    logger.info(f"Insights saved to {insights_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize experiment results")
    parser.add_argument("--results_file", default="./results/results.csv",
                       help="Path to results CSV file")
    parser.add_argument("--output_dir", default="./results/analysis",
                       help="Output directory for plots and analysis")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                       help="Output format for plots")
    parser.add_argument("--experiments_dir", default="./experiments",
                       help="Directory containing experiment results for GPU metrics")
    parser.add_argument("--optimizers", nargs="+",
                       default=["adamw", "sgd", "adabound", "hybrid"],
                       help="Optimizers to analyze")

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = load_results(args.results_file)
    if df is None:
        return 1
    
    logger.info(f"Analyzing {len(df)} models")
    logger.info(f"Models: {', '.join(df['model'].tolist())}")
    
    # Generate all analyses
    try:
        logger.info("Generating static plots...")
        create_accuracy_comparison(df, args.output_dir)
        create_performance_metrics(df, args.output_dir)
        create_radar_chart(df, args.output_dir)
        create_improvement_analysis(df, args.output_dir)
        create_efficiency_analysis(df, args.output_dir)
        create_summary_table(df, args.output_dir)
        generate_insights(df, args.output_dir)

        logger.info("Generating interactive visualizations...")
        create_interactive_radar(df, args.output_dir)
        create_gpu_timeline(args.experiments_dir, args.output_dir, args.optimizers)
        create_training_comparison(args.experiments_dir, args.output_dir, args.optimizers)

        logger.info("✓ Analysis completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 