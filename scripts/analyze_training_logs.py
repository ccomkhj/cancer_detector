#!/usr/bin/env python3
"""
Training Log Analysis & Visualization Script

Parses SLURM training logs to analyze hyperparameter exploration results
and identify the best performing models.

Usage:
    python scripts/analyze_training_logs.py
    python scripts/analyze_training_logs.py --logs-dir logs/
    python scripts/analyze_training_logs.py --output results/analysis.html
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Install plotly for interactive HTML plots: pip install plotly")


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    lr: float
    train_loss: float
    train_dice: float
    val_loss: float
    val_dice: float
    val_precision: float
    val_recall: float
    val_dice_prostate: Optional[float] = None
    val_dice_target: Optional[float] = None
    val_precision_prostate: Optional[float] = None
    val_precision_target: Optional[float] = None
    val_recall_prostate: Optional[float] = None
    val_recall_target: Optional[float] = None


# Model parameter counts for inference (approximate)
MODEL_PARAMS = {
    31045890: "simple_unet",
    24439070: "smp_unet_resnet34",
    32525154: "smp_unet_resnet50",
    16803906: "smp_unet_efficientnet-b0",
    25556322: "smp_unet_efficientnet-b4",
    26658478: "smp_unetplusplus_resnet34",
    34744418: "smp_unetplusplus_resnet50",
    23566146: "smp_deeplabv3plus_resnet34",
    42518850: "smp_deeplabv3plus_resnet50",
    21542498: "smp_fpn_resnet34",
    26874082: "smp_fpn_resnet50",
    21982338: "smp_manet_resnet34",
    11542082: "smp_linknet_resnet34",
}


@dataclass
class TrainingRun:
    """Complete training run information."""
    job_id: str
    # Config
    model: str = "unknown"
    epochs: int = 0
    lr: float = 0.0
    batch_size: int = 8
    loss: str = "unknown"
    scheduler: str = "unknown"
    stack_depth: int = 5
    ft_gamma: Optional[float] = None
    ft_alpha: Optional[List[float]] = None
    ft_beta: Optional[List[float]] = None
    ft_class_weights: Optional[List[float]] = None
    # Results
    best_val_dice: float = 0.0
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_train_dice: float = 0.0
    model_params: int = 0
    # History
    history: List[EpochMetrics] = field(default_factory=list)
    # Status
    completed: bool = False
    error: Optional[str] = None


def parse_log_file(filepath: Path) -> Optional[TrainingRun]:
    """Parse a single SLURM log file and extract training information."""
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Extract job ID from filename
    match = re.search(r'slurm-(\d+)\.out', filepath.name)
    if not match:
        return None

    job_id = match.group(1)
    run = TrainingRun(job_id=job_id)

    # Check if job completed
    run.completed = "Training Complete!" in content or "Job Complete" in content

    # Extract configuration
    # Model name appears as "  model: simple_unet" in the config section (with leading spaces)
    # Must not match "Saved best model:" lines
    config_patterns = {
        'model': r'^\s+model:\s*(\S+)',  # Leading whitespace, at start of line
        'epochs': r'Epochs:\s*(\d+)',
        'lr': r'Learning rate:\s*([\d.e-]+)',
        'batch_size': r'Batch size:\s*(\d+)',
        'loss': r'Loss:\s*(\S+)',
        'scheduler': r'Scheduler:\s*(\S+)',
        'stack_depth': r'stack_depth:\s*(\d+)',
        'ft_gamma': r'ft_gamma:\s*([\d.]+)',
        'model_params': r'Model parameters:\s*([\d,]+)',
    }

    for key, pattern in config_patterns.items():
        # Use MULTILINE flag for patterns that need ^ anchoring
        flags = re.MULTILINE if pattern.startswith('^') else 0
        match = re.search(pattern, content, flags)
        if match:
            value = match.group(1)
            if key in ['epochs', 'batch_size', 'stack_depth']:
                setattr(run, key, int(value))
            elif key in ['lr', 'ft_gamma']:
                setattr(run, key, float(value))
            elif key == 'model_params':
                setattr(run, key, int(value.replace(',', '')))
            else:
                setattr(run, key, value)

    # Extract ft_alpha, ft_beta, ft_class_weights (list values)
    for key in ['ft_alpha', 'ft_beta', 'ft_class_weights']:
        match = re.search(rf'{key}:\s*\[([\d., ]+)\]', content)
        if match:
            values = [float(x.strip()) for x in match.group(1).split(',')]
            setattr(run, key, values)

    # If model is still unknown, try to infer from parameter count
    if run.model == "unknown" and run.model_params > 0:
        # Find closest match in MODEL_PARAMS
        for param_count, model_name in MODEL_PARAMS.items():
            if abs(run.model_params - param_count) < 1000:  # Allow small tolerance
                run.model = model_name
                break

    # Extract best validation dice
    match = re.search(r'Best validation Dice:\s*([\d.]+)', content)
    if match:
        run.best_val_dice = float(match.group(1))

    # Extract best epoch from the last "Saved best model" line
    # Pattern: model_best_765600_36.pt (Dice: 0.6058, ...)
    best_model_matches = re.findall(r'model_best_\d+_(\d+)\.pt \(Dice: ([\d.]+)', content)
    if best_model_matches:
        # Get the last one (which has the best dice)
        last_match = best_model_matches[-1]
        run.best_epoch = int(last_match[0])

    # Parse epoch-by-epoch metrics
    # Pattern for epoch header
    epoch_pattern = r'Epoch (\d+)/\d+ - LR: ([\d.e-]+)'
    train_pattern = r'Train - Loss: ([\d.]+), Dice: ([\d.]+)'
    val_pattern = r'Val   - Loss: ([\d.]+), Dice: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    dice_class_pattern = r'Dice per class: Prostate=([\d.nan]+), Target=([\d.nan]+)'
    precision_class_pattern = r'Precision per class: Prostate=([\d.]+), Target=([\d.]+)'
    recall_class_pattern = r'Recall per class: Prostate=([\d.]+), Target=([\d.]+)'

    # Split content by epochs
    epoch_sections = re.split(r'(Epoch \d+/\d+ - LR: [\d.e-]+)', content)

    current_epoch = None
    current_lr = None

    for i, section in enumerate(epoch_sections):
        # Check if this is an epoch header
        epoch_match = re.match(epoch_pattern, section)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_lr = float(epoch_match.group(2))
            continue

        if current_epoch is None:
            continue

        # Extract metrics from this section
        train_match = re.search(train_pattern, section)
        val_match = re.search(val_pattern, section)

        if train_match and val_match:
            metrics = EpochMetrics(
                epoch=current_epoch,
                lr=current_lr,
                train_loss=float(train_match.group(1)),
                train_dice=float(train_match.group(2)),
                val_loss=float(val_match.group(1)),
                val_dice=float(val_match.group(2)),
                val_precision=float(val_match.group(3)),
                val_recall=float(val_match.group(4)),
            )

            # Extract per-class metrics
            dice_match = re.search(dice_class_pattern, section)
            if dice_match:
                prostate = dice_match.group(1)
                target = dice_match.group(2)
                metrics.val_dice_prostate = None if prostate == 'nan' else float(prostate)
                metrics.val_dice_target = None if target == 'nan' else float(target)

            precision_match = re.search(precision_class_pattern, section)
            if precision_match:
                metrics.val_precision_prostate = float(precision_match.group(1))
                metrics.val_precision_target = float(precision_match.group(2))

            recall_match = re.search(recall_class_pattern, section)
            if recall_match:
                metrics.val_recall_prostate = float(recall_match.group(1))
                metrics.val_recall_target = float(recall_match.group(2))

            run.history.append(metrics)

            # Track best epoch
            if metrics.val_dice > run.best_val_dice:
                run.best_val_dice = metrics.val_dice
                run.best_epoch = current_epoch

    # Get final training metrics
    if run.history:
        run.final_train_loss = run.history[-1].train_loss
        run.final_train_dice = run.history[-1].train_dice

    return run


def create_summary_dataframe(runs: List[TrainingRun]) -> pd.DataFrame:
    """Create a summary DataFrame from all training runs."""
    data = []
    for run in runs:
        row = {
            'job_id': run.job_id,
            'model': run.model,
            'epochs': run.epochs,
            'lr': run.lr,
            'batch_size': run.batch_size,
            'loss': run.loss,
            'scheduler': run.scheduler,
            'stack_depth': run.stack_depth,
            'ft_gamma': run.ft_gamma,
            'model_params': run.model_params,
            'best_val_dice': run.best_val_dice,
            'best_epoch': run.best_epoch,
            'final_train_loss': run.final_train_loss,
            'final_train_dice': run.final_train_dice,
            'completed': run.completed,
            'num_epochs_run': len(run.history),
        }

        # Add class weights info
        if run.ft_class_weights:
            row['ft_class_weight_target'] = run.ft_class_weights[1] if len(run.ft_class_weights) > 1 else None

        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('best_val_dice', ascending=False)
    return df


def plot_learning_curves_matplotlib(runs: List[TrainingRun], top_n: int = 10, output_dir: Path = None):
    """Plot learning curves for top N runs using matplotlib."""
    # Sort by best val dice
    sorted_runs = sorted(runs, key=lambda x: x.best_val_dice, reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for idx, run in enumerate(sorted_runs):
        if not run.history:
            continue

        epochs = [m.epoch for m in run.history]
        label = f"{run.job_id}: {run.model} (dice={run.best_val_dice:.4f})"
        color = colors[idx]

        # Training Loss
        axes[0, 0].plot(epochs, [m.train_loss for m in run.history],
                       color=color, alpha=0.7, label=label)

        # Validation Loss
        axes[0, 1].plot(epochs, [m.val_loss for m in run.history],
                       color=color, alpha=0.7, label=label)

        # Training Dice
        axes[1, 0].plot(epochs, [m.train_dice for m in run.history],
                       color=color, alpha=0.7, label=label)

        # Validation Dice
        axes[1, 1].plot(epochs, [m.val_dice for m in run.history],
                       color=color, alpha=0.7, label=label)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation Dice')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.suptitle(f'Learning Curves - Top {top_n} Models by Validation Dice', fontsize=14)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'learning_curves.png'}")

    plt.show()


def plot_hyperparameter_analysis_matplotlib(df: pd.DataFrame, output_dir: Path = None):
    """Analyze effect of different hyperparameters using matplotlib."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Model comparison
    if 'model' in df.columns:
        model_perf = df.groupby('model')['best_val_dice'].agg(['mean', 'max', 'count'])
        model_perf = model_perf.sort_values('max', ascending=True)

        ax = axes[0, 0]
        y_pos = range(len(model_perf))
        ax.barh(y_pos, model_perf['max'], alpha=0.8, label='Best')
        ax.barh(y_pos, model_perf['mean'], alpha=0.5, label='Mean')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_perf.index, fontsize=8)
        ax.set_xlabel('Validation Dice')
        ax.set_title('Performance by Model Architecture')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

    # 2. Learning rate effect
    ax = axes[0, 1]
    df_completed = df[df['completed'] == True]
    if len(df_completed) > 0:
        ax.scatter(df_completed['lr'], df_completed['best_val_dice'],
                  alpha=0.6, c=df_completed['best_val_dice'], cmap='viridis')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Best Validation Dice')
        ax.set_title('Effect of Learning Rate')
        ax.grid(True, alpha=0.3)

    # 3. Loss function comparison
    if 'loss' in df.columns:
        ax = axes[0, 2]
        loss_perf = df.groupby('loss')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
        loss_perf = loss_perf.sort_values('max', ascending=False)

        x_pos = range(len(loss_perf))
        ax.bar(x_pos, loss_perf['max'], alpha=0.8, label='Best')
        ax.bar(x_pos, loss_perf['mean'], alpha=0.5, label='Mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_perf.index, rotation=45, ha='right')
        ax.set_ylabel('Validation Dice')
        ax.set_title('Performance by Loss Function')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 4. Scheduler comparison
    if 'scheduler' in df.columns:
        ax = axes[1, 0]
        sched_perf = df.groupby('scheduler')['best_val_dice'].agg(['mean', 'max', 'count'])
        sched_perf = sched_perf.sort_values('max', ascending=False)

        x_pos = range(len(sched_perf))
        ax.bar(x_pos, sched_perf['max'], alpha=0.8, label='Best')
        ax.bar(x_pos, sched_perf['mean'], alpha=0.5, label='Mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sched_perf.index, rotation=45, ha='right')
        ax.set_ylabel('Validation Dice')
        ax.set_title('Performance by Scheduler')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 5. Batch size effect
    if 'batch_size' in df.columns:
        ax = axes[1, 1]
        bs_perf = df.groupby('batch_size')['best_val_dice'].agg(['mean', 'max', 'count'])
        bs_perf = bs_perf.sort_index()

        x_pos = range(len(bs_perf))
        ax.bar(x_pos, bs_perf['max'], alpha=0.8, label='Best')
        ax.bar(x_pos, bs_perf['mean'], alpha=0.5, label='Mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bs_perf.index)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Validation Dice')
        ax.set_title('Performance by Batch Size')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 6. Training epochs vs best dice
    ax = axes[1, 2]
    ax.scatter(df['num_epochs_run'], df['best_val_dice'],
              alpha=0.6, c=df['best_val_dice'], cmap='viridis')
    ax.set_xlabel('Epochs Trained')
    ax.set_ylabel('Best Validation Dice')
    ax.set_title('Training Duration vs Performance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Analysis', fontsize=14)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'hyperparameter_analysis.png'}")

    plt.show()


def create_interactive_plots(runs: List[TrainingRun], df: pd.DataFrame, output_path: Path):
    """Create interactive HTML plots using plotly."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive plots.")
        return

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Top Models - Validation Dice Over Epochs',
            'Model Architecture Comparison',
            'Learning Rate vs Best Dice',
            'Loss Function Comparison',
            'Scheduler Comparison',
            'Best Val Dice Distribution'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 1. Learning curves for top models
    sorted_runs = sorted(runs, key=lambda x: x.best_val_dice, reverse=True)[:10]
    for run in sorted_runs:
        if run.history:
            epochs = [m.epoch for m in run.history]
            val_dice = [m.val_dice for m in run.history]
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=val_dice, mode='lines',
                    name=f"{run.job_id}: {run.model[:20]} (dice={run.best_val_dice:.4f})",
                    hovertemplate=f"Job: {run.job_id}<br>Model: {run.model}<br>Epoch: %{{x}}<br>Val Dice: %{{y:.4f}}"
                ),
                row=1, col=1
            )

    # 2. Model architecture comparison (box plot)
    model_groups = df.groupby('model')['best_val_dice'].apply(list).to_dict()
    for model, dices in model_groups.items():
        fig.add_trace(
            go.Box(y=dices, name=model[:15], boxpoints='all'),
            row=1, col=2
        )

    # 3. Learning rate vs best dice
    fig.add_trace(
        go.Scatter(
            x=df['lr'], y=df['best_val_dice'], mode='markers',
            marker=dict(size=10, color=df['best_val_dice'], colorscale='Viridis'),
            text=df['job_id'],
            hovertemplate="Job: %{text}<br>LR: %{x}<br>Best Dice: %{y:.4f}"
        ),
        row=2, col=1
    )

    # 4. Loss function comparison
    loss_perf = df.groupby('loss')['best_val_dice'].agg(['mean', 'max']).reset_index()
    fig.add_trace(
        go.Bar(x=loss_perf['loss'], y=loss_perf['max'], name='Best', marker_color='steelblue'),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=loss_perf['loss'], y=loss_perf['mean'], name='Mean', marker_color='lightsteelblue'),
        row=2, col=2
    )

    # 5. Scheduler comparison
    sched_perf = df.groupby('scheduler')['best_val_dice'].agg(['mean', 'max']).reset_index()
    fig.add_trace(
        go.Bar(x=sched_perf['scheduler'], y=sched_perf['max'], name='Best', marker_color='coral', showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=sched_perf['scheduler'], y=sched_perf['mean'], name='Mean', marker_color='lightsalmon', showlegend=False),
        row=3, col=1
    )

    # 6. Distribution of best val dice
    fig.add_trace(
        go.Histogram(x=df['best_val_dice'], nbinsx=20, marker_color='teal'),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        title_text="MRI Segmentation - Hyperparameter Exploration Analysis",
        showlegend=True
    )

    # Update axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Validation Dice", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Learning Rate", row=2, col=1)
    fig.update_yaxes(title_text="Best Validation Dice", row=2, col=1)

    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved interactive plot: {output_path}")


def print_summary_report(df: pd.DataFrame, runs: List[TrainingRun]):
    """Print a summary report of the analysis."""
    print("\n" + "="*80)
    print("TRAINING LOG ANALYSIS REPORT")
    print("="*80)

    print(f"\nTotal runs analyzed: {len(df)}")
    print(f"Completed runs: {df['completed'].sum()}")
    print(f"Failed/incomplete runs: {(~df['completed']).sum()}")

    print("\n" + "-"*80)
    print("TOP 10 BEST MODELS (by Validation Dice)")
    print("-"*80)

    top_10 = df.head(10)[['job_id', 'model', 'lr', 'loss', 'scheduler', 'best_val_dice', 'best_epoch', 'epochs']]
    print(top_10.to_string(index=False))

    print("\n" + "-"*80)
    print("BEST CONFIGURATION")
    print("-"*80)

    best = df.iloc[0]
    print(f"  Job ID: {best['job_id']}")
    print(f"  Model: {best['model']}")
    print(f"  Learning Rate: {best['lr']}")
    print(f"  Loss Function: {best['loss']}")
    print(f"  Scheduler: {best['scheduler']}")
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Best Validation Dice: {best['best_val_dice']:.4f}")
    print(f"  Best Epoch: {best['best_epoch']}/{best['epochs']}")

    # Find the run to get more details
    best_run = next((r for r in runs if r.job_id == best['job_id']), None)
    if best_run:
        print(f"  Model Parameters: {best_run.model_params:,}")
        if best_run.ft_gamma:
            print(f"  FT Gamma: {best_run.ft_gamma}")
        if best_run.ft_class_weights:
            print(f"  FT Class Weights: {best_run.ft_class_weights}")

    print("\n" + "-"*80)
    print("PERFORMANCE BY MODEL ARCHITECTURE")
    print("-"*80)

    model_perf = df.groupby('model')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
    model_perf = model_perf.sort_values('max', ascending=False)
    model_perf.columns = ['Mean Dice', 'Max Dice', 'Std', 'Count']
    print(model_perf.to_string())

    print("\n" + "-"*80)
    print("PERFORMANCE BY LOSS FUNCTION")
    print("-"*80)

    loss_perf = df.groupby('loss')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
    loss_perf = loss_perf.sort_values('max', ascending=False)
    loss_perf.columns = ['Mean Dice', 'Max Dice', 'Std', 'Count']
    print(loss_perf.to_string())

    print("\n" + "-"*80)
    print("PERFORMANCE BY SCHEDULER")
    print("-"*80)

    sched_perf = df.groupby('scheduler')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
    sched_perf = sched_perf.sort_values('max', ascending=False)
    sched_perf.columns = ['Mean Dice', 'Max Dice', 'Std', 'Count']
    print(sched_perf.to_string())

    print("\n" + "-"*80)
    print("PERFORMANCE BY LEARNING RATE")
    print("-"*80)

    lr_perf = df.groupby('lr')['best_val_dice'].agg(['mean', 'max', 'count'])
    lr_perf = lr_perf.sort_values('max', ascending=False)
    lr_perf.columns = ['Mean Dice', 'Max Dice', 'Count']
    print(lr_perf.to_string())

    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)

    # Find best combinations
    best_model = model_perf['Max Dice'].idxmax()
    best_loss = loss_perf['Max Dice'].idxmax()
    best_sched = sched_perf['Max Dice'].idxmax()
    best_lr = lr_perf['Max Dice'].idxmax()

    print(f"  Best Model Architecture: {best_model}")
    print(f"  Best Loss Function: {best_loss}")
    print(f"  Best Scheduler: {best_sched}")
    print(f"  Best Learning Rate: {best_lr}")

    print("\n  Suggested next experiment:")
    print(f"    --model {best_model} --loss {best_loss} --scheduler {best_sched} --lr {best_lr}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze MRI training logs")
    parser.add_argument("--logs-dir", type=str, default="logs",
                        help="Directory containing SLURM log files")
    parser.add_argument("--output", type=str, default="results/training_analysis.html",
                        help="Output path for interactive HTML report")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Save summary DataFrame to CSV")
    args = parser.parse_args()

    # Find all log files
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return 1

    log_files = sorted(logs_dir.glob("slurm-*.out"))
    if not log_files:
        print(f"Error: No SLURM log files found in {logs_dir}")
        return 1

    print(f"Found {len(log_files)} log files in {logs_dir}")

    # Parse all log files
    runs = []
    for log_file in log_files:
        run = parse_log_file(log_file)
        if run:
            runs.append(run)

    print(f"Successfully parsed {len(runs)} training runs")

    if not runs:
        print("Error: No valid training runs found")
        return 1

    # Create summary DataFrame
    df = create_summary_dataframe(runs)

    # Print summary report
    print_summary_report(df, runs)

    # Save CSV if requested
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary CSV: {csv_path}")

    # Generate plots
    if not args.no_plots:
        output_dir = Path(args.output).parent

        # Matplotlib plots
        print("\nGenerating matplotlib plots...")
        plot_learning_curves_matplotlib(runs, top_n=10, output_dir=output_dir)
        plot_hyperparameter_analysis_matplotlib(df, output_dir=output_dir)

        # Interactive plotly plots
        if PLOTLY_AVAILABLE:
            print("\nGenerating interactive HTML report...")
            create_interactive_plots(runs, df, Path(args.output))

    return 0


if __name__ == "__main__":
    exit(main())
