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
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


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
# Note: Parameter counts vary slightly with stack_depth changes
# These are base counts for stack_depth=5 (default)
MODEL_PARAMS = {
    31045890: "simple_unet",      # stack_depth=5
    31044738: "simple_unet",      # stack_depth=5 (slight variation)
    31047042: "simple_unet",      # stack_depth=7
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
    best_val_dice: float = 0.0
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_train_dice: float = 0.0
    model_params: int = 0
    history: List[EpochMetrics] = field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None


def parse_log_file(filepath: Path) -> Optional[TrainingRun]:
    """Parse a single SLURM log file and extract training information."""
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    match = re.search(r'slurm-(\d+)\.out', filepath.name)
    if not match:
        return None

    job_id = match.group(1)
    run = TrainingRun(job_id=job_id)
    run.completed = "Training Complete!" in content or "Job Complete" in content

    config_patterns = {
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

    # Parse model name with multiple fallback patterns
    model_patterns = [
        # Pattern 1: "Key Hyperparameters" section (new format)
        r'---\s*Key Hyperparameters.*?model:\s*(\S+)',
        # Pattern 2: Config section with indentation
        r'^\s+model:\s*(\S+)',
        # Pattern 3: Creating SMP model message
        r'Creating SMP (\w+) with (\S+) encoder',
        # Pattern 4: Creating SimpleUNet message
        r'Creating (SimpleUNet)',
    ]

    for i, pattern in enumerate(model_patterns):
        flags = re.MULTILINE | re.DOTALL if '.*?' in pattern else re.MULTILINE
        match = re.search(pattern, content, flags)
        if match:
            if i == 2:  # SMP pattern with architecture and encoder
                arch = match.group(1).lower()
                encoder = match.group(2)
                run.model = f"smp_{arch}_{encoder}"
            elif i == 3:  # SimpleUNet
                run.model = "simple_unet"
            else:
                run.model = match.group(1)
            break

    for key in ['ft_alpha', 'ft_beta', 'ft_class_weights']:
        match = re.search(rf'{key}:\s*\[([\d., ]+)\]', content)
        if match:
            values = [float(x.strip()) for x in match.group(1).split(',')]
            setattr(run, key, values)

    if run.model == "unknown" and run.model_params > 0:
        for param_count, model_name in MODEL_PARAMS.items():
            if abs(run.model_params - param_count) < 1000:
                run.model = model_name
                break

    match = re.search(r'Best validation Dice:\s*([\d.]+)', content)
    if match:
        run.best_val_dice = float(match.group(1))

    best_model_matches = re.findall(r'model_best_\d+_(\d+)\.pt \(Dice: ([\d.]+)', content)
    if best_model_matches:
        last_match = best_model_matches[-1]
        run.best_epoch = int(last_match[0])

    epoch_pattern = r'Epoch (\d+)/\d+ - LR: ([\d.e-]+)'
    train_pattern = r'Train - Loss: ([\d.]+), Dice: ([\d.]+)'
    val_pattern = r'Val   - Loss: ([\d.]+), Dice: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    dice_class_pattern = r'Dice per class: Prostate=([\d.nan]+), Target=([\d.nan]+)'
    precision_class_pattern = r'Precision per class: Prostate=([\d.]+), Target=([\d.]+)'
    recall_class_pattern = r'Recall per class: Prostate=([\d.]+), Target=([\d.]+)'

    epoch_sections = re.split(r'(Epoch \d+/\d+ - LR: [\d.e-]+)', content)
    current_epoch = None
    current_lr = None

    for section in epoch_sections:
        epoch_match = re.match(epoch_pattern, section)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_lr = float(epoch_match.group(2))
            continue

        if current_epoch is None:
            continue

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
        if run.ft_class_weights:
            row['ft_class_weight_target'] = run.ft_class_weights[1] if len(run.ft_class_weights) > 1 else None
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('best_val_dice', ascending=False)
    return df


def create_interactive_html_report(runs: List[TrainingRun], df: pd.DataFrame, output_path: Path):
    """Create a comprehensive interactive HTML report with experiment selection."""
    sorted_runs = sorted(runs, key=lambda x: x.best_val_dice, reverse=True)

    # Prepare epoch data
    all_epochs_data = []
    for run in sorted_runs:
        for m in run.history:
            all_epochs_data.append({
                'job_id': run.job_id,
                'label': f"{run.job_id}: {run.model} (dice={run.best_val_dice:.4f})",
                'epoch': m.epoch,
                'train_loss': m.train_loss,
                'train_dice': m.train_dice,
                'val_loss': m.val_loss,
                'val_dice': m.val_dice,
                'val_precision': m.val_precision,
                'val_recall': m.val_recall,
                'val_precision_prostate': m.val_precision_prostate,
                'val_precision_target': m.val_precision_target,
                'val_recall_prostate': m.val_recall_prostate,
                'val_recall_target': m.val_recall_target,
                'model': run.model,
                'scheduler': run.scheduler,
                'loss_fn': run.loss,
                'best_val_dice': run.best_val_dice,
            })

    epochs_json = json.dumps(all_epochs_data)
    summary_json = df.to_json(orient='records')

    # Build checkbox HTML
    checkbox_html = ""
    for i, run in enumerate(sorted_runs):
        checked = "checked" if i < 10 else ""
        label = f"{run.job_id}: {run.model} | {run.scheduler} | lr={run.lr:.0e} | dice={run.best_val_dice:.4f}"
        checkbox_html += f'''
            <div class="checkbox-item">
                <input type="checkbox" id="exp_{run.job_id}" value="{run.job_id}" {checked} onchange="updatePlots()">
                <label for="exp_{run.job_id}">{label}</label>
            </div>'''

    # Build table HTML
    table_rows = ""
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        row_class = "best-row" if i == 0 else ""
        table_rows += f'''
            <tr class="{row_class}">
                <td>{i+1}</td>
                <td>{row['job_id']}</td>
                <td>{row['model']}</td>
                <td>{row['scheduler']}</td>
                <td>{row['loss']}</td>
                <td>{row['lr']:.0e}</td>
                <td><strong>{row['best_val_dice']:.4f}</strong></td>
                <td>{row['best_epoch']}</td>
                <td>{row['epochs']}</td>
            </tr>'''

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>MRI Segmentation - Training Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background: #e8f5e9; border-radius: 4px; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
        .plot-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .plot-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .plot-cell {{ flex: 1; min-width: 400px; }}
        .selector-container {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .checkbox-container {{ display: flex; flex-wrap: wrap; gap: 10px; max-height: 200px; overflow-y: auto; padding: 10px; background: #fafafa; border-radius: 4px; }}
        .checkbox-item {{ display: flex; align-items: center; padding: 5px 10px; background: white; border-radius: 4px; border: 1px solid #e0e0e0; }}
        .checkbox-item:hover {{ background: #e3f2fd; }}
        .checkbox-item input {{ margin-right: 8px; }}
        .checkbox-item label {{ font-size: 12px; cursor: pointer; }}
        button {{ padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }}
        button:hover {{ background: #45a049; }}
        button.secondary {{ background: #757575; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .best-row {{ background: #e8f5e9 !important; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MRI Segmentation - Hyperparameter Exploration Analysis</h1>

        <div class="summary-box">
            <h2 style="margin-top: 0;">Summary</h2>
            <div class="metric"><div class="metric-label">Total Runs</div><div class="metric-value">{len(df)}</div></div>
            <div class="metric"><div class="metric-label">Completed</div><div class="metric-value">{df['completed'].sum()}</div></div>
            <div class="metric"><div class="metric-label">Best Dice</div><div class="metric-value">{df['best_val_dice'].max():.4f}</div></div>
            <div class="metric"><div class="metric-label">Best Model</div><div class="metric-value">{df.iloc[0]['model']}</div></div>
            <div class="metric"><div class="metric-label">Best Scheduler</div><div class="metric-value">{df.iloc[0]['scheduler']}</div></div>
            <div class="metric"><div class="metric-label">Best LR</div><div class="metric-value">{df.iloc[0]['lr']:.0e}</div></div>
        </div>

        <div class="selector-container">
            <h2 style="margin-top: 0;">Select Experiments to Compare</h2>
            <div style="margin-bottom: 10px;">
                <button onclick="selectTop(5)">Top 5</button>
                <button onclick="selectTop(10)">Top 10</button>
                <button onclick="selectAll()">Select All</button>
                <button onclick="selectNone()" class="secondary">Clear All</button>
            </div>
            <div class="checkbox-container" id="experiment-checkboxes">{checkbox_html}
            </div>
        </div>

        <div class="plot-container">
            <h2>Learning Curves</h2>
            <div class="plot-row">
                <div class="plot-cell"><div id="plot-train-loss"></div></div>
                <div class="plot-cell"><div id="plot-val-loss"></div></div>
            </div>
            <div class="plot-row">
                <div class="plot-cell"><div id="plot-train-dice"></div></div>
                <div class="plot-cell"><div id="plot-val-dice"></div></div>
            </div>
        </div>

        <div class="plot-container">
            <h2>Validation Metrics - Per Class</h2>
            <div class="plot-row">
                <div class="plot-cell"><div id="plot-prostate-precision"></div></div>
                <div class="plot-cell"><div id="plot-target-precision"></div></div>
            </div>
            <div class="plot-row">
                <div class="plot-cell"><div id="plot-prostate-recall"></div></div>
                <div class="plot-cell"><div id="plot-target-recall"></div></div>
            </div>
        </div>

        <div class="plot-container">
            <h2>Hyperparameter Comparison</h2>
            <div class="plot-row">
                <div class="plot-cell"><div id="plot-by-scheduler"></div></div>
                <div class="plot-cell"><div id="plot-by-loss"></div></div>
                <div class="plot-cell"><div id="plot-by-lr"></div></div>
            </div>
        </div>

        <div class="plot-container">
            <h2>Top 20 Experiments</h2>
            <table>
                <tr><th>Rank</th><th>Job ID</th><th>Model</th><th>Scheduler</th><th>Loss</th><th>LR</th><th>Best Dice</th><th>Best Epoch</th><th>Total Epochs</th></tr>
                {table_rows}
            </table>
        </div>
    </div>

    <script>
        const epochsData = {epochs_json};
        const summaryData = {summary_json};
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

        function getSelectedExperiments() {{
            const checkboxes = document.querySelectorAll('#experiment-checkboxes input:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }}

        function selectTop(n) {{
            const checkboxes = document.querySelectorAll('#experiment-checkboxes input');
            checkboxes.forEach((cb, i) => cb.checked = i < n);
            updatePlots();
        }}

        function selectAll() {{
            document.querySelectorAll('#experiment-checkboxes input').forEach(cb => cb.checked = true);
            updatePlots();
        }}

        function selectNone() {{
            document.querySelectorAll('#experiment-checkboxes input').forEach(cb => cb.checked = false);
            updatePlots();
        }}

        function createLinePlot(containerId, title, yField, yTitle) {{
            const selected = getSelectedExperiments();
            const filteredData = epochsData.filter(d => selected.includes(d.job_id));

            const grouped = {{}};
            filteredData.forEach(d => {{
                if (!grouped[d.job_id]) grouped[d.job_id] = [];
                grouped[d.job_id].push(d);
            }});

            const traces = [];
            Object.keys(grouped).forEach((jobId, idx) => {{
                const data = grouped[jobId].sort((a, b) => a.epoch - b.epoch);
                traces.push({{
                    x: data.map(d => d.epoch),
                    y: data.map(d => d[yField]),
                    type: 'scatter',
                    mode: 'lines',
                    name: data[0].label,
                    line: {{ color: colors[idx % colors.length] }},
                    hovertemplate: 'Job: ' + jobId + '<br>Epoch: %{{x}}<br>' + yTitle + ': %{{y:.4f}}<extra></extra>'
                }});
            }});

            const layout = {{
                title: title,
                xaxis: {{ title: 'Epoch' }},
                yaxis: {{ title: yTitle }},
                height: 350,
                margin: {{ t: 40, b: 40, l: 60, r: 20 }},
                showlegend: false
            }};

            Plotly.newPlot(containerId, traces, layout);
        }}

        function updatePlots() {{
            createLinePlot('plot-train-loss', 'Training Loss', 'train_loss', 'Loss');
            createLinePlot('plot-val-loss', 'Validation Loss', 'val_loss', 'Loss');
            createLinePlot('plot-train-dice', 'Training Dice', 'train_dice', 'Dice');
            createLinePlot('plot-val-dice', 'Validation Dice', 'val_dice', 'Dice');
            createLinePlot('plot-prostate-precision', 'Prostate Precision', 'val_precision_prostate', 'Precision');
            createLinePlot('plot-target-precision', 'Target Precision', 'val_precision_target', 'Precision');
            createLinePlot('plot-prostate-recall', 'Prostate Recall', 'val_recall_prostate', 'Recall');
            createLinePlot('plot-target-recall', 'Target Recall', 'val_recall_target', 'Recall');
        }}

        function renderHyperparameterPlots() {{
            // By Scheduler
            const schedulerData = {{}};
            summaryData.forEach(d => {{
                if (!schedulerData[d.scheduler]) schedulerData[d.scheduler] = [];
                schedulerData[d.scheduler].push(d.best_val_dice);
            }});

            const schedulerTraces = Object.keys(schedulerData).map(scheduler => ({{
                y: schedulerData[scheduler],
                name: scheduler,
                type: 'box',
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.8
            }}));

            Plotly.newPlot('plot-by-scheduler', schedulerTraces, {{
                title: 'By Scheduler',
                yaxis: {{ title: 'Best Val Dice' }},
                height: 350,
                showlegend: false
            }});

            // By Loss
            const lossData = {{}};
            summaryData.forEach(d => {{
                if (!lossData[d.loss]) lossData[d.loss] = [];
                lossData[d.loss].push(d.best_val_dice);
            }});

            const lossTraces = Object.keys(lossData).map(loss => ({{
                y: lossData[loss],
                name: loss,
                type: 'box',
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.8
            }}));

            Plotly.newPlot('plot-by-loss', lossTraces, {{
                title: 'By Loss Function',
                yaxis: {{ title: 'Best Val Dice' }},
                height: 350,
                showlegend: false
            }});

            // By LR
            const lrTrace = {{
                x: summaryData.map(d => d.lr),
                y: summaryData.map(d => d.best_val_dice),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 10,
                    color: summaryData.map(d => d.best_val_dice),
                    colorscale: 'Viridis',
                    showscale: true
                }},
                text: summaryData.map(d => 'Job: ' + d.job_id + '<br>Model: ' + d.model),
                hovertemplate: '%{{text}}<br>LR: %{{x}}<br>Best Dice: %{{y:.4f}}<extra></extra>'
            }};

            Plotly.newPlot('plot-by-lr', [lrTrace], {{
                title: 'By Learning Rate',
                xaxis: {{ title: 'Learning Rate', type: 'log' }},
                yaxis: {{ title: 'Best Val Dice' }},
                height: 350
            }});
        }}

        // Initialize
        updatePlots();
        renderHyperparameterPlots();
    </script>
</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Saved interactive HTML report: {output_path}")


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

    best_run = next((r for r in runs if r.job_id == best['job_id']), None)
    if best_run:
        print(f"  Model Parameters: {best_run.model_params:,}")
        if best_run.ft_gamma:
            print(f"  FT Gamma: {best_run.ft_gamma}")
        if best_run.ft_class_weights:
            print(f"  FT Class Weights: {best_run.ft_class_weights}")

    print("\n" + "-"*80)
    print("PERFORMANCE BY SCHEDULER")
    print("-"*80)
    sched_perf = df.groupby('scheduler')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
    sched_perf = sched_perf.sort_values('max', ascending=False)
    sched_perf.columns = ['Mean Dice', 'Max Dice', 'Std', 'Count']
    print(sched_perf.to_string())

    print("\n" + "-"*80)
    print("PERFORMANCE BY LOSS FUNCTION")
    print("-"*80)
    loss_perf = df.groupby('loss')['best_val_dice'].agg(['mean', 'max', 'std', 'count'])
    loss_perf = loss_perf.sort_values('max', ascending=False)
    loss_perf.columns = ['Mean Dice', 'Max Dice', 'Std', 'Count']
    print(loss_perf.to_string())

    print("\n" + "-"*80)
    print("PERFORMANCE BY LEARNING RATE")
    print("-"*80)
    lr_perf = df.groupby('lr')['best_val_dice'].agg(['mean', 'max', 'count'])
    lr_perf = lr_perf.sort_values('max', ascending=False)
    lr_perf.columns = ['Mean Dice', 'Max Dice', 'Count']
    print(lr_perf.to_string())

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze MRI training logs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory containing SLURM log files")
    parser.add_argument("--output", type=str, default="results/training_analysis.html", help="Output path for HTML report")
    parser.add_argument("--save-csv", type=str, default=None, help="Save summary DataFrame to CSV")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return 1

    log_files = sorted(logs_dir.glob("slurm-*.out"))
    if not log_files:
        print(f"Error: No SLURM log files found in {logs_dir}")
        return 1

    print(f"Found {len(log_files)} log files in {logs_dir}")

    runs = []
    for log_file in log_files:
        run = parse_log_file(log_file)
        if run:
            runs.append(run)

    print(f"Successfully parsed {len(runs)} training runs")

    if not runs:
        print("Error: No valid training runs found")
        return 1

    df = create_summary_dataframe(runs)
    print_summary_report(df, runs)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary CSV: {csv_path}")

    print("\nGenerating interactive HTML report...")
    create_interactive_html_report(runs, df, Path(args.output))

    return 0


if __name__ == "__main__":
    exit(main())
