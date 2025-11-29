#!/usr/bin/env python3
"""
Debug Aim - Check what metrics are actually logged
"""
from pathlib import Path

try:
    from aim import Repo
except ImportError:
    print("❌ Aim not installed!")
    exit(1)

# Connect to Aim repo
project_root = Path(__file__).parent.parent
repo_path = project_root / ".aim"
print(f"Checking Aim repository: {repo_path}\n")

# Repo() expects the parent directory, not the .aim directory itself
repo = Repo(str(project_root))

# Get all runs
runs = list(repo.iter_runs())
print(f"Total runs: {len(runs)}\n")

for i, run in enumerate(runs, 1):
    print(f"\n{'='*80}")
    print(f"Run {i}: {run.hash}")
    print(f"{'='*80}")

    # Basic info
    print(f"Name: {run.name}")
    print(f"Experiment: {run.experiment}")
    print(f"Creation time: {run.creation_time}")

    # Check tracked metrics
    print(f"\n📊 Tracked Metrics:")

    tracked_metrics = set()
    metric_contexts = {}

    try:
        # Get all metric traces
        metrics_collection = run.metrics()
        metric_count = 0

        for metric_trace in metrics_collection:
            metric_name = metric_trace.name
            metric_context = metric_trace.context
            metric_count += 1

            if metric_name not in tracked_metrics:
                tracked_metrics.add(metric_name)
                print(f"  ✓ {metric_name}")
                metric_contexts[metric_name] = []

            if metric_context not in metric_contexts[metric_name]:
                metric_contexts[metric_name].append(metric_context)
                print(f"      Context: {metric_context}")

        if metric_count == 0:
            print("  ⚠️  No metrics found!")
        else:
            print(
                f"\n  Total: {len(tracked_metrics)} unique metrics, {metric_count} traces"
            )

    except Exception as e:
        print(f"  ⚠️  Error accessing metrics: {e}")

        # Check what's actually in the run
        print(f"\n🔍 Checking run contents:")
        try:
            print(f"  Run keys: {list(run.keys())}")
        except Exception as e2:
            print(f"  Error getting keys: {e2}")

    # Check hyperparameters
    print(f"\n⚙️  Hyperparameters:")
    try:
        hparams = run.get("hparams", {})
        if hparams:
            for key, value in hparams.items():
                print(f"  - {key}: {value}")
        else:
            print("  No hyperparameters found")
    except Exception as e:
        print(f"  Error accessing hparams: {e}")

print(f"\n{'='*80}")
print("Diagnosis:")
print("If 'No metrics found' appears, the metrics aren't being logged correctly.")
print("If metrics are found, they should be visible in Aim UI.")
print(f"{'='*80}\n")
