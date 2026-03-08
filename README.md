# MRI Modular Pipeline (New Architecture)

This repository uses a config-driven modular architecture for prostate MRI segmentation and classification.

**Official execution paths:**
- `mri/cli/train.py`
- `mri/cli/infer.py`
- `scripts/new/train`
- `scripts/new/inference`

Legacy script paths are intentionally not documented here.

## Architecture Overview

The pipeline is organized around reusable modules in `mri/`:

- `mri/config/`: YAML config loading (`defaults + task config`)
- `mri/data/`: metadata loading, split/index builders, datasets
- `mri/models/`: segmentation/classification model registry + builders
- `mri/tasks/`: task-specific loss/metric handling
- `mri/training/`: trainer loop, checkpointing, device resolution
- `mri/inference/`: segmentation/classification inference runners
- `mri/cli/`: train/infer entrypoints

## Repository Structure

```text
mri/
  cli/
    train.py
    infer.py
  config/
    defaults.yaml
    task/
      segmentation.yaml
      classification.yaml
  data/
  models/
  tasks/
  training/
  inference/

scripts/
  new/
    train
    inference

tools/
  generate_splits.py
```

## Environment Setup

```bash
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## Data Split Preparation

Generate split YAMLs from metadata:

```bash
python tools/generate_splits.py --metadata data/aligned_v2/metadata.json --output data/splits/seg_cases.yaml
python tools/generate_splits.py --metadata data/aligned_v2/metadata.json --output data/splits/cls_cases.yaml
```

## Training

### Local

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
python mri/cli/train.py --config mri/config/task/classification.yaml
```

### HPC (Native, No Container)

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
```

You can pass selected runtime overrides on the CLI, for example:

```bash
sbatch scripts/new/train --config mri/config/task/segmentation.yaml --epochs 50 --batch_size 16
```

## Inference

Set inference values in the config (especially `inference.checkpoint` and `inference.output_dir`), then run:

```bash
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test
python mri/cli/infer.py --config mri/config/task/classification.yaml --split test
```

HPC native launch:

```bash
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

## Native HPC Script Behavior

`scripts/new/train` and `scripts/new/inference`:
- run directly on node Python environment
- support both direct run and `sbatch`
- auto-load `.env` if present
- auto-export runtime paths/caches (`DATA_DIR`, `CHECKPOINT_DIR`, `WANDB_DIR`, `TORCH_HOME`, `HF_HOME`)
- use `srun` automatically inside a SLURM allocation

Dry-run validation:

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/segmentation.yaml --split test
```

## Configuration Contract

Task configs are expected to define:
- `task.name`: `segmentation` or `classification`
- `data.metadata`
- `data.split_file`
- `model.name` and optional `model.params`
- `train.*` and `inference.*` sections (missing fields fall back to `mri/config/defaults.yaml`)

## Notes

- This README reflects only the new architecture.
- For production runs, prefer `scripts/new/*` on HPC and `mri/cli/*` locally.
