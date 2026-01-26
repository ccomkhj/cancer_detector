# Scripts Overview

## Train (HPC / Singularity)
```
./scripts/train/run.sh --config mri/config/task/segmentation.yaml
./scripts/train/run.sh --config mri/config/task/classification.yaml
```

## Inference (HPC / Singularity)
```
./scripts/inference/run.sh --config mri/config/task/segmentation.yaml --split test
./scripts/inference/run.sh --config mri/config/task/classification.yaml --split test
```

## Requirements
- `.env` must exist in project root (e.g., `SINGULARITY_IMAGE`, `DATA_DIR`, `CHECKPOINT_DIR`, `WANDB_DIR`)
- Singularity or Apptainer available on the HPC cluster

## Legacy Scripts
See `scripts/legacy/` for older SLURM submission helpers.
