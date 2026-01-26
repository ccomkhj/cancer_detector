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

## Train (SLURM)
```
sbatch scripts/train/srun.sh --config mri/config/task/segmentation.yaml
sbatch scripts/train/srun.sh --config mri/config/task/classification.yaml
```

## Inference (SLURM)
```
sbatch scripts/inference/srun.sh --config mri/config/task/segmentation.yaml --split test
sbatch scripts/inference/srun.sh --config mri/config/task/classification.yaml --split test
```

## Requirements
- `.env` must exist in project root (e.g., `SINGULARITY_IMAGE`, `DATA_DIR`, `CHECKPOINT_DIR`, `WANDB_DIR`)
- Optional: `SPLITS_DIR` can override where `data/splits/*.yaml` are mounted from (defaults to `${PROJECT_DIR}/data/splits`)
- Singularity or Apptainer available on the HPC cluster
- For SLURM, edit `scripts/train/srun.sh` and `scripts/inference/srun.sh` to match your account/partition/resources

## Legacy Scripts
See `scripts/legacy/` for older SLURM submission helpers.
