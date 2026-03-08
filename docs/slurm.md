# SLURM

The supported HPC path is the native wrapper pair in `scripts/new/`.

## Train

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
```

## Inference

```bash
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

## Dry Run

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/classification.yaml --split test
```

Container-era SLURM scripts were moved to `archive/scripts/`.
