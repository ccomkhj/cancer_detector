# SLURM (HPC Cluster) Job Submission

This repo includes SLURM submission scripts in `scripts/` and supports running training either:

- **In a container** (recommended on HPC): Singularity/Apptainer `.sif`
- **Natively** (no container): Conda/venv on the cluster

## 1) Prepare the project on the cluster

1. Copy/clone this repo onto the cluster (preferably on a fast filesystem like `$SCRATCH`).
2. Ensure your data is available under `data/` (or adjust bind mounts in the job script).
3. Create output directories (if they don’t exist yet):

```bash
mkdir -p checkpoints .aim logs
```

## 2) Build a container image (optional, but recommended)

Most clusters do **not** allow `docker run` on compute nodes. Typical workflow:

1. Build a Docker image on your laptop/workstation
2. Convert it to a Singularity/Apptainer `.sif`
3. Submit with `sbatch`

### Option A: Build Docker → Convert to `.sif`

Build with the SLURM-friendly Dockerfile:

```bash
docker build -f DockerfileSlurm -t mri-train:slurm .
docker save mri-train:slurm -o mri-train-slurm.tar
scp mri-train-slurm.tar <user>@<cluster>:/path/to/project/
```

On the cluster:

```bash
module load apptainer 2>/dev/null || true
apptainer build mri-train.sif docker-archive://mri-train-slurm.tar
```

If your cluster provides `singularity` instead of `apptainer`, use:

```bash
singularity build mri-train.sif docker-archive://mri-train-slurm.tar
```

### Option B: Use the provided helper

If you have Docker + Singularity/Apptainer available on the same machine, you can try:

```bash
./scripts/build_singularity.sh
```

It builds `mri-train.sif` in the project root.

## 3) Submit a training job

### Container mode (default)

`scripts/submit_slurm.sh` runs inside a `.sif` by default:

```bash
sbatch scripts/submit_slurm.sh
```

Custom config / overrides:

```bash
sbatch scripts/submit_slurm.sh --config config_onecycle.yaml
sbatch scripts/submit_slurm.sh --epochs 100 --batch_size 16
```

If your `.sif` is not named `mri-train.sif`:

```bash
SINGULARITY_IMAGE=/path/to/mri-train.sif sbatch scripts/submit_slurm.sh
```

### wandb logging (optional)

```bash
export WANDB_API_KEY="your_key"
sbatch scripts/submit_slurm_wandb.sh
```

### Native Python mode (no container)

```bash
USE_SINGULARITY=0 CONDA_ENV=mri sbatch scripts/submit_slurm.sh
```

## 4) Monitor / debug

```bash
squeue -u $USER
tail -f slurm-<jobid>.out
tail -f slurm-<jobid>.err
scancel <jobid>
```

## 5) Common cluster-specific edits

Edit the `#SBATCH` block at the top of `scripts/submit_slurm.sh`:

- `--partition`, `--account` (often required)
- `--time`, `--mem`, `--cpus-per-task`
- `--gres=gpu:<N>` for GPUs

