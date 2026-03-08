"""Inference entrypoint for the scalable MRI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from mri.config.loader import load_config
from mri.data.metadata import load_metadata
from mri.data.index_builders import load_split_file, build_segmentation_index, build_classification_index
from mri.data.datasets.segmentation import SegmentationDataset
from mri.data.datasets.classification import ClassificationDataset
from mri.models import create_segmentation_model, create_classification_model
from mri.training.trainer import resolve_device
from mri.inference.segmentation import run_segmentation_inference
from mri.inference.classification import run_classification_inference


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)


def _build_dataloader(cfg: Dict[str, Any], task_name: str, split_key: str):
    meta = load_metadata(cfg["data"]["metadata"])
    splits = load_split_file(cfg["data"]["split_file"])
    num_workers = cfg["data"].get("num_workers", 4)

    if task_name == "segmentation":
        stack_depth = cfg["data"].get("stack_depth", meta.config.get("t2_context_window", 5))
        split_index = build_segmentation_index(meta, splits[split_key])
        ds = SegmentationDataset(
            metadata_path=cfg["data"]["metadata"],
            samples_index=split_index,
            stack_depth=stack_depth,
            normalize=True,
        )
        return DataLoader(ds, batch_size=cfg["inference"]["batch_size"], shuffle=False, num_workers=num_workers)

    if task_name == "classification":
        split_index = build_classification_index(meta, splits[split_key])
        ds = ClassificationDataset(
            metadata_path=cfg["data"]["metadata"],
            cases_index=split_index,
            seg_pred_dir=cfg["data"]["seg_pred_dir"],
            depth=cfg["data"]["depth"]["depth"],
            crop_size=cfg["data"]["roi"]["crop_size"],
            output_size=cfg["data"]["roi"]["output_size"],
            modalities=tuple(cfg["data"]["modalities"]),
            zero_pad_missing=cfg["data"].get("zero_pad_missing", True),
            selection_source="pred",
            selection_jitter=0,
            min_prob=cfg["data"]["selection"].get("min_prob", 0.3),
            use_roi=cfg["data"]["roi"].get("use_roi", True),
            normalize=True,
        )
        return DataLoader(ds, batch_size=cfg["inference"]["batch_size"], shuffle=False, num_workers=num_workers)

    raise ValueError(f"Unknown task: {task_name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MRI unified inference CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--split", default="test", help="Split key to run inference on")
    parser.add_argument("--checkpoint", help="Override inference.checkpoint")
    parser.add_argument("--output_dir", help="Override inference.output_dir")
    parser.add_argument("--batch_size", type=int, help="Override inference.batch_size")
    parser.add_argument("--device", help="Override inference.device")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["inference"]["checkpoint"] = args.checkpoint
    if args.output_dir:
        cfg["inference"]["output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["inference"]["batch_size"] = args.batch_size
    if args.device:
        cfg["inference"]["device"] = args.device
    task_name = cfg["task"]["name"]
    device = resolve_device(cfg["inference"].get("device", "auto"))

    dataloader = _build_dataloader(cfg, task_name, args.split)

    if task_name == "segmentation":
        model = create_segmentation_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
        if cfg["inference"].get("checkpoint"):
            _load_checkpoint(model, cfg["inference"]["checkpoint"], device)
        run_segmentation_inference(
            model=model,
            dataloader=dataloader,
            metadata_path=cfg["data"]["metadata"],
            output_dir=cfg["inference"]["output_dir"],
            device=device,
        )
        return 0

    if task_name == "classification":
        model = create_classification_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
        if cfg["inference"].get("checkpoint"):
            _load_checkpoint(model, cfg["inference"]["checkpoint"], device)
        output_csv = Path(cfg["inference"]["output_dir"]) / "predictions.csv"
        run_classification_inference(
            model=model,
            dataloader=dataloader,
            device=device,
            output_csv=output_csv,
        )
        return 0

    raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    raise SystemExit(main())
