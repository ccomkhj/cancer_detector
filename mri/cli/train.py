"""Training entrypoint for the scalable MRI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from mri.config.loader import load_config
from mri.data.metadata import load_metadata
from mri.data.index_builders import load_split_file, build_segmentation_index, build_classification_index
from mri.data.datasets.segmentation import SegmentationDataset
from mri.data.datasets.classification import ClassificationDataset
from mri.models import create_segmentation_model, create_classification_model
from mri.tasks.segmentation import SegmentationTask
from mri.tasks.classification import ClassificationTask
from mri.training.trainer import Trainer, resolve_device


def _build_dataloaders(cfg: Dict[str, Any], task_name: str):
    meta = load_metadata(cfg["data"]["metadata"])
    splits = load_split_file(cfg["data"]["split_file"])
    num_workers = cfg["data"].get("num_workers", 4)

    if task_name == "segmentation":
        stack_depth = cfg["data"].get("stack_depth", meta.config.get("t2_context_window", 5))
        train_index = build_segmentation_index(meta, splits["train"])
        val_index = build_segmentation_index(meta, splits["val"])
        if len(train_index) == 0:
            raise ValueError(
                "Train split is empty. Populate the split YAML first, e.g.:\n"
                "  python tools/generate_splits.py --metadata data/aligned_v2/metadata.json "
                "--output data/splits/seg_cases.yaml"
            )

        train_ds = SegmentationDataset(
            metadata_path=cfg["data"]["metadata"],
            samples_index=train_index,
            stack_depth=stack_depth,
            normalize=True,
        )
        val_ds = SegmentationDataset(
            metadata_path=cfg["data"]["metadata"],
            samples_index=val_index,
            stack_depth=stack_depth,
            normalize=True,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    if task_name == "classification":
        train_index = build_classification_index(meta, splits["train"])
        val_index = build_classification_index(meta, splits["val"])
        if len(train_index) == 0:
            raise ValueError(
                "Train split is empty. Populate the split YAML first, e.g.:\n"
                "  python tools/generate_splits.py --metadata data/aligned_v2/metadata.json "
                "--output data/splits/cls_cases.yaml"
            )

        train_ds = ClassificationDataset(
            metadata_path=cfg["data"]["metadata"],
            cases_index=train_index,
            seg_pred_dir=cfg["data"]["seg_pred_dir"],
            depth=cfg["data"]["depth"]["depth"],
            crop_size=cfg["data"]["roi"]["crop_size"],
            output_size=cfg["data"]["roi"]["output_size"],
            modalities=tuple(cfg["data"]["modalities"]),
            zero_pad_missing=cfg["data"].get("zero_pad_missing", True),
            selection_source=cfg["data"]["selection"].get("source", "pred"),
            selection_jitter=cfg["data"]["selection"].get("jitter", 2),
            min_prob=cfg["data"]["selection"].get("min_prob", 0.3),
            use_roi=cfg["data"]["roi"].get("use_roi", True),
            normalize=True,
        )
        val_ds = ClassificationDataset(
            metadata_path=cfg["data"]["metadata"],
            cases_index=val_index,
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

        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    raise ValueError(f"Unknown task: {task_name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MRI unified training CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", help="Override train.output_dir")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--batch_size", type=int, help="Override train.batch_size")
    parser.add_argument("--lr", type=float, help="Override train.lr")
    parser.add_argument("--device", help="Override train.device")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.device:
        cfg["train"]["device"] = args.device
    task_name = cfg["task"]["name"]

    device = resolve_device(cfg["train"].get("device", "auto"))

    train_loader, val_loader = _build_dataloaders(cfg, task_name)

    if task_name == "segmentation":
        model = create_segmentation_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
        task = SegmentationTask(loss_name=cfg["loss"]["name"], loss_params=cfg["loss"].get("params", {}))
    elif task_name == "classification":
        model = create_classification_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
        num_classes = cfg["model"].get("params", {}).get("num_classes", 2)
        task = ClassificationTask(num_classes=num_classes, loss_name=cfg["loss"]["name"], loss_params=cfg["loss"].get("params", {}))
    else:
        raise ValueError(f"Unknown task: {task_name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"]["name"]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{model_name}_{timestamp}"

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        run_name=run_name,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["train"]["epochs"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
