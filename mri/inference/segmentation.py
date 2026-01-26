"""Segmentation inference and prediction export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri.data.metadata import load_metadata


def run_segmentation_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metadata_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> None:
    model = model.to(device)
    model.eval()

    meta = load_metadata(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_buffers: Dict[str, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            metas = batch[2]
            if isinstance(metas, dict):
                meta_list = [
                    {k: metas[k][i] for k in metas}
                    for i in range(len(metas[next(iter(metas))]))
                ]
            else:
                meta_list = metas

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, m in enumerate(meta_list):
                case_id = m["case_id"]
                slice_idx = int(m["slice_idx"])
                if case_id not in case_buffers:
                    num_slices = int(meta.cases[case_id]["num_slices"])
                    h, w = probs.shape[2], probs.shape[3]
                    case_buffers[case_id] = {
                        "target": np.zeros((num_slices, h, w), dtype=np.float32),
                        "prostate": np.zeros((num_slices, h, w), dtype=np.float32),
                    }
                case_buffers[case_id]["prostate"][slice_idx] = probs[i, 0]
                if probs.shape[1] > 1:
                    case_buffers[case_id]["target"][slice_idx] = probs[i, 1]

    for case_id, arrays in case_buffers.items():
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "prostate_prob.npy", arrays["prostate"])
        np.save(case_dir / "target_prob.npy", arrays["target"])
