"""Classification inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import csv
import torch
from torch.utils.data import DataLoader


def run_classification_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_csv: str | Path | None = None,
) -> List[Dict]:
    model = model.to(device)
    model.eval()

    results: List[Dict] = []
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
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i, m in enumerate(meta_list):
                results.append(
                    {
                        "case_id": m["case_id"],
                        "pred": int(preds[i].item()),
                        "prob": float(probs[i, preds[i]].item()),
                    }
                )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "pred", "prob"])
            writer.writeheader()
            writer.writerows(results)

    return results
