"""Segmentation task definitions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import Task
from service.metrics import compute_dice_score
from service.models import DiceLoss, DiceBCELoss, FocalTverskyLoss


class SegmentationTask(Task):
    name = "segmentation"

    def __init__(self, loss_name: str = "dice_bce", loss_params: Dict | None = None):
        self.loss_name = loss_name
        self.loss_params = loss_params or {}
        self.loss_fn = self._build_loss(loss_name, self.loss_params)

    def _build_loss(self, loss_name: str, params: Dict) -> nn.Module:
        if loss_name == "dice":
            return DiceLoss()
        if loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        if loss_name == "dice_bce":
            return DiceBCELoss(
                dice_weight=params.get("dice_weight", 0.5),
                bce_weight=params.get("bce_weight", 0.5),
            )
        if loss_name == "focal_tversky":
            return FocalTverskyLoss(
                alpha=params.get("alpha", [0.6, 0.8]),
                beta=params.get("beta", [0.4, 0.2]),
                gamma=params.get("gamma", 1.33),
                class_weights=params.get("class_weights", [1.0, 2.0]),
            )
        raise ValueError(f"Unknown segmentation loss: {loss_name}")

    def training_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, masks = batch[0], batch[1]
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        dice = compute_dice_score(logits.detach().cpu(), masks.detach().cpu())
        metrics = {"loss": loss.item(), "dice": dice}
        return loss, metrics

    def validation_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, masks = batch[0], batch[1]
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        dice = compute_dice_score(logits.detach().cpu(), masks.detach().cpu())
        metrics = {"loss": loss.item(), "dice": dice}
        return loss, metrics

    def aggregate_metrics(self, metrics_list: list[Dict]) -> Dict:
        if not metrics_list:
            return {}
        agg = {"loss": 0.0, "dice": 0.0}
        for m in metrics_list:
            for k in agg:
                agg[k] += m.get(k, 0.0)
        for k in agg:
            agg[k] /= len(metrics_list)
        return agg

    def primary_metric(self, metrics: Dict) -> float:
        return metrics.get("dice", 0.0)
