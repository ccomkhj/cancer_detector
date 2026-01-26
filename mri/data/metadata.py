"""Metadata loading helpers for aligned_v2 metadata.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import json


@dataclass
class Metadata:
    path: Path
    raw: Dict[str, Any]

    @property
    def samples(self) -> List[Dict[str, Any]]:
        return self.raw.get("samples", [])

    @property
    def cases(self) -> Dict[str, Dict[str, Any]]:
        return self.raw.get("cases", {})

    @property
    def config(self) -> Dict[str, Any]:
        return self.raw.get("config", {})


def load_metadata(path: str | Path) -> Metadata:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found: {path}")
    with path.open() as f:
        raw = json.load(f)
    return Metadata(path=path, raw=raw)
