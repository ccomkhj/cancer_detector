#!/usr/bin/env python3
"""Generate train/val/test split YAML files from metadata.json."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mri.data.metadata import load_metadata


def _split_list(items: List[str], ratios: List[float], rng: random.Random) -> Dict[str, List[str]]:
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def build_splits(meta_path: str | Path, ratios: List[float], seed: int, stratify: bool) -> Dict[str, List[str]]:
    meta = load_metadata(meta_path)
    rng = random.Random(seed)

    if not stratify:
        return _split_list(list(meta.cases.keys()), ratios, rng)

    # Stratify by class
    by_class: Dict[int, List[str]] = {}
    for case_id, case_info in meta.cases.items():
        cls = int(case_info.get("class", 0))
        by_class.setdefault(cls, []).append(case_id)

    splits = {"train": [], "val": [], "test": []}
    for case_ids in by_class.values():
        part = _split_list(case_ids, ratios, rng)
        for k in splits:
            splits[k].extend(part[k])

    # Shuffle final splits to remove class ordering
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate YAML splits from metadata.json")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output YAML path")
    parser.add_argument("--ratios", default="0.7,0.15,0.15", help="Comma-separated ratios train,val,test")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--no-stratify", action="store_true", help="Disable class stratification")
    args = parser.parse_args(argv)

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must be three numbers summing to 1.0")

    splits = build_splits(args.metadata, ratios, args.seed, stratify=not args.no_stratify)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(splits, f, sort_keys=False)

    print(f"Wrote splits to {out_path}")
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(splits[k])} cases")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
