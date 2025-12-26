#!/usr/bin/env python3
"""
Validate sequence matching by creating per-case grids of T2/ep2d_calc/ep2d_adc.

Each output image has 3 columns (t2, ep2d_calc, ep2d_adc) and N rows of slices.
Rows are sampled slice indices (shared baseline), with nearest available slices
per sequence when exact indices are missing.

Output:
  validation_results/class{n}/case_{i}/sequence_match_{series_uid}.png
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SEQUENCE_CONFIGS = {
    "t2": {"processed_dir": Path("data/processed")},
    "ep2d_calc": {"processed_dir": Path("data/processed_ep2d_calc")},
    "ep2d_adc": {"processed_dir": Path("data/processed_ep2d_adc")},
}


def read_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def list_indices(images_dir: Path) -> List[int]:
    indices = []
    for img_file in images_dir.glob("*.png"):
        try:
            indices.append(int(img_file.stem))
        except ValueError:
            continue
    return sorted(indices)


def build_case_image_series_index(case_dir: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    if not case_dir.exists():
        return index

    for series_dir in case_dir.iterdir():
        if not series_dir.is_dir():
            continue
        images_dir = series_dir / "images"
        if not images_dir.exists():
            continue
        meta = read_meta(series_dir / "meta.json")
        if "num_slices" not in meta:
            meta["num_slices"] = len(list(images_dir.glob("*.png")))
        index[series_dir.name] = meta
    return index


def select_image_series_uid(
    seg_series_uid: str,
    t2_case_dir: Optional[Path],
    image_series_meta_index: Dict[str, Dict],
) -> Tuple[Optional[str], str]:
    if seg_series_uid in image_series_meta_index:
        return seg_series_uid, "exact"

    seg_meta = {}
    if t2_case_dir is not None:
        seg_meta = read_meta(t2_case_dir / seg_series_uid / "meta.json")

    study_uid = seg_meta.get("StudyInstanceUID")
    target_slices = seg_meta.get("num_slices")

    if study_uid:
        candidates = {
            uid: meta
            for uid, meta in image_series_meta_index.items()
            if meta.get("StudyInstanceUID") == study_uid
        }
        if candidates:
            if target_slices is None:
                target_slices = max(m.get("num_slices", 0) for m in candidates.values())
            best_uid = min(
                candidates.keys(),
                key=lambda uid: abs(
                    int(candidates[uid].get("num_slices", 0)) - int(target_slices)
                ),
            )
            return best_uid, "study_uid"

    if not image_series_meta_index:
        return None, "missing"

    if target_slices is None:
        target_slices = 0
    best_uid = min(
        image_series_meta_index.keys(),
        key=lambda uid: abs(
            int(image_series_meta_index[uid].get("num_slices", 0)) - int(target_slices)
        ),
    )
    return best_uid, "fallback"


def nearest_index(target: int, indices: List[int]) -> Optional[int]:
    if not indices:
        return None
    return min(indices, key=lambda x: abs(x - target))


def choose_base_indices(
    indices_by_seq: Dict[str, List[int]], max_slices: int
) -> List[int]:
    available_sets = [set(v) for v in indices_by_seq.values() if v]
    if not available_sets:
        return []
    intersection = set.intersection(*available_sets)
    if intersection:
        base = sorted(intersection)
    else:
        base = sorted(indices_by_seq.get("t2", []))

    if not base:
        return []

    if len(base) <= max_slices:
        return base

    step = (len(base) - 1) / (max_slices - 1)
    return [base[int(round(i * step))] for i in range(max_slices)]


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def pad_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, color=(0, 0, 0))
    img_rgb = img.convert("RGB")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img_rgb, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, text: str, position: Tuple[int, int]) -> None:
    font = ImageFont.load_default()
    draw.text(position, text, fill=(255, 255, 0), font=font)


def build_case_grid(
    case_output_dir: Path,
    class_name: str,
    case_name: str,
    series_uid: str,
    images_by_seq: Dict[str, Dict],
    indices_by_seq: Dict[str, List[int]],
    match_types: Dict[str, str],
    max_slices: int,
) -> Optional[Path]:
    base_indices = choose_base_indices(indices_by_seq, max_slices)
    if not base_indices:
        print(f"    ⚠ No slices found for {class_name}/{case_name}")
        return None

    # Determine tile size
    max_w = 0
    max_h = 0
    for seq_name, seq_data in images_by_seq.items():
        indices = indices_by_seq.get(seq_name, [])
        if not indices:
            continue
        img_path = seq_data["images_dir"] / f"{indices[0]:04d}.png"
        if not img_path.exists():
            continue
        img = load_image(img_path)
        max_w = max(max_w, img.width)
        max_h = max(max_h, img.height)

    if max_w == 0 or max_h == 0:
        return None

    cols = ["t2", "ep2d_calc", "ep2d_adc"]
    header_h = 28
    tile_w = max_w
    tile_h = max_h
    canvas_w = tile_w * len(cols)
    canvas_h = header_h + tile_h * len(base_indices)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    for col_idx, seq_name in enumerate(cols):
        header = f"{seq_name} ({match_types.get(seq_name, 'n/a')})"
        draw_label(
            draw,
            header,
            (col_idx * tile_w + 4, 6),
        )

    # Populate grid
    for row_idx, base_idx in enumerate(base_indices):
        for col_idx, seq_name in enumerate(cols):
            seq_data = images_by_seq.get(seq_name)
            if seq_data is None:
                continue
            indices = indices_by_seq.get(seq_name, [])
            if not indices:
                continue
            img_idx = nearest_index(base_idx, indices)
            if img_idx is None:
                continue
            img_path = seq_data["images_dir"] / f"{img_idx:04d}.png"
            if not img_path.exists():
                continue
            img = load_image(img_path)
            tile = pad_image(img, (tile_w, tile_h))
            tile_draw = ImageDraw.Draw(tile)
            label = f"{img_idx:04d} {img.width}x{img.height}"
            draw_label(tile_draw, label, (4, 4))
            x = col_idx * tile_w
            y = header_h + row_idx * tile_h
            canvas.paste(tile, (x, y))

    output_path = case_output_dir / f"sequence_match_{series_uid}.png"
    canvas.save(output_path)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create per-case sequence match grids (t2/ep2d_calc/ep2d_adc)."
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        help="Validate specific class only (e.g., class1, class2)",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=10,
        help="Max slices (rows) per case (default: 10)",
    )

    args = parser.parse_args()

    processed_seg_root = Path("data/processed_seg")
    if not processed_seg_root.exists():
        print(f"✗ Processed seg directory not found: {processed_seg_root}")
        sys.exit(1)

    if args.class_name:
        classes = [args.class_name]
    else:
        classes = sorted([d.name for d in processed_seg_root.glob("class*") if d.is_dir()])

    for class_name in classes:
        print(f"\n=== {class_name} ===")
        seg_class_dir = processed_seg_root / class_name
        if not seg_class_dir.exists():
            print(f"  ✗ Missing seg class dir: {seg_class_dir}")
            continue

        case_dirs = sorted([d for d in seg_class_dir.glob("case_*") if d.is_dir()])
        if not case_dirs:
            print(f"  ✗ No cases found in {seg_class_dir}")
            continue

        for case_dir in case_dirs:
            case_name = case_dir.name
            series_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
            if not series_dirs:
                continue

            # Use first series unless multiple exist
            for series_dir in series_dirs:
                series_uid = series_dir.name
                images_by_seq = {}
                indices_by_seq = {}
                match_types = {}

                t2_case_dir = SEQUENCE_CONFIGS["t2"]["processed_dir"] / class_name / case_name
                for seq_name, cfg in SEQUENCE_CONFIGS.items():
                    processed_dir = cfg["processed_dir"] / class_name / case_name
                    image_series_meta_index = build_case_image_series_index(processed_dir)
                    if not image_series_meta_index:
                        continue

                    image_uid, match_type = select_image_series_uid(
                        series_uid,
                        t2_case_dir if t2_case_dir.exists() else None,
                        image_series_meta_index,
                    )
                    if image_uid is None:
                        continue

                    images_dir = processed_dir / image_uid / "images"
                    if not images_dir.exists():
                        continue

                    indices = list_indices(images_dir)
                    if not indices:
                        continue

                    images_by_seq[seq_name] = {
                        "images_dir": images_dir,
                        "series_uid": image_uid,
                    }
                    indices_by_seq[seq_name] = indices
                    match_types[seq_name] = match_type

                if set(images_by_seq.keys()) != {"t2", "ep2d_calc", "ep2d_adc"}:
                    print(
                        f"  ⚠ Skipping {case_name}/{series_uid[:12]}: "
                        f"incomplete sequences ({', '.join(sorted(images_by_seq.keys()))})"
                    )
                    continue

                case_output_dir = Path("validation_results") / class_name / case_name
                case_output_dir.mkdir(parents=True, exist_ok=True)

                output_path = build_case_grid(
                    case_output_dir,
                    class_name,
                    case_name,
                    series_uid,
                    images_by_seq,
                    indices_by_seq,
                    match_types,
                    args.max_slices,
                )
                if output_path:
                    print(f"  ✓ Saved {output_path.name}")


if __name__ == "__main__":
    main()
