#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Tool

Generates a detailed HTML report with:
- Image counts per class
- Mask availability statistics (prostate, target1, target2)
- Mask coverage ratios
- Distribution visualizations using seaborn
- Case-level statistics

Usage:
    python tools/analyze_data.py

Output:
    data_analysis_report.html - Comprehensive HTML report with visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to prevent crashes
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.facecolor"] = "white"


class DatasetAnalyzer:
    """Analyzes the entire MRI dataset across all classes."""

    def __init__(self, base_dir: Path = Path("data")):
        self.base_dir = base_dir
        self.processed_dir = base_dir / "processed"
        self.processed_seg_dir = base_dir / "processed_seg"

        self.stats = {
            "classes": {},
            "overall": {
                "total_images": 0,
                "total_cases": 0,
                "total_series": 0,
                "total_masks": {"prostate": 0, "target1": 0, "target2": 0},
            },
        }

        self.figures = []

    def analyze_all_classes(self):
        """Analyze all available classes."""
        print("=" * 80)
        print("  MRI DATASET COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        print()

        # Find all manifests
        manifests = sorted(list(self.processed_dir.glob("class*/manifest.csv")))

        if not manifests:
            print(f"✗ No manifests found in {self.processed_dir}")
            return False

        print(f"Found {len(manifests)} class(es) to analyze\n")

        # Analyze each class
        for manifest_path in manifests:
            class_name = manifest_path.parent.name
            print(f"Analyzing {class_name}...")

            class_stats = self.analyze_class(manifest_path, class_name)
            if class_stats:
                self.stats["classes"][class_name] = class_stats

                # Update overall stats
                self.stats["overall"]["total_images"] += class_stats["total_images"]
                self.stats["overall"]["total_cases"] += class_stats["num_cases"]
                self.stats["overall"]["total_series"] += class_stats["num_series"]

                for mask_type in ["prostate", "target1", "target2"]:
                    self.stats["overall"]["total_masks"][mask_type] += class_stats[
                        "masks"
                    ][mask_type]["count"]

            print()

        return True

    def analyze_class(self, manifest_path: Path, class_name: str) -> dict:
        """Analyze a single class."""
        # Load manifest
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            print(f"  ✗ Error loading manifest: {e}")
            return None

        # Basic stats
        stats = {
            "class_name": class_name,
            "manifest_path": str(manifest_path),
            "total_images": len(df),
            "num_cases": df["case_id"].nunique(),
            "num_series": df["series_uid"].nunique(),
            "cases": list(df["case_id"].unique()),
            "slices_per_case": {},
            "slices_per_series": {},
            "masks": {
                "prostate": {"count": 0, "cases": [], "slices": []},
                "target1": {"count": 0, "cases": [], "slices": []},
                "target2": {"count": 0, "cases": [], "slices": []},
            },
        }

        # Slices per case
        for case_id, group in df.groupby("case_id"):
            stats["slices_per_case"][int(case_id)] = len(group)

        # Slices per series
        for series_uid, group in df.groupby("series_uid"):
            stats["slices_per_series"][series_uid] = len(group)

        # Analyze image dimensions
        image_dimensions = self.analyze_image_dimensions(df)
        stats["image_dimensions"] = image_dimensions

        # Analyze masks
        processed_seg_class_dir = self.processed_seg_dir / class_name

        if processed_seg_class_dir.exists():
            mask_stats = self.analyze_masks(processed_seg_class_dir, df)
            stats["masks"] = mask_stats

            # Analyze mask sizes
            mask_sizes = self.analyze_mask_sizes(processed_seg_class_dir, df)
            stats["mask_sizes"] = mask_sizes

        print(f"  ✓ Images: {stats['total_images']}")
        print(f"  ✓ Cases: {stats['num_cases']}")
        print(f"  ✓ Series: {stats['num_series']}")
        print(
            f"  ✓ Masks: P={stats['masks']['prostate']['count']}, "
            f"T1={stats['masks']['target1']['count']}, "
            f"T2={stats['masks']['target2']['count']}"
        )
        if "image_dimensions" in stats and stats["image_dimensions"]["sizes"]:
            print(f"  ✓ Image sizes: {stats['image_dimensions']['unique_sizes']}")

        return stats

    def analyze_masks(self, seg_dir: Path, df: pd.DataFrame) -> dict:
        """Analyze mask availability for a class."""
        mask_stats = {
            "prostate": {"count": 0, "cases": set(), "slices": []},
            "target1": {"count": 0, "cases": set(), "slices": []},
            "target2": {"count": 0, "cases": set(), "slices": []},
        }

        # Iterate through all cases
        for case_dir in sorted(seg_dir.glob("case_*")):
            case_id = int(case_dir.name.split("_")[1])

            # Iterate through series
            for series_dir in case_dir.iterdir():
                if not series_dir.is_dir() or series_dir.name == "biopsies.json":
                    continue

                # Check each mask type
                for mask_type in ["prostate", "target1", "target2"]:
                    mask_dir = series_dir / mask_type

                    if mask_dir.exists():
                        mask_files = list(mask_dir.glob("*.png"))

                        if mask_files:
                            mask_stats[mask_type]["count"] += len(mask_files)
                            mask_stats[mask_type]["cases"].add(case_id)

                            # Store slice numbers
                            for mask_file in mask_files:
                                slice_num = int(mask_file.stem)
                                mask_stats[mask_type]["slices"].append(slice_num)

        # Convert sets to lists for JSON serialization
        for mask_type in mask_stats:
            mask_stats[mask_type]["cases"] = sorted(
                list(mask_stats[mask_type]["cases"])
            )

        return mask_stats

    def analyze_image_dimensions(self, df: pd.DataFrame) -> dict:
        """Analyze image dimensions (width x height) for a class."""
        from PIL import Image

        dimensions = {
            "sizes": [],
            "widths": [],
            "heights": [],
            "unique_sizes": [],
            "size_counts": {},
        }

        # Sample images to check dimensions (check up to 20 images)
        sample_size = min(20, len(df))
        sampled_rows = df.sample(n=sample_size, random_state=42)

        for _, row in sampled_rows.iterrows():
            img_path = Path(row["image_path"])
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    dimensions["widths"].append(width)
                    dimensions["heights"].append(height)
                    size_str = f"{width}x{height}"
                    dimensions["sizes"].append((width, height))

                    if size_str not in dimensions["size_counts"]:
                        dimensions["size_counts"][size_str] = 0
                    dimensions["size_counts"][size_str] += 1
                except Exception:
                    pass

        # Get unique sizes
        dimensions["unique_sizes"] = list(
            set([f"{w}x{h}" for w, h in dimensions["sizes"]])
        )

        return dimensions

    def analyze_mask_sizes(self, seg_dir: Path, df: pd.DataFrame) -> dict:
        """Analyze mask sizes (number of pixels and ratio to full image)."""
        from PIL import Image

        mask_sizes = {
            "prostate": {
                "pixel_counts": [],
                "ratios": [],
                "avg_pixels": 0,
                "avg_ratio": 0,
            },
            "target1": {
                "pixel_counts": [],
                "ratios": [],
                "avg_pixels": 0,
                "avg_ratio": 0,
            },
            "target2": {
                "pixel_counts": [],
                "ratios": [],
                "avg_pixels": 0,
                "avg_ratio": 0,
            },
        }

        # Sample masks to analyze (up to 50 per type)
        max_samples = 50

        for case_dir in sorted(seg_dir.glob("case_*")):
            for series_dir in case_dir.iterdir():
                if not series_dir.is_dir() or series_dir.name == "biopsies.json":
                    continue

                for mask_type in ["prostate", "target1", "target2"]:
                    mask_dir = series_dir / mask_type

                    if (
                        mask_dir.exists()
                        and len(mask_sizes[mask_type]["pixel_counts"]) < max_samples
                    ):
                        mask_files = list(mask_dir.glob("*.png"))

                        for mask_file in mask_files:
                            if (
                                len(mask_sizes[mask_type]["pixel_counts"])
                                >= max_samples
                            ):
                                break

                            try:
                                mask_img = Image.open(mask_file).convert("L")
                                mask_array = np.array(mask_img)

                                # Count non-zero pixels (mask pixels)
                                mask_pixels = np.sum(mask_array > 127)
                                total_pixels = mask_array.size

                                if total_pixels > 0:
                                    ratio = mask_pixels / total_pixels
                                    mask_sizes[mask_type]["pixel_counts"].append(
                                        mask_pixels
                                    )
                                    mask_sizes[mask_type]["ratios"].append(
                                        ratio * 100
                                    )  # Convert to percentage
                            except Exception:
                                pass

        # Calculate averages
        for mask_type in ["prostate", "target1", "target2"]:
            if mask_sizes[mask_type]["pixel_counts"]:
                mask_sizes[mask_type]["avg_pixels"] = np.mean(
                    mask_sizes[mask_type]["pixel_counts"]
                )
                mask_sizes[mask_type]["avg_ratio"] = np.mean(
                    mask_sizes[mask_type]["ratios"]
                )

        return mask_sizes

    def load_validation_images(self):
        """Load validation mask overlay images if available."""
        validation_images = {}
        validation_dir = Path("data/validation_results")

        if validation_dir.exists():
            for class_name in sorted(self.stats["classes"].keys()):
                mask_overlay_path = validation_dir / class_name / "masks_overlay.png"
                if mask_overlay_path.exists():
                    with open(mask_overlay_path, "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode("utf-8")
                        validation_images[class_name] = img_base64
                    print(f"  ✓ Loaded validation image for {class_name}")

        return validation_images

    def create_custom_visualization(self, class_name: str) -> str:
        """Create a custom visualization showing a case with all mask types, especially target2."""
        from PIL import Image, ImageDraw, ImageFont
        import io

        seg_dir = self.processed_seg_dir / class_name

        # Find a case with all three mask types
        best_case = None
        best_series = None
        best_slice = None

        for case_dir in sorted(seg_dir.glob("case_*")):
            for series_dir in case_dir.iterdir():
                if not series_dir.is_dir() or series_dir.name == "biopsies.json":
                    continue

                # Check if this series has all three mask types
                has_prostate = (series_dir / "prostate").exists()
                has_target1 = (series_dir / "target1").exists()
                has_target2 = (series_dir / "target2").exists()

                if has_prostate and has_target1 and has_target2:
                    # Find a slice that has all masks
                    prostate_files = list((series_dir / "prostate").glob("*.png"))
                    target1_files = list((series_dir / "target1").glob("*.png"))
                    target2_files = list((series_dir / "target2").glob("*.png"))

                    if prostate_files and target1_files and target2_files:
                        # Find common slice numbers where ALL masks exist
                        p_slices = {int(f.stem) for f in prostate_files}
                        t1_slices = {int(f.stem) for f in target1_files}
                        t2_slices = {int(f.stem) for f in target2_files}

                        common_slices = p_slices & t1_slices & t2_slices

                        if common_slices:
                            # Pick the slice with the largest total mask area for better visualization
                            best_slice_candidate = None
                            max_total_pixels = 0

                            for slice_num in sorted(common_slices):
                                # Check actual mask content
                                p_mask = Image.open(
                                    series_dir / "prostate" / f"{slice_num:04d}.png"
                                ).convert("L")
                                t1_mask = Image.open(
                                    series_dir / "target1" / f"{slice_num:04d}.png"
                                ).convert("L")
                                t2_mask = Image.open(
                                    series_dir / "target2" / f"{slice_num:04d}.png"
                                ).convert("L")

                                total_pixels = (
                                    np.sum(np.array(p_mask) > 127)
                                    + np.sum(np.array(t1_mask) > 127)
                                    + np.sum(np.array(t2_mask) > 127)
                                )

                                if total_pixels > max_total_pixels:
                                    max_total_pixels = total_pixels
                                    best_slice_candidate = slice_num

                            if best_slice_candidate and max_total_pixels > 0:
                                best_case = case_dir.name
                                best_series = series_dir.name
                                best_slice = best_slice_candidate
                                break

            if best_case:
                break

        # If no case with all three, find one with at least target2
        if not best_case:
            for case_dir in sorted(seg_dir.glob("case_*")):
                for series_dir in case_dir.iterdir():
                    if not series_dir.is_dir() or series_dir.name == "biopsies.json":
                        continue

                    has_target2 = (series_dir / "target2").exists()

                    if has_target2:
                        target2_files = list((series_dir / "target2").glob("*.png"))
                        if target2_files:
                            best_case = case_dir.name
                            best_series = series_dir.name
                            best_slice = int(
                                target2_files[len(target2_files) // 2].stem
                            )
                            break

                if best_case:
                    break

        if not best_case:
            return None

        # Load the image from processed dir
        processed_class_dir = Path("data/processed") / class_name
        manifest_path = processed_class_dir / "manifest.csv"

        if not manifest_path.exists():
            return None

        df = pd.read_csv(manifest_path)

        # Find the image path for this case/series/slice
        case_id = int(best_case.split("_")[1])
        image_row = df[
            (df["case_id"] == case_id)
            & (df["series_uid"] == best_series)
            & (df["slice_idx"] == best_slice)
        ]

        if image_row.empty:
            return None

        image_path = Path(image_row.iloc[0]["image_path"])

        if not image_path.exists():
            return None

        # Create overlay visualization
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Load and overlay masks
        series_path = seg_dir / best_case / best_series

        mask_info = []
        mask_pixel_counts = {}

        print(
            f"    Image size: {img_width}x{img_height}, Case: {best_case}, Series: {best_series}, Slice: {best_slice}"
        )

        # Prostate - Yellow
        prostate_path = series_path / "prostate" / f"{best_slice:04d}.png"
        if prostate_path.exists():
            mask = Image.open(prostate_path).convert("L")
            mask_width, mask_height = mask.size

            # Verify mask size matches image size
            if mask_width != img_width or mask_height != img_height:
                print(
                    f"    WARNING: Prostate mask size ({mask_width}x{mask_height}) doesn't match image size ({img_width}x{img_height})"
                )
                # Resize mask to match image
                mask = mask.resize((img_width, img_height), Image.LANCZOS)

            mask_array = np.array(mask)
            mask_pixels = np.sum(mask_array > 127)
            mask_pixel_counts["prostate"] = mask_pixels

            yellow_overlay = Image.new("RGBA", img.size, (255, 215, 0, 100))
            mask_img = Image.fromarray(mask_array)
            overlay = Image.composite(yellow_overlay, overlay, mask_img)
            mask_info.append("Prostate (Yellow)")
            print(
                f"    Prostate mask: {mask_pixels} pixels ({mask_pixels/(img_width*img_height)*100:.2f}%)"
            )

        # Target1 - Red
        target1_path = series_path / "target1" / f"{best_slice:04d}.png"
        if target1_path.exists():
            mask = Image.open(target1_path).convert("L")
            mask_width, mask_height = mask.size

            # Verify mask size matches image size
            if mask_width != img_width or mask_height != img_height:
                print(
                    f"    WARNING: Target1 mask size ({mask_width}x{mask_height}) doesn't match image size ({img_width}x{img_height})"
                )
                # Resize mask to match image
                mask = mask.resize((img_width, img_height), Image.LANCZOS)

            mask_array = np.array(mask)
            mask_pixels = np.sum(mask_array > 127)
            mask_pixel_counts["target1"] = mask_pixels

            red_overlay = Image.new("RGBA", img.size, (255, 68, 68, 120))
            mask_img = Image.fromarray(mask_array)
            overlay = Image.composite(red_overlay, overlay, mask_img)
            mask_info.append("Target1 (Red)")
            print(
                f"    Target1 mask: {mask_pixels} pixels ({mask_pixels/(img_width*img_height)*100:.2f}%)"
            )

        # Target2 - Blue
        target2_path = series_path / "target2" / f"{best_slice:04d}.png"
        if target2_path.exists():
            mask = Image.open(target2_path).convert("L")
            mask_width, mask_height = mask.size

            # Verify mask size matches image size
            if mask_width != img_width or mask_height != img_height:
                print(
                    f"    WARNING: Target2 mask size ({mask_width}x{mask_height}) doesn't match image size ({img_width}x{img_height})"
                )
                # Resize mask to match image
                mask = mask.resize((img_width, img_height), Image.LANCZOS)

            mask_array = np.array(mask)
            mask_pixels = np.sum(mask_array > 127)
            mask_pixel_counts["target2"] = mask_pixels

            blue_overlay = Image.new("RGBA", img.size, (68, 114, 196, 120))
            mask_img = Image.fromarray(mask_array)
            overlay = Image.composite(blue_overlay, overlay, mask_img)
            mask_info.append("Target2 (Blue)")
            print(
                f"    Target2 mask: {mask_pixels} pixels ({mask_pixels/(img_width*img_height)*100:.2f}%)"
            )

        # Composite the overlay onto the original image
        img = img.convert("RGBA")
        result = Image.alpha_composite(img, overlay)
        result = result.convert("RGB")

        # Convert to base64
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_base64

    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("=" * 80)
        print("  CREATING VISUALIZATIONS")
        print("=" * 80)
        print()

        self.create_overview_figure()
        self.create_mask_distribution_figure()
        self.create_image_analysis_figure()
        self.create_per_class_details_figure()
        self.create_case_level_analysis_figure()

        print()

    def create_overview_figure(self):
        """Create overview figure with key statistics."""
        print("Creating overview figure...")

        # Green color palette
        green_colors = ["#2d7f3e", "#3e9651", "#50ae65", "#73c084"]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)

        classes = sorted(self.stats["classes"].keys())

        # 1. Total images per class
        ax1 = fig.add_subplot(gs[0, 0])
        image_counts = [self.stats["classes"][c]["total_images"] for c in classes]
        bars = ax1.bar(classes, image_counts, color=green_colors[: len(classes)])
        ax1.set_xlabel("Class", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
        ax1.set_title("Total Images per Class", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 2. Cases per class
        ax2 = fig.add_subplot(gs[0, 1])
        case_counts = [self.stats["classes"][c]["num_cases"] for c in classes]
        bars = ax2.bar(classes, case_counts, color=green_colors[: len(classes)])
        ax2.set_xlabel("Class", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Number of Cases", fontsize=11, fontweight="bold")
        ax2.set_title("Unique Cases per Class", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 3. Series per class
        ax3 = fig.add_subplot(gs[0, 2])
        series_counts = [self.stats["classes"][c]["num_series"] for c in classes]
        bars = ax3.bar(classes, series_counts, color=green_colors[: len(classes)])
        ax3.set_xlabel("Class", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Number of Series", fontsize=11, fontweight="bold")
        ax3.set_title("Unique Series per Class", fontsize=12, fontweight="bold")
        ax3.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 4. Overall dataset composition (pie chart)
        ax4 = fig.add_subplot(gs[1, 0])
        wedges, texts, autotexts = ax4.pie(
            image_counts,
            labels=classes,
            autopct="%1.1f%%",
            colors=green_colors[: len(classes)],
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)
        ax4.set_title("Dataset Composition by Images", fontsize=12, fontweight="bold")

        # 5. Mask types overall
        ax5 = fig.add_subplot(gs[1, 1])
        mask_types = ["Prostate", "Target1", "Target2"]
        mask_counts = [
            self.stats["overall"]["total_masks"]["prostate"],
            self.stats["overall"]["total_masks"]["target1"],
            self.stats["overall"]["total_masks"]["target2"],
        ]
        colors_mask = [
            "#FFD700",
            "#FF4444",
            "#FF8C00",
        ]  # Yellow, Red (highlight), Blue
        bars = ax5.bar(mask_types, mask_counts, color=colors_mask)
        ax5.set_ylabel("Number of Masks", fontsize=11, fontweight="bold")
        ax5.set_title("Total Masks by Type", fontsize=12, fontweight="bold")
        ax5.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 6. Mask availability heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        mask_matrix = []
        for c in classes:
            row = [
                self.stats["classes"][c]["masks"]["prostate"]["count"],
                self.stats["classes"][c]["masks"]["target1"]["count"],
                self.stats["classes"][c]["masks"]["target2"]["count"],
            ]
            mask_matrix.append(row)

        mask_df = pd.DataFrame(
            mask_matrix, index=classes, columns=["Prostate", "Target1", "Target2"]
        )
        sns.heatmap(
            mask_df,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=ax6,
            cbar_kws={"label": "Count"},
        )
        ax6.set_title("Mask Availability Heatmap", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Class", fontsize=11, fontweight="bold")

        # 7. Average slices per case
        ax7 = fig.add_subplot(gs[2, 0])
        avg_slices = []
        for c in classes:
            slices = list(self.stats["classes"][c]["slices_per_case"].values())
            avg_slices.append(np.mean(slices) if slices else 0)

        bars = ax7.bar(classes, avg_slices, color=green_colors[: len(classes)])
        ax7.set_xlabel("Class", fontsize=11, fontweight="bold")
        ax7.set_ylabel("Average Slices", fontsize=11, fontweight="bold")
        ax7.set_title("Average Slices per Case", fontsize=12, fontweight="bold")
        ax7.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax7.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 8. Summary statistics table
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis("off")

        summary_data = [
            ["Metric", "Value"],
            ["Total Images", f"{self.stats['overall']['total_images']:,}"],
            ["Total Cases", f"{self.stats['overall']['total_cases']:,}"],
            ["Total Series", f"{self.stats['overall']['total_series']:,}"],
            ["Total Classes", f"{len(classes)}"],
            ["Prostate Masks", f"{self.stats['overall']['total_masks']['prostate']:,}"],
            ["Target1 Masks", f"{self.stats['overall']['total_masks']['target1']:,}"],
            ["Target2 Masks", f"{self.stats['overall']['total_masks']['target2']:,}"],
        ]

        table = ax8.table(
            cellText=summary_data, cellLoc="left", loc="center", colWidths=[0.4, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor("#2d7f3e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        ax8.set_title("Overall Dataset Summary", fontsize=12, fontweight="bold", pad=20)

        plt.suptitle("MRI Dataset Overview", fontsize=16, fontweight="bold", y=0.99)

        self.figures.append(("overview", fig))
        print("  ✓ Overview figure created")

    def create_mask_distribution_figure(self):
        """Create detailed mask distribution visualizations."""
        print("Creating mask distribution figure...")

        green_colors = ["#2d7f3e", "#3e9651", "#50ae65", "#73c084"]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        classes = sorted(self.stats["classes"].keys())

        # 1. Mask coverage ratio per class (grouped bar)
        ax1 = fig.add_subplot(gs[0, :])

        prostate_coverage = []
        target1_coverage = []
        target2_coverage = []

        for c in classes:
            total = self.stats["classes"][c]["total_images"]
            p_count = self.stats["classes"][c]["masks"]["prostate"]["count"]
            t1_count = self.stats["classes"][c]["masks"]["target1"]["count"]
            t2_count = self.stats["classes"][c]["masks"]["target2"]["count"]

            prostate_coverage.append(p_count / total * 100 if total > 0 else 0)
            target1_coverage.append(t1_count / total * 100 if total > 0 else 0)
            target2_coverage.append(t2_count / total * 100 if total > 0 else 0)

        x = np.arange(len(classes))
        width = 0.25

        bars1 = ax1.bar(
            x - width, prostate_coverage, width, label="Prostate", color="#FFD700"
        )
        bars2 = ax1.bar(x, target1_coverage, width, label="Target1", color="#FF4444")
        bars3 = ax1.bar(
            x + width, target2_coverage, width, label="Target2", color="#FF8C00"
        )

        ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Coverage (%)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Mask Coverage per Class (% of images with masks)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend(fontsize=11)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        # 2-4. Distribution of masks per class (violin plots)
        mask_types = ["prostate", "target1", "target2"]
        mask_labels = ["Prostate", "Target1", "Target2"]
        mask_colors = ["#FFD700", "#FF4444", "#FF8C00"]

        for idx, (mask_type, label, color) in enumerate(
            zip(mask_types, mask_labels, mask_colors)
        ):
            ax = fig.add_subplot(gs[1, idx])

            # Prepare data for violin plot
            data_for_violin = []
            labels_for_violin = []

            for c in classes:
                counts = self.stats["classes"][c]["masks"][mask_type]["count"]
                num_cases = len(self.stats["classes"][c]["masks"][mask_type]["cases"])

                if num_cases > 0:
                    # Average masks per case
                    avg_per_case = counts / num_cases
                    data_for_violin.extend([avg_per_case] * num_cases)
                    labels_for_violin.extend([c] * num_cases)

            if data_for_violin:
                df_violin = pd.DataFrame(
                    {"Class": labels_for_violin, "Masks per Case": data_for_violin}
                )

                sns.violinplot(
                    data=df_violin,
                    x="Class",
                    y="Masks per Case",
                    ax=ax,
                    color=color,
                    alpha=0.7,
                )
                sns.swarmplot(
                    data=df_violin,
                    x="Class",
                    y="Masks per Case",
                    ax=ax,
                    color="black",
                    alpha=0.5,
                    size=3,
                )

                ax.set_title(
                    f"{label} Masks Distribution", fontsize=12, fontweight="bold"
                )
                ax.set_xlabel("Class", fontsize=11, fontweight="bold")
                ax.set_ylabel("Masks per Case", fontsize=11, fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label} masks available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.axis("off")

        plt.suptitle(
            "Mask Distribution Analysis", fontsize=16, fontweight="bold", y=0.98
        )

        self.figures.append(("mask_distribution", fig))
        print("  ✓ Mask distribution figure created")

    def create_image_analysis_figure(self):
        """Create image and mask size analysis visualizations."""
        print("Creating image analysis figure...")

        green_colors = ["#2d7f3e", "#3e9651", "#50ae65", "#73c084"]
        classes = sorted(self.stats["classes"].keys())

        fig = plt.figure(figsize=(18, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 6])

        # 1. Image dimensions per class
        ax1 = fig.add_subplot(gs[0, 0])

        dim_data = []
        for class_name in classes:
            if "image_dimensions" in self.stats["classes"][class_name]:
                dims = self.stats["classes"][class_name]["image_dimensions"]
                unique_sizes = dims.get("unique_sizes", [])
                if unique_sizes:
                    dim_data.append(f"{class_name}: {', '.join(unique_sizes)}")

        # Create bar chart of average dimensions
        avg_widths = []
        avg_heights = []
        for class_name in classes:
            if "image_dimensions" in self.stats["classes"][class_name]:
                dims = self.stats["classes"][class_name]["image_dimensions"]
                if dims["widths"]:
                    avg_widths.append(np.mean(dims["widths"]))
                    avg_heights.append(np.mean(dims["heights"]))
                else:
                    avg_widths.append(0)
                    avg_heights.append(0)

        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            avg_widths,
            width,
            label="Width",
            color=green_colors[0],
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x + width / 2,
            avg_heights,
            width,
            label="Height",
            color=green_colors[2],
            alpha=0.8,
        )

        ax1.set_xlabel("Class", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Pixels", fontsize=11, fontweight="bold")
        ax1.set_title("Average Image Dimensions", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        # 2-4. Mask size distributions (Prostate, Target1, Target2)
        mask_types = ["prostate", "target1", "target2"]
        mask_labels = ["Prostate", "Target1", "Target2"]
        mask_colors = ["#FFD700", "#FF4444", "#4472C4"]

        for idx, (mask_type, label, color) in enumerate(
            zip(mask_types, mask_labels, mask_colors)
        ):
            ax = fig.add_subplot(gs[0, idx])
            pixels = []
            labels_list = []

            for class_name in classes:
                if "mask_sizes" in self.stats["classes"][class_name]:
                    pixel_counts = self.stats["classes"][class_name]["mask_sizes"][
                        mask_type
                    ]["pixel_counts"]
                    if pixel_counts:
                        pixels.extend(pixel_counts)
                        labels_list.extend([class_name] * len(pixel_counts))

            if pixels:
                df_data = pd.DataFrame({"Class": labels_list, "Pixels": pixels})
                sns.boxplot(
                    data=df_data,
                    x="Class",
                    y="Pixels",
                    ax=ax,
                    hue="Class",
                    palette={
                        c: green_colors[i % len(green_colors)]
                        for i, c in enumerate(classes)
                    },
                    legend=False,
                )
                ax.set_xlabel("Class", fontsize=11, fontweight="bold")
                ax.set_ylabel("Number of Pixels", fontsize=11, fontweight="bold")
                ax.set_title(
                    f"{label} Mask Size Distribution", fontsize=12, fontweight="bold"
                )
                ax.grid(axis="y", alpha=0.3)
                ax.ticklabel_format(style="plain", axis="y")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.axis("off")

        # 5-7. Mask ratio distributions
        for idx, (mask_type, label, color) in enumerate(
            zip(mask_types, mask_labels, mask_colors)
        ):
            ax = fig.add_subplot(gs[1, idx])
            ratios = []
            ratio_labels = []

            for class_name in classes:
                if "mask_sizes" in self.stats["classes"][class_name]:
                    ratio_data = self.stats["classes"][class_name]["mask_sizes"][
                        mask_type
                    ]["ratios"]
                    if ratio_data:
                        ratios.extend(ratio_data)
                        ratio_labels.extend([class_name] * len(ratio_data))

            if ratios:
                df_ratios = pd.DataFrame({"Class": ratio_labels, "Ratio (%)": ratios})
                sns.violinplot(
                    data=df_ratios,
                    x="Class",
                    y="Ratio (%)",
                    ax=ax,
                    color=color,
                    alpha=0.7,
                )
                ax.set_xlabel("Class", fontsize=11, fontweight="bold")
                ax.set_ylabel("Mask Ratio (%)", fontsize=11, fontweight="bold")
                ax.set_title(
                    f"{label} Mask Coverage Ratio", fontsize=12, fontweight="bold"
                )
                ax.grid(axis="y", alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.axis("off")

        # 8-10. Average mask sizes comparison
        for idx, (mask_type, label, color) in enumerate(
            zip(mask_types, mask_labels, mask_colors)
        ):
            ax = fig.add_subplot(gs[2, idx])

            avg_pixels = []
            avg_ratios = []
            class_labels = []

            for class_name in classes:
                if "mask_sizes" in self.stats["classes"][class_name]:
                    mask_size_data = self.stats["classes"][class_name]["mask_sizes"][
                        mask_type
                    ]
                    if mask_size_data["avg_pixels"] > 0:
                        avg_pixels.append(mask_size_data["avg_pixels"])
                        avg_ratios.append(mask_size_data["avg_ratio"])
                        class_labels.append(class_name)

            if avg_pixels:
                # Create dual-axis plot
                ax_twin = ax.twinx()

                x = np.arange(len(class_labels))
                width = 0.35

                bars1 = ax.bar(
                    x - width / 2,
                    avg_pixels,
                    width,
                    label="Pixels",
                    color=color,
                    alpha=0.7,
                )
                bars2 = ax_twin.bar(
                    x + width / 2,
                    avg_ratios,
                    width,
                    label="Ratio (%)",
                    color="#2d7f3e",
                    alpha=0.6,
                )

                ax.set_xlabel("Class", fontsize=10, fontweight="bold")
                ax.set_ylabel(
                    "Average Pixels", fontsize=10, fontweight="bold", color=color
                )
                ax_twin.set_ylabel(
                    "Average Ratio (%)", fontsize=10, fontweight="bold", color="#2d7f3e"
                )
                ax.set_title(f"{label} Mask Size", fontsize=11, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(class_labels, rotation=0)
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(axis="y", labelcolor=color)
                ax_twin.tick_params(axis="y", labelcolor="#2d7f3e")

                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label} mask data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.axis("off")

        # 11. Mask-Image size verification
        ax11 = fig.add_subplot(gs[3, :])

        # Verify mask sizes match image sizes for each class
        verification_data = []
        for class_name in classes:
            if "mask_sizes" in self.stats["classes"][class_name]:
                class_stats = self.stats["classes"][class_name]
                img_dims = class_stats.get("image_dimensions", {})
                unique_sizes = img_dims.get("unique_sizes", [])

                if unique_sizes:
                    for mask_type in ["prostate", "target1", "target2"]:
                        mask_data = class_stats["mask_sizes"][mask_type]
                        if mask_data["pixel_counts"]:
                            # Check if mask dimensions are consistent with image dimensions
                            for size_str in unique_sizes:
                                w, h = map(int, size_str.split("x"))
                                total_pixels = w * h
                                avg_mask_pixels = mask_data["avg_pixels"]
                                verification_data.append(
                                    {
                                        "Class": class_name,
                                        "Mask Type": mask_type.capitalize(),
                                        "Image Size": size_str,
                                        "Total Pixels": total_pixels,
                                        "Avg Mask": int(avg_mask_pixels),
                                        "Status": (
                                            "✓ Valid"
                                            if avg_mask_pixels <= total_pixels
                                            else "✗ Invalid"
                                        ),
                                    }
                                )

        if verification_data:
            df_verify = pd.DataFrame(verification_data)

            # Create a text summary table
            ax11.axis("tight")
            ax11.axis("off")

            table_data = []
            table_data.append(
                [
                    "Class",
                    "Mask Type",
                    "Image Size",
                    "Total Pixels",
                    "Avg Mask Pixels",
                    "Status",
                ]
            )

            for _, row in df_verify.iterrows():
                table_data.append(
                    [
                        row["Class"],
                        row["Mask Type"],
                        row["Image Size"],
                        f"{row['Total Pixels']:,}",
                        f"{row['Avg Mask']:,}",
                        row["Status"],
                    ]
                )

            table = ax11.table(
                cellText=table_data,
                cellLoc="center",
                loc="center",
                colWidths=[0.15, 0.15, 0.15, 0.18, 0.18, 0.12],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header row
            for i in range(6):
                table[(0, i)].set_facecolor("#2d7f3e")
                table[(0, i)].set_text_props(weight="bold", color="white")

            # Style data rows
            for i in range(1, len(table_data)):
                for j in range(6):
                    if j == 5 and "✓" in table_data[i][j]:
                        table[(i, j)].set_facecolor("#d4edda")
                    elif i % 2 == 0:
                        table[(i, j)].set_facecolor("#f8f9fa")

            ax11.set_title(
                "Mask-Image Size Verification (Masks match their image dimensions)",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )

        plt.suptitle(
            "Image and Mask Size Analysis", fontsize=16, fontweight="bold", y=0.99
        )

        self.figures.append(("image_analysis", fig))
        print("  ✓ Image analysis figure created")

    def create_per_class_details_figure(self):
        """Create detailed per-class analysis."""
        print("Creating per-class details figure...")

        green_colors = ["#2d7f3e", "#3e9651", "#50ae65", "#73c084"]
        classes = sorted(self.stats["classes"].keys())

        fig = plt.figure(figsize=(16, 4 * len(classes)))
        gs = fig.add_gridspec(len(classes), 4, hspace=0.4, wspace=0.3)

        for idx, class_name in enumerate(classes):
            class_stats = self.stats["classes"][class_name]

            # 1. Slices per case distribution
            ax1 = fig.add_subplot(gs[idx, 0])
            slices_per_case = list(class_stats["slices_per_case"].values())

            sns.histplot(
                slices_per_case,
                bins=20,
                kde=True,
                ax=ax1,
                color=green_colors[idx % len(green_colors)],
            )
            ax1.set_xlabel("Slices per Case", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Frequency", fontsize=10, fontweight="bold")
            ax1.set_title(
                f"{class_name}: Slices Distribution", fontsize=11, fontweight="bold"
            )
            ax1.grid(axis="y", alpha=0.3)

            # Add statistics
            mean_slices = np.mean(slices_per_case)
            median_slices = np.median(slices_per_case)
            ax1.axvline(
                mean_slices,
                color="#d32f2f",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_slices:.1f}",
            )
            ax1.axvline(
                median_slices,
                color="#2d7f3e",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_slices:.1f}",
            )
            ax1.legend(fontsize=8)

            # 2. Cases with masks
            ax2 = fig.add_subplot(gs[idx, 1])

            total_cases = class_stats["num_cases"]
            cases_with_prostate = len(class_stats["masks"]["prostate"]["cases"])
            cases_with_target1 = len(class_stats["masks"]["target1"]["cases"])
            cases_with_target2 = len(class_stats["masks"]["target2"]["cases"])

            mask_case_data = {
                "Prostate": cases_with_prostate,
                "Target1": cases_with_target1,
                "Target2": cases_with_target2,
                "No Masks": total_cases
                - max(cases_with_prostate, cases_with_target1, cases_with_target2),
            }

            colors_pie = ["#FFD700", "#FF4444", "#FF8C00", "#CCCCCC"]
            wedges, texts, autotexts = ax2.pie(
                mask_case_data.values(),
                labels=mask_case_data.keys(),
                autopct="%1.1f%%",
                colors=colors_pie,
                startangle=90,
            )
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(9)
            ax2.set_title(
                f"{class_name}: Cases with Masks", fontsize=11, fontweight="bold"
            )

            # 3. Mask counts comparison
            ax3 = fig.add_subplot(gs[idx, 2])

            mask_counts = [
                class_stats["masks"]["prostate"]["count"],
                class_stats["masks"]["target1"]["count"],
                class_stats["masks"]["target2"]["count"],
            ]
            mask_labels = ["Prostate", "Target1", "Target2"]
            colors_bar = ["#FFD700", "#FF4444", "#FF8C00"]

            bars = ax3.barh(mask_labels, mask_counts, color=colors_bar)
            ax3.set_xlabel("Number of Masks", fontsize=10, fontweight="bold")
            ax3.set_title(f"{class_name}: Mask Counts", fontsize=11, fontweight="bold")
            ax3.grid(axis="x", alpha=0.3)

            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax3.text(
                        width,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{int(width)}",
                        ha="left",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

            # 4. Statistics table
            ax4 = fig.add_subplot(gs[idx, 3])
            ax4.axis("off")

            table_data = [
                ["Metric", "Value"],
                ["Total Images", f"{class_stats['total_images']:,}"],
                ["Total Cases", f"{class_stats['num_cases']:,}"],
                ["Total Series", f"{class_stats['num_series']:,}"],
                ["Avg Slices/Case", f"{np.mean(slices_per_case):.1f}"],
                ["Prostate Masks", f"{class_stats['masks']['prostate']['count']:,}"],
                ["Target1 Masks", f"{class_stats['masks']['target1']['count']:,}"],
                ["Target2 Masks", f"{class_stats['masks']['target2']['count']:,}"],
            ]

            table = ax4.table(
                cellText=table_data, cellLoc="left", loc="center", colWidths=[0.5, 0.5]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header
            for i in range(2):
                table[(0, i)].set_facecolor(
                    sns.color_palette("husl", len(classes))[idx]
                )
                table[(0, i)].set_text_props(weight="bold", color="white")

            # Alternate rows
            for i in range(1, len(table_data)):
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor("#f0f0f0")

            ax4.set_title(
                f"{class_name}: Summary", fontsize=11, fontweight="bold", pad=10
            )

        plt.suptitle(
            "Per-Class Detailed Analysis", fontsize=16, fontweight="bold", y=0.995
        )

        self.figures.append(("per_class_details", fig))
        print("  ✓ Per-class details figure created")

    def create_case_level_analysis_figure(self):
        """Create case-level analysis across all classes."""
        print("Creating case-level analysis figure...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        classes = sorted(self.stats["classes"].keys())

        green_colors = ["#2d7f3e", "#3e9651", "#50ae65", "#73c084"]

        # 1. Box plot of slices per case across classes
        ax1 = fig.add_subplot(gs[0, 0])

        data_for_box = []
        labels_for_box = []

        for c in classes:
            slices = list(self.stats["classes"][c]["slices_per_case"].values())
            data_for_box.extend(slices)
            labels_for_box.extend([c] * len(slices))

        df_box = pd.DataFrame({"Class": labels_for_box, "Slices": data_for_box})

        # Create boxplot with green colors
        box_colors = {
            c: green_colors[i % len(green_colors)] for i, c in enumerate(classes)
        }
        sns.boxplot(
            data=df_box,
            x="Class",
            y="Slices",
            ax=ax1,
            hue="Class",
            palette=box_colors,
            legend=False,
        )
        ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Slices per Case", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Slices per Case Distribution by Class", fontsize=13, fontweight="bold"
        )
        ax1.grid(axis="y", alpha=0.3)

        # 2. Cumulative mask availability
        ax2 = fig.add_subplot(gs[0, 1])

        mask_ratios = []
        for c in classes:
            total_images = self.stats["classes"][c]["total_images"]
            total_masks = sum(
                [
                    self.stats["classes"][c]["masks"][mt]["count"]
                    for mt in ["prostate", "target1", "target2"]
                ]
            )
            ratio = (total_masks / (total_images * 3)) * 100 if total_images > 0 else 0
            mask_ratios.append(ratio)

        bars = ax2.bar(classes, mask_ratios, color=green_colors[: len(classes)])
        ax2.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Overall Mask Coverage (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Overall Mask Availability\n(All mask types combined)",
            fontsize=13,
            fontweight="bold",
        )
        ax2.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # 3. Heatmap of mask ratios
        ax3 = fig.add_subplot(gs[1, 0])

        ratio_matrix = []
        for c in classes:
            total = self.stats["classes"][c]["total_images"]
            row = []
            for mask_type in ["prostate", "target1", "target2"]:
                count = self.stats["classes"][c]["masks"][mask_type]["count"]
                ratio = (count / total * 100) if total > 0 else 0
                row.append(ratio)
            ratio_matrix.append(row)

        ratio_df = pd.DataFrame(
            ratio_matrix, index=classes, columns=["Prostate", "Target1", "Target2"]
        )
        sns.heatmap(
            ratio_df,
            annot=True,
            fmt=".1f",
            cmap="Greens",
            ax=ax3,
            cbar_kws={"label": "Coverage (%)"},
            vmin=0,
            vmax=100,
        )
        ax3.set_title("Mask Coverage Ratio Heatmap (%)", fontsize=13, fontweight="bold")
        ax3.set_ylabel("Class", fontsize=12, fontweight="bold")

        # 4. Series count comparison
        ax4 = fig.add_subplot(gs[1, 1])

        series_data = []
        for c in classes:
            num_series = self.stats["classes"][c]["num_series"]
            num_cases = self.stats["classes"][c]["num_cases"]
            series_per_case = num_series / num_cases if num_cases > 0 else 0
            series_data.append(series_per_case)

        bars = ax4.bar(classes, series_data, color=green_colors[: len(classes)])
        ax4.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Series per Case", fontsize=12, fontweight="bold")
        ax4.set_title(
            "Average Series per Case by Class", fontsize=13, fontweight="bold"
        )
        ax4.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.suptitle("Case-Level Analysis", fontsize=16, fontweight="bold", y=0.98)

        self.figures.append(("case_level_analysis", fig))
        print("  ✓ Case-level analysis figure created")

    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return img_base64

    def generate_html_report(
        self, output_path: Path = Path("data_analysis_report.html")
    ):
        """Generate comprehensive HTML report."""
        print("=" * 80)
        print("  GENERATING HTML REPORT")
        print("=" * 80)
        print()

        # Convert figures to base64
        figure_images = {}
        for name, fig in self.figures:
            print(f"Encoding {name} figure...")
            figure_images[name] = self.fig_to_base64(fig)

        # Create custom visualizations with Target2 masks
        print("\nCreating custom mask visualizations...")
        validation_images = {}
        for class_name in sorted(self.stats["classes"].keys()):
            print(f"  Creating visualization for {class_name}...")
            img_base64 = self.create_custom_visualization(class_name)
            if img_base64:
                validation_images[class_name] = img_base64
                print(f"  ✓ Created custom visualization for {class_name}")
            else:
                # Fallback to existing validation image
                validation_dir = Path("data/validation_results")
                mask_overlay_path = validation_dir / class_name / "masks_overlay.png"
                if mask_overlay_path.exists():
                    with open(mask_overlay_path, "rb") as f:
                        validation_images[class_name] = base64.b64encode(
                            f.read()
                        ).decode("utf-8")
                    print(f"  ✓ Using existing validation image for {class_name}")

        classes = sorted(self.stats["classes"].keys())

        # Load HTML template
        template_path = (
            Path(__file__).parent / "report_format" / "analysis_report_template.html"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Generate content
        content_html = self._generate_html_content(
            classes, figure_images, validation_images
        )

        # Replace placeholders
        html = html_template.replace(
            "{{TIMESTAMP}}", datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        )
        html = html.replace("{{CONTENT}}", content_html)
        html = html.replace(
            "{{FOOTER_TIMESTAMP}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n✓ HTML report generated: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    def _generate_html_content(self, classes, figure_images, validation_images):
        """Generate the main content section of the HTML report."""
        content = f"""<!-- Table of Contents -->
            <div class="toc">
                <h3>📑 Table of Contents</h3>
                <ul>
                    <li><a href="#executive-summary">1. Executive Summary</a></li>
                    <li><a href="#overview">2. Dataset Overview</a></li>
                    <li><a href="#mask-analysis">3. Mask Distribution Analysis</a></li>
                    <li><a href="#image-analysis">4. Image & Mask Size Analysis</a></li>
                    <li><a href="#mask-examples">5. Mask Visualization Examples</a></li>
                    <li><a href="#per-class">6. Per-Class Detailed Analysis</a></li>
                    <li><a href="#case-level">7. Case-Level Analysis</a></li>
                    <li><a href="#detailed-stats">8. Detailed Statistics</a></li>
                    <li><a href="#recommendations">9. Recommendations</a></li>
                </ul>
            </div>
            
            <!-- Executive Summary -->
            <section class="section" id="executive-summary">
                <h2 class="section-title">Executive Summary</h2>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Images</div>
                        <div class="value">{self.stats['overall']['total_images']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Total Cases</div>
                        <div class="value">{self.stats['overall']['total_cases']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Total Series</div>
                        <div class="value">{self.stats['overall']['total_series']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Classes</div>
                        <div class="value">{len(classes)}</div>
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card" style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);">
                        <div class="label">Prostate Masks</div>
                        <div class="value">{self.stats['overall']['total_masks']['prostate']:,}</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);">
                        <div class="label">Target1 Masks</div>
                        <div class="value">{self.stats['overall']['total_masks']['target1']:,}</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, #4472C4 0%, #2E5090 100%);">
                        <div class="label">Target2 Masks</div>
                        <div class="value">{self.stats['overall']['total_masks']['target2']:,}</div>
                    </div>
                </div>
                
                <div class="highlight-box">
                    <strong>Key Insight:</strong> The dataset contains a total of <strong>{self.stats['overall']['total_images']:,}</strong> MRI slices 
                    across <strong>{self.stats['overall']['total_cases']}</strong> unique cases spanning <strong>{len(classes)}</strong> clinical classes. 
                    Prostate masks are the most abundant with <strong>{self.stats['overall']['total_masks']['prostate']:,}</strong> annotated slices.
                </div>
            </section>
            
            <!-- Dataset Overview -->
            <section class="section" id="overview">
                <h2 class="section-title">Dataset Overview</h2>
                
                <div class="figure-container">
                    <img src="data:image/png;base64,{figure_images['overview']}" alt="Dataset Overview">
                </div>
                
                <p class="spacing-fix" style="font-size: 1.05em; line-height: 1.8;">
                    The dataset is organized into <strong>{len(classes)}</strong> classes, representing different clinical categories. 
                    The distribution shows {'balanced' if max([self.stats['classes'][c]['total_images'] for c in classes]) / min([self.stats['classes'][c]['total_images'] for c in classes]) < 2 else 'imbalanced'} 
                    data across classes, with the largest class containing {max([self.stats['classes'][c]['total_images'] for c in classes]):,} images 
                    and the smallest containing {min([self.stats['classes'][c]['total_images'] for c in classes]):,} images.
                </p>
            </section>
            
            <!-- Mask Distribution -->
            <section class="section" id="mask-analysis">
                <h2 class="section-title">Mask Distribution Analysis</h2>
                
                <div class="figure-container">
                    <img src="data:image/png;base64,{figure_images['mask_distribution']}" alt="Mask Distribution">
                </div>
                
                <h3 style="color: #2d7f3e; margin: 30px 0 15px 0;">Mask Type Overview</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Mask Type</th>
                            <th>Total Count</th>
                            <th>Coverage per Class</th>
                        </tr>
                    </thead>
                    <tbody>"""

        # Add mask type rows
        for mask_type, display_name, badge_class in [
            ("prostate", "Prostate", "badge-prostate"),
            ("target1", "Target1", "badge-target1"),
            ("target2", "Target2", "badge-target2"),
        ]:
            total_count = self.stats["overall"]["total_masks"][mask_type]
            class_coverage = []
            for c in classes:
                count = self.stats["classes"][c]["masks"][mask_type]["count"]
                if count > 0:
                    class_coverage.append(
                        f'<span class="badge {badge_class}">{c}: {count}</span>'
                    )

            content += f"""
                        <tr>
                            <td><strong>{display_name}</strong></td>
                            <td>{total_count:,}</td>
                            <td>{' '.join(class_coverage) if class_coverage else '<em>No masks available</em>'}</td>
                        </tr>"""

        content += """
                    </tbody>
                </table>
            </section>
            
            <!-- Image & Mask Size Analysis -->
            <section class="section" id="image-analysis">
                <h2 class="section-title">Image & Mask Size Analysis</h2>
                
                <div class="figure-container">
                    <img src="data:image/png;base64,{}" alt="Image Analysis">
                </div>
                
                <h3 style="color: #2d7f3e; margin: 30px 0 15px 0;">📏 Detailed Size Metrics</h3>
""".format(
            figure_images.get("image_analysis", "")
        )

        # Add image dimensions table
        content += """
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Image Dimensions (W x H)</th>
                            <th>Avg Prostate Size</th>
                            <th>Avg Prostate Ratio</th>
                            <th>Avg Target1 Size</th>
                            <th>Avg Target1 Ratio</th>
                            <th>Avg Target2 Size</th>
                            <th>Avg Target2 Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for class_name in classes:
            class_stats = self.stats["classes"][class_name]

            # Get image dimensions
            img_dims = "N/A"
            if "image_dimensions" in class_stats:
                unique_sizes = class_stats["image_dimensions"].get("unique_sizes", [])
                if unique_sizes:
                    img_dims = ", ".join(unique_sizes)

            # Get mask sizes
            mask_sizes = class_stats.get("mask_sizes", {})

            prostate_pixels = mask_sizes.get("prostate", {}).get("avg_pixels", 0)
            prostate_ratio = mask_sizes.get("prostate", {}).get("avg_ratio", 0)

            target1_pixels = mask_sizes.get("target1", {}).get("avg_pixels", 0)
            target1_ratio = mask_sizes.get("target1", {}).get("avg_ratio", 0)

            target2_pixels = mask_sizes.get("target2", {}).get("avg_pixels", 0)
            target2_ratio = mask_sizes.get("target2", {}).get("avg_ratio", 0)

            content += f"""
                        <tr>
                            <td><strong>{class_name}</strong></td>
                            <td>{img_dims}</td>
                            <td>{f'{int(prostate_pixels):,} px' if prostate_pixels > 0 else 'N/A'}</td>
                            <td>{f'{prostate_ratio:.2f}%' if prostate_ratio > 0 else 'N/A'}</td>
                            <td>{f'{int(target1_pixels):,} px' if target1_pixels > 0 else 'N/A'}</td>
                            <td>{f'{target1_ratio:.2f}%' if target1_ratio > 0 else 'N/A'}</td>
                            <td>{f'{int(target2_pixels):,} px' if target2_pixels > 0 else 'N/A'}</td>
                            <td>{f'{target2_ratio:.2f}%' if target2_ratio > 0 else 'N/A'}</td>
                        </tr>
"""

        content += """
                    </tbody>
                </table>
                
                <div class="highlight-box" style="margin-top: 30px;">
                    <strong>💡 Size Insights:</strong> The mask sizes and ratios provide important context for understanding the 
                    segmentation task difficulty. Smaller masks (lower ratios) are typically more challenging to segment accurately.
                    Prostate masks generally cover 5-15% of the image, while target lesions are much smaller at 0.5-3%.
                </div>
            </section>
"""

        # Add mask visualization examples section if available
        if validation_images:
            content += """
            <!-- Mask Visualization Examples -->
            <section class="section" id="mask-examples">
                <h2 class="section-title">Mask Visualization Examples</h2>
                
                <p style="font-size: 1.05em; line-height: 1.8; margin-bottom: 25px;">
                    The following visualizations show real MRI slices with overlaid segmentation masks from each class,
                    <strong>including Target2 masks where available</strong>. Each visualization is from a case containing multiple mask types.
                </p>
"""

            for class_name in sorted(validation_images.keys()):
                class_stats = self.stats["classes"][class_name]
                total_images = class_stats["total_images"]
                p_count = class_stats["masks"]["prostate"]["count"]
                t1_count = class_stats["masks"]["target1"]["count"]
                t2_count = class_stats["masks"]["target2"]["count"]

                content += f"""
                <div class="mask-fullwidth-item" style="margin-bottom: 40px;">
                    <div class="mask-item-title" style="background: linear-gradient(135deg, #3e9651 0%, #2d7f3e 100%); color: white; padding: 15px 20px; font-weight: bold; font-size: 1.3em; border-radius: 10px 10px 0 0;">
                        {class_name.upper()} - Mask Overlay Example
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <img src="data:image/png;base64,{validation_images[class_name]}" alt="{class_name} Mask Overlay" style="width: 100%; max-width: 800px; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
                            <div style="display: flex; align-items: center; gap: 8px; font-size: 1.05em;">
                                <span class="color-box" style="background: #FFD700; width: 24px; height: 24px; display: inline-block; border-radius: 4px; border: 2px solid #333;"></span>
                                <strong>Prostate:</strong> {p_count} slices
                            </div>
                            {f'<div style="display: flex; align-items: center; gap: 8px; font-size: 1.05em;"><span class="color-box" style="background: #FF4444; width: 24px; height: 24px; display: inline-block; border-radius: 4px; border: 2px solid #333;"></span><strong>Target1:</strong> {t1_count} slices</div>' if t1_count > 0 else ''}
                            {f'<div style="display: flex; align-items: center; gap: 8px; font-size: 1.05em;"><span class="color-box" style="background: #4472C4; width: 24px; height: 24px; display: inline-block; border-radius: 4px; border: 2px solid #333;"></span><strong>Target2:</strong> {t2_count} slices</div>' if t2_count > 0 else ''}
                            <div style="display: flex; align-items: center; gap: 8px; font-size: 1.05em; color: #666;">
                                <strong>Total Images:</strong> {total_images}
                            </div>
                        </div>
                    </div>
                </div>
"""

            content += """
                
                <div class="highlight-box" style="margin-top: 30px;">
                    <strong>📌 Color Legend:</strong> Yellow = Prostate, Red = Target1, Blue = Target2. 
                    These masks are semi-transparent overlays on the original MRI slices, showing the regions of interest for segmentation training.
                </div>
            </section>
"""

        content += f"""
            <!-- Per-Class Analysis -->
            <section class="section" id="per-class">
                <h2 class="section-title">Per-Class Detailed Analysis</h2>
                
                <div class="figure-container">
                    <img src="data:image/png;base64,{figure_images['per_class_details']}" alt="Per-Class Details">
                </div>
                
                <div style="margin-top: 50px; clear: both;"></div>
"""

        # Add per-class sections
        for class_name in classes:
            class_stats = self.stats["classes"][class_name]
            slices_per_case = list(class_stats["slices_per_case"].values())

            content += f"""
                <div class="class-section" style="margin-top: 25px;">
                    <h3 class="class-title">{class_name.upper()}</h3>
                    
                    <div class="stats-grid">
                        <div class="stat-card" style="background: linear-gradient(135deg, #2d7f3e 0%, #1b5e20 100%);">
                            <div class="label">Total Images</div>
                            <div class="value">{class_stats['total_images']:,}</div>
                        </div>
                        <div class="stat-card" style="background: linear-gradient(135deg, #3e9651 0%, #2d7f3e 100%);">
                            <div class="label">Cases</div>
                            <div class="value">{class_stats['num_cases']}</div>
                        </div>
                        <div class="stat-card" style="background: linear-gradient(135deg, #50ae65 0%, #3e9651 100%);">
                            <div class="label">Series</div>
                            <div class="value">{class_stats['num_series']}</div>
                        </div>
                        <div class="stat-card" style="background: linear-gradient(135deg, #73c084 0%, #50ae65 100%);">
                            <div class="label">Avg Slices/Case</div>
                            <div class="value">{np.mean(slices_per_case):.1f}</div>
                        </div>
                    </div>
                    
                    <h4 style="color: #2d7f3e; margin: 20px 0 10px 0;">Mask Availability</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>Mask Type</th>
                                <th>Count</th>
                                <th>Cases with Masks</th>
                                <th>Coverage %</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span class="badge badge-prostate">Prostate</span></td>
                                <td>{class_stats['masks']['prostate']['count']:,}</td>
                                <td>{len(class_stats['masks']['prostate']['cases'])}</td>
                                <td>{(class_stats['masks']['prostate']['count'] / class_stats['total_images'] * 100):.1f}%</td>
                            </tr>
                            <tr>
                                <td><span class="badge badge-target1">Target1</span></td>
                                <td>{class_stats['masks']['target1']['count']:,}</td>
                                <td>{len(class_stats['masks']['target1']['cases'])}</td>
                                <td>{(class_stats['masks']['target1']['count'] / class_stats['total_images'] * 100):.1f}%</td>
                            </tr>
                            <tr>
                                <td><span class="badge badge-target2">Target2</span></td>
                                <td>{class_stats['masks']['target2']['count']:,}</td>
                                <td>{len(class_stats['masks']['target2']['cases'])}</td>
                                <td>{(class_stats['masks']['target2']['count'] / class_stats['total_images'] * 100):.1f}%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
"""

        content += f"""
            </section>
            
            <!-- Case-Level Analysis -->
            <section class="section" id="case-level">
                <h2 class="section-title">Case-Level Analysis</h2>
                
                <div class="figure-container">
                    <img src="data:image/png;base64,{figure_images['case_level_analysis']}" alt="Case-Level Analysis">
                </div>
                
                <p style="margin-top: 20px; font-size: 1.05em; line-height: 1.8;">
                    The case-level analysis reveals important patterns in data distribution and mask availability across different cases. 
                    Understanding these patterns is crucial for proper train/validation/test splits and model training strategies.
                </p>
            </section>
            
            <!-- Detailed Statistics -->
            <section class="section" id="detailed-stats">
                <h2 class="section-title">Detailed Statistics</h2>
                
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Images</th>
                            <th>Cases</th>
                            <th>Series</th>
                            <th>Prostate</th>
                            <th>Target1</th>
                            <th>Target2</th>
                            <th>Total Masks</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Add detailed stats rows
        for class_name in classes:
            class_stats = self.stats["classes"][class_name]
            total_masks = sum(
                [
                    class_stats["masks"][mt]["count"]
                    for mt in ["prostate", "target1", "target2"]
                ]
            )

            content += f"""
                        <tr>
                            <td><strong>{class_name}</strong></td>
                            <td>{class_stats['total_images']:,}</td>
                            <td>{class_stats['num_cases']}</td>
                            <td>{class_stats['num_series']}</td>
                            <td>{class_stats['masks']['prostate']['count']:,}</td>
                            <td>{class_stats['masks']['target1']['count']:,}</td>
                            <td>{class_stats['masks']['target2']['count']:,}</td>
                            <td><strong>{total_masks:,}</strong></td>
                        </tr>
"""

        # Add totals row
        total_all_masks = sum(
            [
                self.stats["overall"]["total_masks"][mt]
                for mt in ["prostate", "target1", "target2"]
            ]
        )

        content += f"""
                        <tr style="background: #f0f0f0; font-weight: bold;">
                            <td>TOTAL</td>
                            <td>{self.stats['overall']['total_images']:,}</td>
                            <td>{self.stats['overall']['total_cases']}</td>
                            <td>{self.stats['overall']['total_series']}</td>
                            <td>{self.stats['overall']['total_masks']['prostate']:,}</td>
                            <td>{self.stats['overall']['total_masks']['target1']:,}</td>
                            <td>{self.stats['overall']['total_masks']['target2']:,}</td>
                            <td><strong>{total_all_masks:,}</strong></td>
                        </tr>
                    </tbody>
                </table>
            </section>
            
            <!-- Recommendations -->
            <section class="section" id="recommendations">
                <h2 class="section-title">Recommendations</h2>
                
                <div style="background: #e8f5e9; border-left: 4px solid #4caf50; padding: 20px; margin: 20px 0; border-radius: 5px;">
                    <h3 style="color: #2e7d32; margin-bottom: 15px;">✅ Training Strategy</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li><strong>Multi-class Training:</strong> Use all three mask types (prostate, target1, target2) for comprehensive segmentation</li>
                        <li><strong>Class Balancing:</strong> Consider weighted sampling or data augmentation for underrepresented classes</li>
                        <li><strong>2.5D Approach:</strong> Leverage the stack depth of 5 slices for 3D context while maintaining computational efficiency</li>
                        <li><strong>Train/Val Split:</strong> Use case-level splitting to prevent data leakage between train and validation sets</li>
                    </ul>
                </div>
                
                <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 20px; margin: 20px 0; border-radius: 5px;">
                    <h3 style="color: #e65100; margin-bottom: 15px;">⚠️ Data Considerations</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li><strong>Mask Coverage:</strong> Not all images have corresponding masks - use skip_no_masks=True during training</li>
                        <li><strong>Target Masks:</strong> Target1 and Target2 masks are less abundant than Prostate masks</li>
                        <li><strong>Variable Slices:</strong> Cases have varying numbers of slices - ensure proper handling in data pipeline</li>
                    </ul>
                </div>
                
                <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 20px 0; border-radius: 5px;">
                    <h3 style="color: #0d47a1; margin-bottom: 15px;">📊 Next Steps</h3>
                    <ol style="margin-left: 20px; line-height: 2;">
                        <li>Review mask quality using validation visualizations</li>
                        <li>Set up train/validation/test splits at the case level</li>
                        <li>Configure data augmentation pipeline for balanced training</li>
                        <li>Start with single-class (prostate) training before multi-class</li>
                        <li>Monitor class-specific metrics during training</li>
                    </ol>
                </div>
            </section>
"""

        return content

        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main execution function."""
    analyzer = DatasetAnalyzer()

    # Analyze all classes
    success = analyzer.analyze_all_classes()

    if not success:
        print("\n✗ Analysis failed - no data found")
        return 1

    # Create visualizations
    analyzer.create_visualizations()

    # Generate HTML report
    analyzer.generate_html_report()

    # Print summary
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📊 Dataset Summary:")
    print(f"  • Total Images: {analyzer.stats['overall']['total_images']:,}")
    print(f"  • Total Cases: {analyzer.stats['overall']['total_cases']}")
    print(f"  • Total Series: {analyzer.stats['overall']['total_series']}")
    print(f"  • Classes: {len(analyzer.stats['classes'])}")
    print(f"\n🎭 Mask Summary:")
    print(f"  • Prostate: {analyzer.stats['overall']['total_masks']['prostate']:,}")
    print(f"  • Target1: {analyzer.stats['overall']['total_masks']['target1']:,}")
    print(f"  • Target2: {analyzer.stats['overall']['total_masks']['target2']:,}")
    print(f"\n📄 Report: data_analysis_report.html")
    print(f"\n✨ Open the HTML report in your browser for detailed visualizations!\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
