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

        # Analyze masks
        processed_seg_class_dir = self.processed_seg_dir / class_name

        if processed_seg_class_dir.exists():
            mask_stats = self.analyze_masks(processed_seg_class_dir, df)
            stats["masks"] = mask_stats

        print(f"  ✓ Images: {stats['total_images']}")
        print(f"  ✓ Cases: {stats['num_cases']}")
        print(f"  ✓ Series: {stats['num_series']}")
        print(
            f"  ✓ Masks: P={stats['masks']['prostate']['count']}, "
            f"T1={stats['masks']['target1']['count']}, "
            f"T2={stats['masks']['target2']['count']}"
        )

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

    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("=" * 80)
        print("  CREATING VISUALIZATIONS")
        print("=" * 80)
        print()

        self.create_overview_figure()
        self.create_mask_distribution_figure()
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
        ]  # Yellow, Red (highlight), Orange
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

        # Load validation images
        print("\nLoading validation images...")
        validation_images = self.load_validation_images()

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
                    <li><a href="#mask-examples">4. Mask Visualization Examples</a></li>
                    <li><a href="#per-class">5. Per-Class Detailed Analysis</a></li>
                    <li><a href="#case-level">6. Case-Level Analysis</a></li>
                    <li><a href="#detailed-stats">7. Detailed Statistics</a></li>
                    <li><a href="#recommendations">8. Recommendations</a></li>
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
                    <div class="stat-card" style="background: linear-gradient(135deg, #FF8C00 0%, #FF6600 100%);">
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
"""

        # Add mask visualization examples section if available
        if validation_images:
            content += """
            <!-- Mask Visualization Examples -->
            <section class="section" id="mask-examples">
                <h2 class="section-title">Mask Visualization Examples</h2>
                
                <p style="font-size: 1.05em; line-height: 1.8; margin-bottom: 25px;">
                    The following visualizations show real MRI slices with overlaid segmentation masks from each class.
                    <span class="tooltip-container">
                        Hover over each image
                        <span class="tooltip">Interactive tooltips show class and mask type information</span>
                    </span>
                    to see detailed information about the class and mask types present.
                </p>
                
                <div class="mask-gallery">
"""

            for class_name in sorted(validation_images.keys()):
                class_stats = self.stats["classes"][class_name]
                total_images = class_stats["total_images"]
                p_count = class_stats["masks"]["prostate"]["count"]
                t1_count = class_stats["masks"]["target1"]["count"]
                t2_count = class_stats["masks"]["target2"]["count"]

                content += f"""
                    <div class="mask-item tooltip-container">
                        <div class="mask-item-title">{class_name.upper()}</div>
                        <img src="data:image/png;base64,{validation_images[class_name]}" alt="{class_name} Mask Overlay">
                        <div class="mask-legend">
                            <div><span class="color-box" style="background: #FFD700;"></span> Prostate ({p_count} slices)</div>
                            {f'<div><span class="color-box" style="background: #FF4444;"></span> Target1 ({t1_count} slices)</div>' if t1_count > 0 else ''}
                            {f'<div><span class="color-box" style="background: #FF8C00;"></span> Target2 ({t2_count} slices)</div>' if t2_count > 0 else ''}
                        </div>
                        <span class="tooltip">
                            <strong>{class_name.upper()}</strong><br>
                            Total: {total_images} images<br>
                            Prostate: {p_count} masks<br>
                            {'Target1: ' + str(t1_count) + ' masks<br>' if t1_count > 0 else ''}
                            {'Target2: ' + str(t2_count) + ' masks' if t2_count > 0 else ''}
                        </span>
                    </div>
"""

            content += """
                </div>
                
                <div class="highlight-box" style="margin-top: 30px;">
                    <strong>📌 Color Legend:</strong> Yellow = Prostate, Red = Target1, Orange = Target2. 
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
"""

        # Add per-class sections
        for class_name in classes:
            class_stats = self.stats["classes"][class_name]
            slices_per_case = list(class_stats["slices_per_case"].values())

            content += f"""
                <div class="class-section">
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
