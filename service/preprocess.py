#!/usr/bin/env python3
"""
Data Preprocessing Pipeline

Orchestrates all data conversion and preparation steps:
1. Convert Excel to Parquet (if needed)
2. Merge datasets (if needed)
3. Generate TCIA manifests (if needed)
4. Convert DICOM to PNG (if needed)
5. Process overlay segmentations (if needed)
6. Validate 2.5D setup

Usage:
    # Run full pipeline
    python service/preprocess.py --all
    
    # Run specific steps
    python service/preprocess.py --step dicom_to_png
    python service/preprocess.py --step overlay_to_masks
    
    # Process specific class
    python service/preprocess.py --class 2 --step dicom_to_png
"""

import sys
import argparse
from pathlib import Path
import subprocess
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates all data preprocessing steps."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.tools_dir = self.project_root / "tools"
        self.data_dir = self.project_root / "data"
        
    def run_command(self, cmd: list, description: str) -> bool:
        """
        Run a command and return success status.
        
        Args:
            cmd: Command to run as list
            description: Description of what's being run
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✓ Success: {description}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {description}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def step_1_convert_excel(self) -> bool:
        """Step 1: Convert Excel files to Parquet."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Convert Excel to Parquet")
        logger.info("="*80)
        
        # Check if already done
        output_dir = self.data_dir / "splitted_images"
        if output_dir.exists() and list(output_dir.glob("class=*/PIRADS_*.parquet")):
            logger.info("✓ Excel files already converted to Parquet")
            return True
        
        cmd = ["python", str(self.tools_dir / "convert_xlsx2parquet.py")]
        return self.run_command(cmd, "Excel to Parquet conversion")
    
    def step_2_merge_datasets(self) -> bool:
        """Step 2: Merge multi-source datasets."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Merge Multi-Source Datasets")
        logger.info("="*80)
        
        # Check if already done
        output_dir = self.data_dir / "splitted_info"
        if output_dir.exists() and list(output_dir.glob("class=*/PIRADS_*.parquet")):
            logger.info("✓ Datasets already merged")
            return True
        
        cmd = ["python", str(self.tools_dir / "merge_datasets.py")]
        return self.run_command(cmd, "Dataset merging")
    
    def step_3_generate_tcia_manifests(self) -> bool:
        """Step 3: Generate TCIA manifest files."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Generate TCIA Manifests")
        logger.info("="*80)
        
        # Check if already done
        tcia_dir = self.data_dir / "tcia"
        if tcia_dir.exists() and list(tcia_dir.glob("*/*.tcia")):
            logger.info("✓ TCIA manifests already generated")
            return True
        
        # Generate by class (T2, ADC, CALC)
        cmd = ["python", str(self.tools_dir / "generate_tcia_by_class.py")]
        if not self.run_command(cmd, "TCIA manifest generation (by class)"):
            return False
        
        # Generate by study (full download)
        cmd = ["python", str(self.tools_dir / "generate_tcia_by_study.py")]
        return self.run_command(cmd, "TCIA manifest generation (by study)")
    
    def step_4_convert_dicom_to_png(self, class_num: int = None) -> bool:
        """Step 4: Convert DICOM files to PNG slices."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Convert DICOM to PNG")
        logger.info("="*80)
        
        # Check if DICOM data exists
        nbia_dir = self.data_dir / "nbia"
        if not nbia_dir.exists() or not list(nbia_dir.glob("class*")):
            logger.warning("⚠ DICOM files not found in data/nbia/")
            logger.info("Please download DICOM files using NBIA Data Retriever:")
            logger.info("  1. Open .tcia files from data/tcia/")
            logger.info("  2. Download to data/nbia/class{1,2,3,4}/")
            return False
        
        # Check if already done
        processed_dir = self.data_dir / "processed"
        if class_num:
            check_dir = processed_dir / f"class{class_num}"
            if check_dir.exists() and (check_dir / "manifest.csv").exists():
                logger.info(f"✓ Class {class_num} already converted")
                return True
            
            cmd = ["python", str(self.tools_dir / "dicom_converter.py"), 
                   "--class", str(class_num)]
            return self.run_command(cmd, f"DICOM to PNG conversion (class {class_num})")
        else:
            # Check if all classes are done
            all_done = all(
                (processed_dir / f"class{i}" / "manifest.csv").exists()
                for i in range(1, 5)
                if (nbia_dir / f"class{i}").exists()
            )
            if all_done:
                logger.info("✓ All classes already converted")
                return True
            
            cmd = ["python", str(self.tools_dir / "dicom_converter.py"), "--all"]
            return self.run_command(cmd, "DICOM to PNG conversion (all classes)")
    
    def step_5_process_overlays(self) -> bool:
        """Step 5: Process overlay segmentations to masks."""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Process Overlay Segmentations")
        logger.info("="*80)
        
        # Check if overlay data exists
        overlay_dir = self.data_dir / "overlay"
        if not overlay_dir.exists():
            logger.warning("⚠ Overlay data not found, skipping")
            return True
        
        # Check if already done
        output_dir = self.data_dir / "processed_seg"
        if output_dir.exists() and list(output_dir.glob("class*/case_*")):
            logger.info("✓ Overlay segmentations already processed")
            return True
        
        cmd = ["python", str(self.tools_dir / "process_overlay_aligned.py")]
        return self.run_command(cmd, "Overlay to mask conversion")
    
    def step_6_validate_2d5_setup(self) -> bool:
        """Step 6: Validate 2.5D pipeline setup."""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Validate 2.5D Setup")
        logger.info("="*80)
        
        cmd = ["python", str(self.tools_dir / "validate_2d5_setup.py")]
        return self.run_command(cmd, "2.5D setup validation")
    
    def run_all(self, skip_manual_steps: bool = True) -> bool:
        """Run all preprocessing steps."""
        logger.info("\n" + "="*80)
        logger.info("FULL PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        steps = [
            ("Excel to Parquet", self.step_1_convert_excel),
            ("Merge Datasets", self.step_2_merge_datasets),
        ]
        
        if not skip_manual_steps:
            steps.append(("Generate TCIA Manifests", self.step_3_generate_tcia_manifests))
        
        steps.extend([
            ("DICOM to PNG", self.step_4_convert_dicom_to_png),
            ("Process Overlays", self.step_5_process_overlays),
            ("Validate 2.5D Setup", self.step_6_validate_2d5_setup),
        ])
        
        results = {}
        for name, step_func in steps:
            success = step_func()
            results[name] = success
            if not success:
                logger.error(f"✗ Pipeline stopped at: {name}")
                break
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)
        for name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            logger.info(f"{name}: {status}")
        
        all_success = all(results.values())
        if all_success:
            logger.info("\n✓ All preprocessing steps completed successfully!")
            logger.info("\nNext steps:")
            logger.info("  1. Review validation output above")
            logger.info("  2. Start training: python service/train.py")
        else:
            logger.info("\n✗ Some steps failed. Check logs above.")
        
        return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Data Preprocessing Pipeline for 2.5D MRI Segmentation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all preprocessing steps"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "excel_to_parquet",
            "merge_datasets",
            "generate_tcia",
            "dicom_to_png",
            "process_overlays",
            "validate_2d5"
        ],
        help="Run specific preprocessing step"
    )
    parser.add_argument(
        "--class",
        dest="class_num",
        type=int,
        choices=[1, 2, 3, 4],
        help="Process specific class (for DICOM conversion)"
    )
    parser.add_argument(
        "--skip-manual",
        action="store_true",
        default=True,
        help="Skip steps requiring manual intervention (default: True)"
    )
    
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline()
    
    if args.all:
        success = pipeline.run_all(skip_manual_steps=args.skip_manual)
        return 0 if success else 1
    
    elif args.step:
        step_map = {
            "excel_to_parquet": pipeline.step_1_convert_excel,
            "merge_datasets": pipeline.step_2_merge_datasets,
            "generate_tcia": pipeline.step_3_generate_tcia_manifests,
            "dicom_to_png": lambda: pipeline.step_4_convert_dicom_to_png(args.class_num),
            "process_overlays": pipeline.step_5_process_overlays,
            "validate_2d5": pipeline.step_6_validate_2d5_setup,
        }
        
        success = step_map[args.step]()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

