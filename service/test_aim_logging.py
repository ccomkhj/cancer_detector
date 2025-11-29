#!/usr/bin/env python3
"""
Test Aim logging to verify it's working correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logger import ExperimentLogger
import numpy as np

def test_aim_logging():
    """Test that Aim logging is working"""
    
    print("Testing Aim logging...")
    
    # Create logger
    logger = ExperimentLogger(
        experiment_name="test_logging",
        hyperparams={
            "batch_size": 8,
            "lr": 1e-4,
            "test": True,
        },
        run_name="test_run",
    )
    
    # Log some fake metrics
    print("\nLogging test metrics...")
    for epoch in range(10):
        # Simulate training metrics
        train_loss = 1.0 - (epoch * 0.08) + np.random.rand() * 0.1
        train_dice = epoch * 0.08 + np.random.rand() * 0.1
        
        logger.log_metrics(
            {"loss": train_loss, "dice": train_dice},
            step=epoch,
            context="train",
        )
        
        # Simulate validation metrics  
        val_loss = 1.0 - (epoch * 0.09) + np.random.rand() * 0.1
        val_dice = epoch * 0.09 + np.random.rand() * 0.1
        
        logger.log_metrics(
            {"loss": val_loss, "dice": val_dice},
            step=epoch,
            context="val",
        )
        
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Log system info
    logger.log_system_info()
    
    # Close
    logger.close()
    
    print("\n" + "="*80)
    print("✓ Test complete!")
    print("="*80)
    print("\nNow go to: http://127.0.0.1:43800")
    print("\n1. Click 'Metrics' in the left sidebar")
    print("2. You should see 'loss' and 'dice' metrics")
    print("3. Select metrics to visualize")
    print("4. Group by 'context.subset' to see train vs val")
    print("="*80)

if __name__ == "__main__":
    test_aim_logging()

