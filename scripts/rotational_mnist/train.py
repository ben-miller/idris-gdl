#!/usr/bin/env python3
"""
Training pipeline for rotational MNIST comparison.

Trains models:
- baseline: Standard CNN on upright MNIST only
- augmented: Standard CNN on augmented dataset (all rotations)
- equivariant: ESCNN equivariant CNN on upright MNIST only
- e2wrn: E(2)-equivariant Wide ResNet on upright MNIST only

Loads configuration from config/config-default.yml and config/config.yml (if present).
Results saved to models/training_results.json.

Usage:
    poetry run python scripts/rotational_mnist/train.py                # Train all three
    poetry run python scripts/rotational_mnist/train.py baseline       # Train baseline only
    poetry run python scripts/rotational_mnist/train.py baseline augmented  # Train two
    poetry run python scripts/rotational_mnist/train.py e2wrn          # Train E2WRN only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.config import TrainingConfig, configure_logging
from test.experiments.rotational_mnist.train_baseline import train_standard_cnn_baseline
from test.experiments.rotational_mnist.train_augmented import train_standard_cnn_augmented
from test.experiments.rotational_mnist.train_e2_simple import train_e2_simple
from test.experiments.rotational_mnist.train_e2wrn import train_e2wrn

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)

# Fixed paths
DATA_DIR = "data"
OUTPUT_DIR = "models"


def main() -> None:
    """Train specified models."""
    parser = argparse.ArgumentParser(
        description="Train rotational MNIST models"
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["baseline", "augmented", "e2_simple", "e2wrn"],
        help="Models to train: baseline, augmented, e2_simple, e2wrn (default: all four)",
    )
    args = parser.parse_args()

    config = TrainingConfig.load()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}

    # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Training for {config.epochs} epochs, batch size {config.batch_size}")
    logger.info(f"Models: {', '.join(args.models)}")

    if "baseline" in args.models:
        logger.info("=" * 60)
        logger.info("Training: Standard CNN on upright MNIST (baseline)")
        model1, tracker1 = train_standard_cnn_baseline(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["standard_cnn_baseline"] = tracker1.to_dict()

        # Save baseline results
        baseline_results_file = Path(OUTPUT_DIR) / "training_results.baseline.json"
        with open(baseline_results_file, "w") as f:
            json.dump(tracker1.to_dict(), f, indent=2)
        logger.info(f"Baseline training time: {tracker1.elapsed_time:.2f}s")
        logger.info(f"Baseline final accuracy: {tracker1.val_accuracies[-1]:.4f}")
        logger.info(f"Saved baseline results to {baseline_results_file}\n")

    if "augmented" in args.models:
        logger.info("=" * 60)
        logger.info("Training: Standard CNN on augmented MNIST")
        model2, tracker2 = train_standard_cnn_augmented(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["standard_cnn_augmented"] = tracker2.to_dict()

        # Save augmented results
        augmented_results_file = Path(OUTPUT_DIR) / "training_results.augmented.json"
        with open(augmented_results_file, "w") as f:
            json.dump(tracker2.to_dict(), f, indent=2)
        logger.info(f"Augmented training time: {tracker2.elapsed_time:.2f}s")
        logger.info(f"Augmented final accuracy: {tracker2.val_accuracies[-1]:.4f}")
        logger.info(f"Saved augmented results to {augmented_results_file}\n")

    if "e2_simple" in args.models:
        logger.info("=" * 60)
        logger.info("Training: Lightweight E(2)-equivariant CNN on upright MNIST")
        model5, tracker5 = train_e2_simple(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["e2_simple"] = tracker5.to_dict()

        # Save E2_simple results
        e2_simple_results_file = Path(OUTPUT_DIR) / "training_results.e2_simple.json"
        with open(e2_simple_results_file, "w") as f:
            json.dump(tracker5.to_dict(), f, indent=2)
        logger.info(f"E2_simple training time: {tracker5.elapsed_time:.2f}s")
        logger.info(f"E2_simple final accuracy: {tracker5.val_accuracies[-1]:.4f}")
        logger.info(f"Saved E2_simple results to {e2_simple_results_file}\n")

    if "e2wrn" in args.models:
        logger.info("=" * 60)
        logger.info("Training: E(2)-equivariant Wide ResNet on upright MNIST")
        model6, tracker6 = train_e2wrn(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["e2wrn"] = tracker6.to_dict()

        # Save E2WRN results
        e2wrn_results_file = Path(OUTPUT_DIR) / "training_results.e2wrn.json"
        with open(e2wrn_results_file, "w") as f:
            json.dump(tracker6.to_dict(), f, indent=2)
        logger.info(f"E2WRN training time: {tracker6.elapsed_time:.2f}s")
        logger.info(f"E2WRN final accuracy: {tracker6.val_accuracies[-1]:.4f}")
        logger.info(f"Saved E2WRN results to {e2wrn_results_file}\n")

    # Also save combined results for reference
    results_file = Path(OUTPUT_DIR) / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {results_file}")


if __name__ == "__main__":
    main()
