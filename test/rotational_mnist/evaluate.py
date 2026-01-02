"""
Evaluation script for comparing rotational robustness of three MNIST models.

Loads trained models and evaluates them on rotated test sets to compare:
- Case 1: Standard CNN trained on upright MNIST only
- Case 2: Standard CNN trained on augmented dataset (all rotations)
- Case 3: ESCNN equivariant CNN trained on upright MNIST only
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from lib.models import StandardCNN, ESCNNCnn
from lib.training import evaluate
from test.rotational_mnist.mnist_loader import get_rotation_test_loaders

logger = logging.getLogger(__name__)


def load_model(
    model_path: str, model_class: type, device: torch.device
) -> nn.Module:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model weights (.pt file)
        model_class: Model class to instantiate (StandardCNN or ESCNNCnn)
        device: Device to load model to

    Returns:
        Loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model_on_rotations(
    model: nn.Module, test_loaders: Dict[int, torch.utils.data.DataLoader],
    device: torch.device
) -> Dict[int, float]:
    """
    Evaluate a model on test sets at different rotation angles.

    Args:
        model: Neural network model
        test_loaders: Dictionary mapping angle -> DataLoader
        device: Device to evaluate on

    Returns:
        Dictionary mapping angle -> accuracy
    """
    results = {}

    for angle, loader in sorted(test_loaders.items()):
        accuracy = evaluate(model, loader, device, log_interval=500)
        results[angle] = accuracy
        logger.info(f"  Angle {angle:3d}°: Accuracy {accuracy:.4f}")

    return results


def create_visualization(
    results: Dict[str, Dict[int, float]], output_path: str
) -> None:
    """
    Create a visualization of robustness across rotation angles.

    Args:
        results: Dictionary mapping case name -> (angle -> accuracy)
        output_path: Path to save the visualization (PNG/PDF)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
        return

    angles = sorted(list(results[list(results.keys())[0]].keys()))
    cases = list(results.keys())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs rotation angle
    for case in cases:
        if results[case]:
            accs = [results[case][angle] * 100 for angle in angles]
            ax1.plot(angles, accs, marker="o", label=case.capitalize(), linewidth=2)

    ax1.set_xlabel("Rotation Angle (degrees)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Model Robustness to Rotations", fontsize=14, fontweight="bold")
    ax1.set_xticks(angles)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 105])

    # Plot 2: Mean accuracy and variance
    case_names = []
    mean_accs = []
    std_accs = []

    for case in cases:
        if results[case]:
            accs = [results[case][angle] * 100 for angle in angles]
            mean_acc = sum(accs) / len(accs)
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5

            case_names.append(case.capitalize())
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax2.bar(case_names, mean_accs, yerr=std_accs, capsize=5, color=colors, alpha=0.7)
    ax2.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax2.set_title("Average Robustness (with std dev)", fontsize=14, fontweight="bold")
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (name, acc, std) in enumerate(zip(case_names, mean_accs, std_accs)):
        ax2.text(i, acc + std + 2, f"{acc:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def main(
    data_dir: str = "data",
    models_dir: str = "models",
    results_dir: str = "models",
    batch_size: int = 32,
) -> None:
    """
    Evaluate all three trained models on rotated MNIST test sets.

    Args:
        data_dir: Directory containing MNIST data files
        models_dir: Directory containing trained model files
        results_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load test loaders for all rotation angles
    logger.info(f"Loading test sets from {data_dir}")
    test_loaders = get_rotation_test_loaders(data_dir, batch_size=batch_size)

    if not test_loaders:
        logger.error(f"No test data found in {data_dir}")
        return

    logger.info(f"Loaded {len(test_loaders)} rotation variants")

    # Initialize results dictionary
    all_results = {}

    # ====== Case 1: Standard CNN on upright MNIST ======
    logger.info("\n" + "=" * 60)
    logger.info("Case 1: Standard CNN trained on upright MNIST only")
    logger.info("=" * 60)

    # Try multiple possible file names for the baseline model
    possible_paths = [
        Path(models_dir) / "standard_cnn_baseline.pt",
        Path(models_dir) / "case1_standard_cnn.pt",
    ]
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path:
        logger.info(f"Loading model from {model_path}")
        model = load_model(str(model_path), StandardCNN, device)
        results_1 = evaluate_model_on_rotations(model, test_loaders, device)
        all_results["baseline"] = results_1
        logger.info(f"✓ Case 1 complete")
    else:
        logger.warning(f"✗ Case 1 model not found (tried: {possible_paths})")
        all_results["baseline"] = {}

    # ====== Case 2: Standard CNN on augmented dataset ======
    logger.info("\n" + "=" * 60)
    logger.info("Case 2: Standard CNN trained on augmented dataset")
    logger.info("=" * 60)
    model_path = Path(models_dir) / "standard_cnn_augmented.pt"

    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = load_model(str(model_path), StandardCNN, device)
        results_2 = evaluate_model_on_rotations(model, test_loaders, device)
        all_results["augmented"] = results_2
        logger.info(f"✓ Case 2 complete")
    else:
        logger.warning(f"✗ Case 2 model not found at {model_path}")
        all_results["augmented"] = {}

    # ====== Case 3: ESCNN equivariant CNN ======
    logger.info("\n" + "=" * 60)
    logger.info("Case 3: ESCNN equivariant CNN trained on upright MNIST only")
    logger.info("=" * 60)

    # Try multiple possible file names for the ESCNN model
    possible_paths = [
        Path(models_dir) / "escnn_cnn.pt",
        Path(models_dir) / "case3_escnn.pt",
    ]
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path:
        logger.info(f"Loading model from {model_path}")
        model = load_model(str(model_path), ESCNNCnn, device)
        results_3 = evaluate_model_on_rotations(model, test_loaders, device)
        all_results["equivariant"] = results_3
        logger.info(f"✓ Case 3 complete")
    else:
        logger.warning(f"✗ Case 3 model not found (tried: {possible_paths})")
        all_results["equivariant"] = {}

    # ====== Save results ======
    logger.info("\n" + "=" * 60)
    logger.info("Saving results")
    logger.info("=" * 60)

    results_file = Path(results_dir) / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_file}")

    # ====== Print comparison table ======
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)
    print_results_table(all_results)

    # ====== Create visualization ======
    logger.info("\n" + "=" * 60)
    logger.info("Creating visualization")
    logger.info("=" * 60)
    viz_path = Path(results_dir) / "evaluation_results.png"
    create_visualization(all_results, str(viz_path))


def print_results_table(results: Dict[str, Dict[int, float]]) -> None:
    """
    Print a formatted comparison table of results.

    Args:
        results: Dictionary mapping case name -> (angle -> accuracy)
    """
    angles = [0, 15, 30, 45, 60, 90, 180, 270]
    cases = ["baseline", "augmented", "equivariant"]

    # Print header
    header = "Angle (°) | Baseline | Augmented | Equivariant"
    print(header)
    print("-" * len(header))

    # Print rows
    for angle in angles:
        row = f"{angle:9d} |"
        for case in cases:
            if case in results and angle in results[case]:
                acc = results[case][angle]
                row += f" {acc*100:7.2f}% |"
            else:
                row += "    N/A  |"
        print(row)

    # Print summary statistics
    print("\n" + "=" * len(header))
    print("Summary Statistics (mean accuracy across all angles):")
    print("-" * len(header))

    for case in cases:
        if case in results and results[case]:
            accs = list(results[case].values())
            mean_acc = sum(accs) / len(accs)
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f"{case:11s}: {mean_acc*100:6.2f}% ± {std_acc*100:5.2f}%")
        else:
            print(f"{case:11s}: No results")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main()
