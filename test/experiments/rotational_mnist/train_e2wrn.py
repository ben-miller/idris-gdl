"""
E(2)-equivariant Wide ResNet trained on upright MNIST only.

This model uses E(2)-equivariant convolutions (8-fold discrete rotations + reflections)
to achieve robustness without data augmentation. Trained on upright (0°) MNIST only but
automatically equivariant to rotations and reflections via the D8 symmetry group.
Expected behavior: strong robustness to rotations achieved through group-theoretic design.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from lib.models import E2WRN_MNIST
from lib.training import TrainingTracker, train_model
from test.rotational_mnist.mnist_loader import get_mnist_loaders

logger = logging.getLogger(__name__)


def train_e2wrn(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Train E(2)-equivariant Wide ResNet on upright MNIST only.

    Args:
        data_dir: Directory containing MNIST files
        output_dir: Directory to save model and metrics
        num_epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        (model, tracker) tuple
    """
    logger.info("Loading upright MNIST data")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, augmented=False)
    logger.info(f"  Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = E2WRN_MNIST(num_classes=10)
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap grayscale MNIST images (1 channel) to 3 channels for Wide_ResNet
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        # Repeat grayscale to 3 channels: (B, 1, 28, 28) -> (B, 3, 28, 28)
        images = images.repeat(1, 3, 1, 1)
        labels = torch.tensor(labels)
        return images, labels

    # Create new loaders with custom collate function
    train_loader_wrapped = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_loader.num_workers,
    )
    test_loader_wrapped = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=test_loader.num_workers,
    )

    tracker = train_model(model, train_loader_wrapped, test_loader_wrapped, num_epochs=num_epochs)

    # Save model
    output_path = Path(output_dir) / "e2wrn_mnist.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker
