"""Equivariant CNN model using ESCNN for rotational invariance."""

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn


class ESCNNCnn(nn.Module):
    """
    SO(2)-equivariant CNN using ESCNN (E(2)-equivariant Steerable CNNs).

    This model is equivariant to 2D rotations, meaning if the input rotates,
    the output feature maps rotate in the same way. This allows the model
    to learn rotational invariance without data augmentation.

    Architecture:
    - Input: 1 scalar channel (grayscale)
    - Conv1: 1 scalar -> 8 regular SO(2) representations
    - Conv2: 8 regular -> 16 regular SO(2) representations
    - Conv3: 16 regular -> 32 regular SO(2) representations
    - Conv4: 32 regular -> 32 scalars (projection to trivial for classification)
    - Each conv followed by ReLU and MaxPool (2x2, except last conv)
    - FC layers: flatten -> Dense 128 -> ReLU -> Dense 10

    Key difference from naive approach:
    - Uses SO(2) REGULAR representations, not trivial representations
    - Regular reps encode rotation information and transform under group actions
    - Only the final layer projects to trivial (scalar) representation for classification
    - ESCNN's equivariant pooling preserves geometric structure

    Used for:
    - Case 3: Trained on upright MNIST only, but equivariant to rotations
    """

    def __init__(self) -> None:
        super().__init__()

        # Create SO(2) group space (rotations in 2D)
        self.r2_act = gspaces.rot2dOnR2()

        # Input field type: 1 trivial (scalar) channel (grayscale pixel values)
        self.in_type = escnn_nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr])

        # Layer 1: 1 scalar -> only trivial reps to enable pointwise ReLU
        # We'll use many trivial channels and steerable kernels for equivariance
        self.out_type1 = escnn_nn.FieldType(
            self.r2_act,
            16 * [self.r2_act.trivial_repr]
        )
        self.conv1 = escnn_nn.R2Conv(
            self.in_type,
            self.out_type1,
            kernel_size=5,
            padding=2,
            bias=True
        )
        self.relu1 = escnn_nn.ReLU(self.out_type1)
        self.pool1 = escnn_nn.PointwiseMaxPool(self.out_type1, kernel_size=2)

        # Layer 2: Increase channels
        self.out_type2 = escnn_nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.trivial_repr]
        )
        self.conv2 = escnn_nn.R2Conv(
            self.out_type1,
            self.out_type2,
            kernel_size=5,
            padding=2,
            bias=True
        )
        self.relu2 = escnn_nn.ReLU(self.out_type2)
        self.pool2 = escnn_nn.PointwiseMaxPool(self.out_type2, kernel_size=2)

        # Layer 3: Further feature extraction
        self.out_type3 = escnn_nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.trivial_repr]
        )
        self.conv3 = escnn_nn.R2Conv(
            self.out_type2,
            self.out_type3,
            kernel_size=5,
            padding=2,
            bias=True
        )
        self.relu3 = escnn_nn.ReLU(self.out_type3)

        # Final layer output remains in trivial representation (scalars)

        # Fully connected layers
        # After 2 max pooling layers: 28 -> 14 -> 7
        # Calculate FC input size dynamically to handle variable irrep dimensions
        dummy_x = torch.randn(1, 1, 28, 28)
        dummy_in = escnn_nn.GeometricTensor(dummy_x, self.in_type)
        with torch.no_grad():
            dummy_out = self._forward_convs(dummy_in)
        fc_input_size = dummy_out.tensor.view(1, -1).shape[1]

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _forward_convs(self, x: escnn_nn.GeometricTensor) -> escnn_nn.GeometricTensor:
        """
        Forward pass through convolutional layers only.

        Args:
            x: Input GeometricTensor

        Returns:
            Output GeometricTensor after all conv layers (before FC)
        """
        # First equivariant conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second equivariant conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Third equivariant conv block
        x = self.conv3(x)
        x = self.relu3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the equivariant network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Wrap input in GeometricTensor (required by ESCNN)
        x = escnn_nn.GeometricTensor(x, self.in_type)

        # Forward through convolutional layers
        x = self._forward_convs(x)

        # Extract raw tensor and flatten
        x = x.tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
