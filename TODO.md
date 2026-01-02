# Rotational MNIST Robustness Comparison Test

## Objective
Compare three approaches to handling rotated MNIST digits:
1. **Standard CNN** trained on upright digits only
2. **Standard CNN** trained on augmented dataset (upright + all rotations)
3. **Equivariant CNN** (ESCNN) trained on upright digits only

## Data
- Use existing rotated MNIST files in `/data/` (angles: 0°, 15°, 30°, 45°, 60°, 90°, 180°, 270°)
- Standard training set: 60k images
- Standard test set: 10k images
- Augmented set: 60k × 8 = 480k images (combining all rotations)

## Implementation Checklist

### Phase 1: Data Loading
- [x] Create MNIST data loading module (`test/rotational_mnist/mnist_loader.py`)
  - [x] Parse binary idx format files (idx3-ubyte for images, idx1-ubyte for labels)
  - [x] Load standard MNIST train/test
  - [x] Load individual rotated variants
  - [x] Create augmented dataset (combine all rotations)
  - [x] Add DataLoader wrappers for batch loading

### Phase 2: Model Definitions
- [x] Create baseline CNN model (`lib/models/standard_cnn.py`)
  - [x] Design 2-3 layer CNN suitable for MNIST
  - [x] Consistent architecture for cases 1 & 2
- [x] Create ESCNN model (`lib/models/escnn_cnn.py`)
  - [x] Use SO(2)-equivariant convolutions
  - [x] Match capacity roughly to standard CNN

### Phase 3: Training Pipeline
- [x] Create training script (`test/rotational_mnist/train.py`)
  - [x] Case 1: Standard CNN on upright MNIST only
  - [x] Case 2: Standard CNN on augmented dataset
  - [x] Case 3: ESCNN on upright MNIST only
  - [x] Track: training loss, validation accuracy, training time
  - [x] Save trained models

### Phase 4: Evaluation & Testing
- [x] Create evaluation script (`scripts/rotational_mnist/evaluate.py`)
  - [x] Load all three trained models
  - [x] Evaluate on each rotation variant (0°, 15°, 30°, 45°, 60°, 90°, 180°, 270°)
  - [x] Compute accuracy for each rotation angle
  - [x] Generate comparison table
  - [x] Create visualization (accuracy vs rotation angle)

### Phase 5: Results & Documentation
- [ ] Generate results summary
  - [ ] Table showing accuracy across rotations for all three cases
  - [ ] Plot comparing robustness curves
  - [ ] Summary of findings (data efficiency of equivariance vs augmentation)
- [ ] Update README with results and findings

## Expected Outcomes
- **Case 1**: High accuracy at 0°, drops significantly at other angles
- **Case 2**: Consistent accuracy across all angles (but needs 8x more training data)
- **Case 3**: Consistent accuracy across all angles (with only upright training data)
