# Pix2Pix (cGAN) for SAR → Optical Translation

[← Back to main README](./README.md)

This implementation provides a complete Pix2Pix training and evaluation pipeline for SAR to optical image translation, with comprehensive ablation studies and comparison to UNet baseline.

## Features

- **Complete Pix2Pix Implementation**: UNet generator + PatchGAN discriminator
- **Mixed Precision Training**: Automatic mixed precision with CUDA
- **Comprehensive Ablation Studies**: L1 weight, perceptual loss, learning rate, batch size, discriminator variants
- **Warm-start Training**: Initialize generator from pre-trained UNet
- **Resume Training**: Full checkpointing and resume functionality
- **Reproducible Experiments**: Fixed seeds, config files, hardware logging
- **TensorBoard Logging**: Real-time training monitoring
- **Sample Generation**: Automatic sample image generation during training

## Quick Start

### 1. Train Default Pix2Pix Model

```bash
uv run python scripts/train_pix2pix.py --config configs/pix2pix_default.yaml --data-root datasets/v_2
```

### 2. Train with UNet Warm-start

```bash
uv run python scripts/train_pix2pix.py --config configs/pix2pix_warmstart.yaml --data-root datasets/v_2
```

### 3. Run All Ablation Experiments

```bash
uv run python scripts/run_pix2pix_experiments.py --data-root datasets/v_2
```

### 4. Evaluate Model

```bash
uv run python scripts/eval_pix2pix.py --checkpoint results/pix2pix/default/checkpoints/best_by_val_lpips.pt --data-root datasets/v_2 --save-samples
```

## Model Architecture

### Generator (UNet)
- **Input**: 1-channel SAR image (256×256)
- **Output**: 3-channel RGB image (256×256)
- **Architecture**: Encoder-decoder with skip connections
- **Base Features**: 64 (configurable)

### Discriminator (PatchGAN)
- **Input**: 4-channel concatenated image (SAR + RGB)
- **Output**: 1-channel patch predictions (70×70 receptive field)
- **Architecture**: 5-layer CNN with optional spectral normalization

## Training Configuration

### Default Hyperparameters
- **Learning Rate**: 2e-4
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **Batch Size**: 12
- **Epochs**: 100
- **L1 Weight (λ_L1)**: 100
- **Adversarial Loss**: BCEWithLogitsLoss

### Loss Functions
- **Adversarial Loss**: BCE or Hinge loss
- **Reconstruction Loss**: L1 loss (weighted by λ_L1)
- **Perceptual Loss**: VGG-based (optional, weighted by λ_perc)

## Ablation Experiments

| Experiment | Config File | Description |
|------------|-------------|-------------|
| **Default** | `pix2pix_default.yaml` | Baseline configuration |
| **Warm-start** | `pix2pix_warmstart.yaml` | Generator initialized from UNet |
| **L1=50** | `pix2pix_l1_50.yaml` | Lower reconstruction emphasis |
| **L1=200** | `pix2pix_l1_200.yaml` | Higher reconstruction emphasis |
| **Perceptual** | `pix2pix_perceptual.yaml` | VGG perceptual loss (λ_perc=1.0) |
| **Perceptual+** | `pix2pix_perceptual_5.yaml` | Stronger perceptual loss (λ_perc=5.0) |
| **LR=1e-4** | `pix2pix_lr_1e4.yaml` | Lower learning rate |
| **Batch=16** | `pix2pix_batch16.yaml` | Larger batch size |
| **Spectral Norm** | `pix2pix_spectral_norm.yaml` | Spectral normalization on discriminator |

## Directory Structure

```
results/pix2pix/
├── default/                    # Default experiment
│   ├── checkpoints/           # Model checkpoints
│   │   ├── epoch_5.pt
│   │   ├── best_by_val_lpips.pt
│   │   └── best_by_val_l1.pt
│   ├── samples/               # Generated samples
│   ├── tensorboard/           # TensorBoard logs
│   ├── config.yaml            # Experiment config
│   └── training.log           # Training log
├── warmstart/                 # Warm-start experiment
├── l1_50/                     # L1 weight ablation
└── ...
```

## Training Features

### Mixed Precision
- Automatic mixed precision training with `torch.cuda.amp`
- Reduces memory usage and speeds up training
- Automatic gradient scaling

### Checkpointing
- **Regular checkpoints**: Every 5 epochs
- **Best checkpoints**: Best by validation LPIPS and L1 loss
- **Resume training**: Full state restoration
- **Config preservation**: All hyperparameters saved

### Learning Rate Scheduling
- **Linear Decay**: Constant for 50 epochs, then linear decay to 0
- **Plateau**: Reduce on validation LPIPS plateau
- **Custom**: Configurable scheduler options

### Early Stopping
- Monitor validation LPIPS
- Stop if no improvement for 10 epochs
- Configurable patience and minimum delta

## Evaluation

### Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **L1 Loss**: Mean Absolute Error

### Sample Generation
- Automatic sample generation during training
- Visual comparison: SAR input → Ground truth → Generated
- Configurable number of samples

## Usage Examples

### Single Experiment
```bash
# Train with custom experiment name
uv run python scripts/train_pix2pix.py \
    --config configs/pix2pix_default.yaml \
    --data-root datasets/v_2 \
    --experiment-name my_experiment
```

### Resume Training
```bash
# Resume from checkpoint
uv run python scripts/train_pix2pix.py \
    --config configs/pix2pix_default.yaml \
    --data-root datasets/v_2 \
    --resume results/pix2pix/default/checkpoints/epoch_50.pt
```

### Specific Ablations
```bash
# Run only L1 weight experiments
uv run python scripts/run_pix2pix_experiments.py \
    --data-root datasets/v_2 \
    --experiments l1_50 l1_200
```

### Evaluation with Samples
```bash
# Evaluate and save 50 sample images
uv run python scripts/eval_pix2pix.py \
    --checkpoint results/pix2pix/default/checkpoints/best_by_val_lpips.pt \
    --data-root datasets/v_2 \
    --save-samples \
    --num-samples 50
```

## Configuration

All experiments use YAML configuration files with the following structure:

```yaml
# Model parameters
in_channels: 1
out_channels: 3
base_features: 64
use_spectral_norm: false

# Training parameters
epochs: 100
batch_size: 12
lr: 0.0002
betas: [0.5, 0.999]
scheduler: "linear_decay"

# Loss configuration
adv_loss: "bce"
lambda_l1: 100
use_perceptual_loss: false
lambda_perc: 1.0

# Data parameters
image_size: 256
num_workers: 4

# Training configuration
checkpoint_interval: 5
log_interval: 100
early_stopping: true
patience: 10

# Reproducibility
seed: 42

# Warm-start
warmstart_path: null
```

## Monitoring Training

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir results/pix2pix/default/tensorboard
```

### Log Files
- **Training log**: `results/pix2pix/{experiment}/training.log`
- **Config**: `results/pix2pix/{experiment}/config.yaml`
- **Metrics CSV**: Available through TensorBoard

## Comparison with UNet

The Pix2Pix implementation is designed to be directly comparable to the UNet baseline:

- **Same dataset splits**: Uses identical train/val/test splits
- **Same preprocessing**: Identical normalization and augmentation
- **Same evaluation metrics**: PSNR, SSIM, LPIPS, L1 loss
- **Warm-start option**: Can initialize from best UNet checkpoint

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended
- **RAM**: 16GB+ recommended for batch size 12
- **Storage**: ~2GB per experiment (checkpoints + logs)

## Troubleshooting

### Memory Issues
- Reduce batch size in config
- Reduce number of workers
- Enable gradient checkpointing

### Training Instability
- Use spectral normalization
- Reduce learning rate
- Adjust L1 weight
- Monitor discriminator/generator loss ratio

### Slow Training
- Enable mixed precision (automatic)
- Increase number of workers
- Use faster storage for dataset

## Results Analysis

After training, compare results across experiments:

1. **Load TensorBoard logs** for training curves
2. **Compare validation metrics** across configs
3. **Visual inspection** of generated samples
4. **Statistical analysis** of test set performance

The implementation provides comprehensive logging and evaluation tools for thorough analysis of the Pix2Pix approach compared to the UNet baseline.

## 📊 Results (Placeholder)

TODO: Insert qualitative samples and a quantitative metrics table (PSNR/SSIM/LPIPS) for Pix2Pix here.

## Pix2Pix Metrics Plots (TensorBoard export)

You can export clean, publication-ready plots from TensorBoard logs and generate a quick summary report.

Dependencies: tensorboard (event processing) and scipy (for smoothing).

### Usage (import as functions)

```python
from scripts.plot_metrics_pix2pix import export_tensorboard_plots, create_summary_report

export_tensorboard_plots(
    logdir='results/pix2pix/pix2pix_20250920_005609/tensorboard',
    output_dir='results/pix2pix/pix2pix_20250920_005609/plots'
)

create_summary_report(
    logdir='results/pix2pix/pix2pix_20250920_005609/tensorboard',
    output_dir='results/pix2pix/pix2pix_20250920_005609/plots'
)
```

Example output files:

```
plots/
├── train_loss.png
├── val_loss.png  
├── psnr.png
├── ssim.png
├── generator_loss.png
├── discriminator_loss.png
└── training_summary.txt
```

Each plot matches the clean PSNR style—individual, smoothed, and publication-ready. The bonus `training_summary.txt` gives a quick overview of key numbers without opening each plot.

### Optional: export TensorBoard scalars to CSV

If you want raw CSVs for custom analysis, you can export all scalar tags:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

def export_tensorboard_data(logdir, output_dir):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    os.makedirs(output_dir, exist_ok=True)
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        df = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
            for e in scalar_events
        ])
        filename = tag.replace('/', '_').replace('\\', '_') + '.csv'
        df.to_csv(os.path.join(output_dir, filename), index=False)

# Example
export_tensorboard_data(
    'results/pix2pix/pix2pix_20250920_005609/tensorboard',
    'results/pix2pix/pix2pix_20250920_005609/tensorboard_exports'
)
```

This creates one CSV per scalar tag in the specified exports folder.