# U-Net Baseline for SAR → Optical Translation

[← Back to main README](./README.md)

This document covers the U-Net baseline used to colorize single-channel SAR images into 3-channel optical RGB images. It includes setup, training, evaluation, logging, and tips targeted for Windows + NVIDIA GPUs.

## Features

- Lightweight U-Net generator for SAR→RGB translation
- AMP (automatic mixed precision) on CUDA for speed and memory savings
- Robust training loop with resume (optimizer/scheduler/AMP scaler) support
- Metrics: PSNR, SSIM, LPIPS; CSV logs and plots
- Streamlit UI for quick inference demos
- Reproducible runs and organized results directory

## Dataset layout (required by `SARDataset`)

`src/data_loader.py` expects this structure under a root folder (default: `D:\COLLEGE\SAR\2.7 Gb V_2\v_2`):

```
<ROOT>/
  <category_1>/
    s1/
      <scene_1>_..._s1_....png
      <scene_2>_..._s1_....png
      ...
    s2/
      <scene_1>_..._s2_....png
      <scene_2>_..._s2_....png
      ...
  <category_2>/
    s1/
    s2/
  ...
```

- Images are PNG.
- Pairing: for each file in `s1`, replace the 3rd underscore-delimited token with `s2` to find the match in `s2`.
- Both SAR (grayscale) and optical (RGB) are resized to 256×256, normalized to [-1, 1].

Override at runtime with `--data-root "path\to\data"` or edit defaults in `src/data_loader.py`.

## Environment setup (conda + uv)

1) Create/activate conda env (recommended on Windows):

```powershell
conda create -n sar-unet python=3.10 -y
conda activate sar-unet
```

2) Create uv venv and install base deps:

```powershell
uv venv -p 3.10
uv pip install -r requirements.txt
```

3) Install PyTorch (GPU build):

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

CPU-only fallback:

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

4) Verify GPU:

```powershell
uv run python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'GPU available', torch.cuda.is_available())"
```

## Training

Two entrypoints are provided: an editable config script and a CLI.

### A) Config-based training (recommended)
Edit and run `scripts/train_config.py`:

```python
CONFIG = {
  'data_root': r"D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2",
  'save_dir': 'results/unet',
  'epochs': 50,
  'batch_size': 16,
  'image_size': 256,
  'num_workers': 8,
  'lr': 2e-4,
  'save_interval': 5,
  'resume': '',  # e.g., 'results/unet/checkpoints/epoch_5.pt'
  'scheduler': 'plateau',  # 'none' | 'step' | 'plateau'
  'step_size': 20,
  'gamma': 0.5,
}
```

Run:

```powershell
uv run python scripts/train_config.py
```

### B) CLI training

```powershell
uv run python scripts/train.py \
  --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2" \
  --save-dir results/unet \
  --epochs 50 --batch-size 16 --image-size 256 --num-workers 8 \
  --lr 2e-4 --save-interval 5 \
  --resume results/unet/checkpoints/epoch_5.pt \
  --scheduler plateau --gamma 0.5
```

### Artifacts produced

- Checkpoints: `results/unet/checkpoints/epoch_<N>.pt` (full-state)
- Samples: `results/unet/samples/epoch_<N>.png` (SAR | GT | Pred)
- Training log: `results/unet/training_log.csv` (iteration-level L1)
- Validation metrics: `results/unet/val_metrics.csv` (PSNR/SSIM/LPIPS + L1 per epoch)

## Resume training (no LR spikes)

Saved checkpoint contents:

```
{
  'model': state_dict,
  'optimizer': state_dict,
  'scheduler': state_dict or None,
  'scaler': state_dict or None,
  'epoch': int
}
```

Config-based resume:

```python
CONFIG = {
  # ...
  'resume': 'results/unet/checkpoints/epoch_50.pt',
}
```

CLI resume:

```powershell
uv run python scripts/train.py \
  --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2" \
  --save-dir results/unet \
  --epochs 100 --batch-size 16 --image-size 256 --num-workers 8 \
  --lr 2e-4 --save-interval 5 \
  --resume results/unet/checkpoints/epoch_50.pt \
  --scheduler plateau --gamma 0.5
```

Notes:
- Keep the same scheduler setting when resuming to avoid schedule resets.
- If AMP scaler state can't be loaded (e.g., different torch), training continues with a fresh scaler.

## Evaluation and analysis

- Evaluate a single image:

```powershell
uv run python scripts/eval.py \
  --checkpoint results/unet/checkpoints/epoch_50.pt \
  --input path\\to\\sar.png \
  --output pred.png
```

- Metrics across epochs (CSV):

```powershell
uv run python scripts/metrics_across_checkpoints.py --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2"
```

- Plot trends to `results/unet/plots/`:

```powershell
uv run python scripts/plot_metrics.py
```

- Inference UI (Streamlit):

```powershell
uv run streamlit run scripts/ui.py
```

## Tips for RTX 4060 / GPU utilization

- Increase `--batch-size` until VRAM is near full (watch `nvidia-smi`).
- AMP and cuDNN benchmark are enabled; keep a fixed input size (256×256).
- Increase `--num-workers` if data loading is the bottleneck.

## Troubleshooting

- `torch.cuda.is_available() == False`: ensure `nvidia-smi` works and install CUDA-enabled wheels via the PyTorch index.
- Import/module errors: run from the project root so `scripts/` and `src/` are on `sys.path`.

## 📊 Results (Placeholder)

TODO: Insert qualitative samples and a quantitative metrics table (PSNR/SSIM/LPIPS) for U-Net here.

---

- Main overview: see [README.md](./README.md)
- Pix2Pix details: see [PIX2PIX_README.md](./PIX2PIX_README.md)