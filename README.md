## Overview

- **Goal**: UNet model to colorize SAR (1ch) into RGB (3ch)
- **Stack**: Python 3.10, PyTorch, AMP, metrics (PSNR/SSIM/LPIPS), Streamlit UI
- **OS**: Windows 10/11
- **GPU**: NVIDIA (e.g., RTX 4060). Ensure a recent driver (check with `nvidia-smi`).

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

If your data root differs, either:
- update `root_dir` default in `src/data_loader.py`, or
- pass `--data-root "path\to\data"` (CLI) or set `data_root` in `scripts/train_config.py`.

## Environment setup (conda + uv)

1) Create/activate conda env (recommended on Windows):
```
conda create -n sar-unet python=3.10 -y
	
```

2) Create uv venv and install base deps:
```
uv venv -p 3.10
uv pip install -r requirements.txt
```

3) Install PyTorch (GPU build). Use the official index for CUDA wheels:
```
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```
If a CUDA 12.9 wheel is released, replace `cu124` with `cu129`.

CPU-only fallback:
```
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

4) Verify GPU:
```
uv run python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'GPU available', torch.cuda.is_available())"
```

## Training

Two options: config script (recommended) or CLI.

### A) Config-based training
Edit and run `scripts/train_config.py`:
```
CONFIG = {
  'data_root': r"D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2",
  'save_dir': 'results/unet',
  'epochs': 50,
  'batch_size': 16,
  'image_size': 256,
  'num_workers': 8,
  'lr': 2e-4,
  'save_interval': 5,
  'resume': '',  # e.g., 'results/unet/checkpoints/epoch_5.pt' to continue
  'scheduler': 'plateau',  # 'none' | 'step' | 'plateau'
  'step_size': 20,
  'gamma': 0.5,
}

uv run python scripts/train_config.py
```

### B) CLI
```
uv run python scripts/train.py \
  --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2" \
  --save-dir results/unet \
  --epochs 50 --batch-size 16 --image-size 256 --num-workers 8 \
  --lr 2e-4 --save-interval 5 \
  --resume results/unet/checkpoints/epoch_5.pt \
  --scheduler plateau --gamma 0.5
```

### What you get after training
- Checkpoints: `results/unet/checkpoints/epoch_<N>.pt` (full-state: model, optimizer, scheduler, AMP scaler, epoch)
- Samples: `results/unet/samples/epoch_<N>.png` (input SAR, target RGB, prediction)
- Training log: `results/unet/training_log.csv` (iteration-level L1 loss)
- Validation metrics: `results/unet/val_metrics.csv` (PSNR/SSIM/LPIPS and val L1 per epoch)

### Resume training (to avoid spikes after interruptions)

When resuming, always restore the optimizer, scheduler, and AMP scaler states in addition to model weights. Otherwise, the learning-rate schedule and optimizer moments reset, which can cause visible spikes/dips in validation curves.

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

Config-based resume (recommended):
```
# scripts/train_config.py
CONFIG = {
  # ...
  'resume': 'results/unet/checkpoints/epoch_50.pt',
}
```

CLI resume:
```
uv run python scripts/train.py \
  --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2" \
  --save-dir results/unet \
  --epochs 100 --batch-size 16 --image-size 256 --num-workers 8 \
  --lr 2e-4 --save-interval 5 \
  --resume results/unet/checkpoints/epoch_50.pt \
  --scheduler plateau --gamma 0.5
```

Notes:
- If the original run used a scheduler but you pass `--scheduler none` when resuming, the scheduler state is ignored and a fresh scheduler (or none) is used. Keep the same scheduler settings to continue the schedule smoothly.
- If AMP/scaler state can't be loaded (different PyTorch version), training continues with a fresh scaler.

## Evaluate a single image
```
uv run python scripts/eval.py \
  --checkpoint results/unet/checkpoints/epoch_50.pt \
  --input path\\to\\sar.png \
  --output pred.png
```

## Metrics across epochs
Compute metrics for all checkpoints and write a CSV:
```
uv run python scripts/metrics_across_checkpoints.py --data-root "D:\\COLLEGE\\SAR\\2.7 Gb V_2\\v_2"
```

## Plot trends
Generate plots (saved to `results/unet/plots/`):
```
uv run python scripts/plot_metrics.py
```

## Inference UI (Streamlit)
Run a local UI to upload a SAR `.png` and select a checkpoint:
```
uv run streamlit run scripts/ui.py
```

## Tips for RTX 4060 / GPU utilization
- Use larger `--batch-size` until VRAM is near full (watch `nvidia-smi`).
- AMP is enabled by default; keep it on for speed and memory savings.
- cuDNN benchmark is enabled; keep fixed input size (e.g., 256×256).
- Try increasing `--num-workers` for faster data loading.

## Troubleshooting
- `torch.cuda.is_available() == False`: ensure `nvidia-smi` works and install CUDA-enabled wheels from the PyTorch index (not default PyPI).
- Import errors: run commands from the project root so `scripts/` and `src/` are on `sys.path`.

---

## Git repository and weight handling

- This repo ships with a `.gitignore` that excludes large artifacts and model weights (`*.pt`, `*.pth`, `results/`, `checkpoints/`, datasets, venvs, caches). This keeps the repository lightweight.
- Do NOT commit training outputs or datasets. Host trained weights externally (e.g., Google Drive, OneDrive, Dropbox, Hugging Face, or Git LFS if you choose).

### Initialize and push a new repo
From the project root:
```
git init
git add .
git commit -m "Initial commit: SAR UNet (training + inference)"
# Create a new repo on GitHub/GitLab and then:
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Sharing checkpoints for others
- Upload your best checkpoint file (e.g., `epoch_50.pt`) to a sharing link.
- In your README or repo Releases, provide a download URL and expected save location, e.g.:
  - Place weight at `scripts/results/unet/checkpoints/epoch_50.pt` or any path you pass to `--checkpoint`.
- Checkpoints created by `scripts/train.py` contain a dict with keys `model`, `optimizer`, and `epoch`. The eval script handles both raw `state_dict` and full dicts.

## CPU-only quickstart for inference (no training required)
If you only need to run inference on CPU:

1) Create env and install dependencies:
```
uv venv -p 3.10
uv pip install -r requirements.txt
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

2) Download a pretrained checkpoint (provided by the repo author) and place it anywhere (path used below).

3) Run inference:
```
uv run python scripts/eval.py --checkpoint path\to\epoch_50.pt --input path\to\sar.png --output pred.png
```
The script automatically uses CPU if no GPU is available. No code changes needed.

### Notes
- Input must be a single-channel `.png` (SAR). The script converts to `L` internally.
- `--image-size` defaults to 256; change it if your training used a different size.
- If the checkpoint was saved as a raw `state_dict`, it's also supported.

### Notes on paths and outputs
- All scripts resolve relative paths against the project root (folder containing `scripts/`, `models/`, `src/`). This ensures that defaults like `results/unet/...` always write under the top-level `results/` even if you run commands from inside `scripts/`.
- Use absolute paths if you want to override the location explicitly.

## Repository structure

```
SAR_Analysis_Project/
  models/
    unet.py                 # UNet architecture
  scripts/
    train.py                # Training loop
    eval.py                 # Inference on a single image (CPU/GPU)
    metrics_across_checkpoints.py
    plot_metrics.py
    train_config.py         # Editable config-based entrypoint
    results/                # (ignored) training outputs; contains .gitkeep
  src/
    data_loader.py          # Dataset and transforms for Sentinel data
    metrics.py              # PSNR/SSIM/LPIPS evaluation
    __init__.py
  results/                  # (ignored) top-level results; contains .gitkeep
  data/                     # (ignored) local data; contains .gitkeep
  datasets/                 # (ignored) optional datasets; contains .gitkeep
  logs/                     # (ignored) logs; contains .gitkeep
  requirements.txt
  README.md
  .gitignore
```

- Directories marked "(ignored)" are excluded from git to keep the repo lightweight; `.gitkeep` files are present so the folder structure remains visible when cloning.
- Checkpoints (`*.pt`, `*.pth`) are ignored. Share a download link so others can place a weight file and run `scripts/eval.py` on CPU or GPU.