## SAR Image Colorization – U-Net Baseline with Pix2Pix Warmstart

This project tackles SAR (1-channel) → Optical RGB (3-channel) translation. We provide:

- A strong U-Net baseline for direct SAR→RGB regression
- A Pix2Pix (cGAN) implementation that can warm-start its generator from U-Net to improve realism
- Full training/evaluation pipelines, logging, and utilities

Quick links:
- U-Net details: see `UNET_README.md`
- Pix2Pix details: see `PIX2PIX_README.md`

Stack: Python 3.10, PyTorch, AMP, PSNR/SSIM/LPIPS, Streamlit UI. Target OS: Windows 10/11 with NVIDIA GPUs.

## Contribution highlights

- U-Net serves as the baseline model for SAR→RGB colorization.
- Pix2Pix (cGAN) with U-Net warmstart provides enhanced sharpness and realism.
- Planned: side-by-side qualitative comparisons and quantitative tables (PSNR/SSIM/LPIPS).

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

## Environment setup (shared)

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

## Model summaries

- U-Net (baseline): direct pixel regression; fast and stable. See details and commands in `UNET_README.md`.
- Pix2Pix (cGAN): adversarially trained for sharper, more realistic results; supports warm-start from U-Net. See `PIX2PIX_README.md`.

## Tips for RTX 4060 / GPU utilization
- Use larger `--batch-size` until VRAM is near full (watch `nvidia-smi`).
- AMP is enabled by default; keep it on for speed and memory savings.
- cuDNN benchmark is enabled; keep fixed input size (e.g., 256×256).
- Increase `--num-workers` if dataloading is the bottleneck.

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

The layout below reflects the current repository folders and key files. Folders like `results/` and `checkpoints/` are gitignored but shown for clarity.

```
├── PIX2PIX_README.md
├── README.md
├── UNET_README.md
├── requirements.txt
├── checkpoints/                      # (gitignored)
│   ├── pix2pix/
│   │   └── Place_Pix2Pix_Epoch_Checkpoint_Here
│   └── unet/
│       └── Place_UNet_Epoch_Checkpoint_Here
├── configs/
│   ├── pix2pix_batch16.yaml
│   ├── pix2pix_default.yaml
│   ├── pix2pix_l1_200.yaml
│   ├── pix2pix_l1_50.yaml
│   ├── pix2pix_lr_1e4.yaml
│   ├── pix2pix_perceptual_5.yaml
│   ├── pix2pix_perceptual.yaml
│   ├── pix2pix_spectral_norm.yaml
│   └── pix2pix_warmstart.yaml
├── datasets/
│   ├── Place_Dataset_Here_(v_2)/
│   └── v_2/
│       ├── agri/
│       │   ├── s1/
│       │   └── s2/
│       ├── barrenland/
│       │   ├── s1/
│       │   └── s2/
│       ├── grassland/
│       │   ├── s1/
│       │   ├── s2/
│       │   ├── TEMP/
│       │   └── TEMP.zip
│       └── urban/
│           ├── s1/
│           └── s2/
├── models/
│   ├── pix2pix.py
│   ├── unet.py
│   └── __pycache__/               # (gitignored)
├── results/                        # (gitignored)
│   ├── pix2pix/
│   │   ├── pix2pix_20250920_005609/
│   │   │   ├── config.yaml
│   │   │   ├── training.log
│   │   │   ├── checkpoints/
│   │   │   ├── plots/
│   │   │   ├── tensorboard/
│   │   │   └── tensorboard_exports/
│   │   └── warmstart/
│   │       ├── config.yaml
│   │       ├── training.log
│   │       ├── checkpoints/
│   │       └── tensorboard/
│   └── unet/
│       ├── metrics_by_epoch.csv
│       ├── training_log.csv
│       ├── val_metrics.csv
│       ├── checkpoints/
│       │   ├── epoch_10.pt
│       │   ├── epoch_11.pt
│       │   ├── epoch_12.pt
│       │   ├── ...               # many epoch_*.pt
│       │   └── epoch_100.pt
│       ├── plots/
│       └── samples/
├── scripts/
│   ├── eval_pix2pix.py
│   ├── eval.py
│   ├── generate_samples.py
│   ├── gpu.py
│   ├── metrics_across_checkpoints.py
│   ├── pix2pix_metric_export_from_tensorboard
│   ├── plot_metrics_pix2pix.py
│   ├── plot_metrics.py
│   ├── run_pix2pix_experiments.py
│   ├── train_config.py
│   ├── train_pix2pix.py
│   ├── train.py
│   └── ui.py
└── src/
  ├── __init__.py
  ├── data_loader.py
  ├── metrics.py
  └── __pycache__/               # (gitignored)
```

- Folders `results/` and `checkpoints/` are gitignored to keep the repository lightweight.
- Example checkpoint names (e.g., `epoch_10.pt`, `epoch_100.pt`) are shown to illustrate typical contents.

## Results (Placeholder)

TODO: Insert qualitative (images) and quantitative (metrics table) results comparing U-Net vs Pix2Pix (with and without warmstart).

## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation
- Isola et al., Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)