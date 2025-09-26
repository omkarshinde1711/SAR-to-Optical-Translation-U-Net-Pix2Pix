import sys
from pathlib import Path
import os
import urllib.request

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
import pandas as pd
import plotly.express as px
from typing import Any, Tuple


# --- Project paths and imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.unet import UNet
from models.pix2pix import UNetGenerator as Pix2PixGenerator  # retained if needed
from src.metrics import MetricsEvaluator
from scripts.eval_pix2pix import infer_single_image as p2p_infer_single, compute_metrics_single as p2p_compute_metrics


# --- Page config
st.set_page_config(page_title='SAR → Optical Translator (UNet & Pix2Pix)', layout='wide')
st.title('SAR → Optical Image Translation')


# --- Device and small helpers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.success(f'Device: {device.type.upper()}')

def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p

def denorm_rgb(t: torch.Tensor) -> torch.Tensor:
    # expects [-1,1], returns [0,1]
    return (t.clamp(-1, 1) + 1) / 2


def resolve_ckpt(ckpt_path: str, secret_key: str) -> str:
    """
    Resolve a checkpoint path:
      - if exists locally -> return absolute path
      - else, download from Hugging Face URL in st.secrets[secret_key] to the intended local path
    """
    target = project_path(ckpt_path)
    if target.exists():
        return str(target)
    url = st.secrets.get(secret_key)
    if url:
        target.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner(f"Downloading checkpoint from Hugging Face ({secret_key})..."):
            urllib.request.urlretrieve(url, str(target))
        return str(target)
    raise FileNotFoundError(f"Checkpoint not found locally and no secret '{secret_key}' configured.")


# --- Transforms
def build_transforms(image_size: int):
    base_t = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return base_t, norm_sar, norm_rgb


# --- Model loaders (cached)
@st.cache_resource
def load_unet(ckpt_path: str, in_ch: int = 1, out_ch: int = 3):
    model = UNet(in_channels=in_ch, out_channels=out_ch).to(device)
    state = torch.load(project_path(ckpt_path), map_location=device)
    model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
    model.eval()
    return model


@st.cache_resource
def load_pix2pix_generator(ckpt_path: str, in_ch: int = 1, out_ch: int = 3):
    gen = Pix2PixGenerator(in_channels=in_ch, out_channels=out_ch).to(device)
    state = torch.load(project_path(ckpt_path), map_location=device)
    # Robust key handling: support pure generator state, or dicts with 'generator'/'model'
    if isinstance(state, dict):
        if 'generator' in state and isinstance(state['generator'], dict):
            gen.load_state_dict(state['generator'])
        elif 'model' in state and isinstance(state['model'], dict):
            # If UNet-style checkpoint provided for warm-started generator
            try:
                gen.load_state_dict(state['model'])
            except Exception:
                # try partial load
                gen.load_state_dict({k: v for k, v in state['model'].items() if k in gen.state_dict() and gen.state_dict()[k].shape == v.shape}, strict=False)
        else:
            # try loading as-is
            try:
                gen.load_state_dict(state)
            except Exception:
                gen.load_state_dict({k: v for k, v in state.items() if k in gen.state_dict() and gen.state_dict()[k].shape == v.shape}, strict=False)
    else:
        gen.load_state_dict(state)
    gen.eval()
    return gen


# --- Inference utility
@torch.no_grad()
def run_unet_inference(ckpt_path: str, sar_pil: Image.Image, image_size: int) -> Tuple[torch.Tensor, Any]:
    import numpy as np
    base_t, norm_sar, _ = build_transforms(image_size)
    x = norm_sar(base_t(sar_pil.convert('L'))).unsqueeze(0).to(device)
    resolved = resolve_ckpt(ckpt_path, "unet_ckpt_url")
    model = load_unet(resolved)
    y = model(x)
    y01 = denorm_rgb(y).squeeze(0).cpu()
    rgb = (y01.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return y.detach().cpu(), rgb


def run_inference(model_name: str, ckpt_path: str, sar_pil: Image.Image, image_size: int, gt_rgb: Image.Image | None = None):
    """
    Unified inference wrapper.
    Returns (rgb_uint8, metrics_dict or None)
    """
    metrics = None
    if model_name.lower() == 'unet':
        pred_t, rgb = run_unet_inference(ckpt_path, sar_pil, image_size)
        if gt_rgb is not None:
            evaluator = MetricsEvaluator(device)
            # build target
            base_t, _, norm_rgb = build_transforms(image_size)
            t = norm_rgb(base_t(gt_rgb.convert('RGB'))).unsqueeze(0).to(device)
            with torch.no_grad():
                m = evaluator.evaluate_batch(pred_t.to(device), t)
                m['l1'] = F.l1_loss(pred_t.to(device), t).item()
            metrics = m
        return rgb, metrics
    elif model_name.lower() == 'pix2pix':
        resolved = resolve_ckpt(ckpt_path, "pix2pix_ckpt_url")
        pred_t, rgb = p2p_infer_single(Path(resolved), sar_pil, image_size=image_size, device=device)
        if gt_rgb is not None:
            metrics = p2p_compute_metrics(pred_t, gt_rgb, image_size=image_size, device=device)
        return rgb, metrics
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# --- Sidebar controls
st.sidebar.header('Model Checkpoints')
default_unet_ckpt = project_path('results/unet/checkpoints/epoch_100.pt')
default_p2p_ckpt = project_path('results/pix2pix/warmstart/checkpoints/best_by_val_lpips.pt')

unet_ckpt = st.sidebar.text_input('UNet checkpoint path', str(default_unet_ckpt))
pix2pix_ckpt = st.sidebar.text_input('Pix2Pix generator checkpoint path', str(default_p2p_ckpt))

image_size = st.sidebar.number_input('Image size', min_value=64, max_value=1024, value=256, step=32)


# --- Tabs layout
tabs = st.tabs(["Inputs", "Inference", "Metrics", "Training Logs"])


# --- Inputs tab
with tabs[0]:
    st.subheader('Upload Images')
    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        uploaded_sar = st.file_uploader('Upload SAR image (PNG/JPG/TIF)', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='sar_upload')
        st.caption('SAR will be converted to single-channel (L), normalized to [-1,1].')
    with col_in2:
        uploaded_rgb = st.file_uploader('Optional: Upload Ground Truth RGB (PNG/JPG)', type=['png', 'jpg', 'jpeg'], key='rgb_upload')
        st.caption('Provide to compute PSNR/SSIM/LPIPS/L1. If omitted, metrics are not computed.')

    if uploaded_sar:
        try:
            img_sar_disp = Image.open(uploaded_sar).convert('L')
            st.image(img_sar_disp, caption='Uploaded SAR (grayscale)', width="stretch")
        except Exception as e:
            st.error(f'Failed to read SAR image: {e}')
    else:
        st.info('Please upload a SAR image to proceed.')


# --- Inference tab
pred_images = {}
with tabs[1]:
    st.subheader('Model Inference')
    if not uploaded_sar:
        st.warning('Upload a SAR image in the Inputs tab.')
    else:
        base_t, norm_sar, norm_rgb = build_transforms(image_size)
        sar_img = Image.open(uploaded_sar).convert('L')
        cols = st.columns(3)
        with cols[0]:
            st.image(sar_img.resize((image_size, image_size)), caption='Input SAR', width="stretch")

        # UNet
        unet_ok = False
        if unet_ckpt and Path(unet_ckpt).suffix in {'.pt', '.pth'}:
            try:
                with st.spinner('Running UNet inference...'):
                    pred_t_unet, img_unet = run_unet_inference(unet_ckpt, sar_img, image_size)
                    pred_images['UNet'] = pred_t_unet
                with cols[1]:
                    st.image(img_unet, caption='UNet Output', width="stretch")
                unet_ok = True
            except Exception as e:
                with cols[1]:
                    st.error(f'UNet failed: {e}')
        else:
            with cols[1]:
                st.info('Provide a valid UNet checkpoint path ending with .pt/.pth.')

        # Pix2Pix
        p2p_ok = False
        if pix2pix_ckpt and Path(pix2pix_ckpt).suffix in {'.pt', '.pth'}:
            try:
                with st.spinner('Running Pix2Pix inference (eval script)...'):
                    # Use standardized wrapper which calls eval helpers
                    img_p2p, m_p2p = run_inference('pix2pix', pix2pix_ckpt, sar_img, image_size, gt_rgb=Image.open(uploaded_rgb).convert('RGB') if 'uploaded_rgb' in locals() and uploaded_rgb else None)
                with cols[2]:
                    st.image(img_p2p, caption='Pix2Pix Output', width="stretch")
                p2p_ok = True
                # Save metrics if available
                if m_p2p is not None:
                    st.session_state.setdefault('metrics_by_model', {})['Pix2Pix'] = m_p2p
            except Exception as e:
                with cols[2]:
                    st.error(f'Pix2Pix failed: {e}')
        else:
            with cols[2]:
                st.info('Provide a valid Pix2Pix checkpoint path ending with .pt/.pth.')

        # Optionally compute and store UNet metrics now if GT provided
        if unet_ok and uploaded_rgb is not None:
            try:
                evaluator = MetricsEvaluator(device)
                base_t, _, norm_rgb = build_transforms(image_size)
                t = norm_rgb(base_t(Image.open(uploaded_rgb).convert('RGB'))).unsqueeze(0).to(device)
                with torch.no_grad():
                    m = evaluator.evaluate_batch(pred_images['UNet'].to(device), t)
                    m['l1'] = F.l1_loss(pred_images['UNet'].to(device), t).item()
                st.session_state.setdefault('metrics_by_model', {})['UNet'] = m
            except Exception:
                pass


# --- Metrics tab
with tabs[2]:
    st.subheader('Per-image Metrics')
    if not uploaded_sar:
        st.info('Upload a SAR image to compute metrics.')
    else:
        metrics_cached = st.session_state.get('metrics_by_model')
        if not pred_images and not metrics_cached:
            st.info('Run inference first (Inference tab).')
        else:
            if uploaded_rgb is None:
                st.warning('Ground truth RGB not provided. Metrics require a target image.')
            else:
                rows = []
                if metrics_cached:
                    for name, m in metrics_cached.items():
                        rows.append({
                            'Model': name,
                            'PSNR': m.get('psnr', float('nan')),
                            'SSIM': m.get('ssim', float('nan')),
                            'LPIPS (↓)': m.get('lpips', float('nan')),
                            'L1 (↓)': m.get('l1', float('nan')),
                        })
                else:
                    # Fallback: compute on the fly using predictions
                    base_t, _, norm_rgb = build_transforms(image_size)
                    target_rgb = Image.open(uploaded_rgb).convert('RGB')
                    t = norm_rgb(base_t(target_rgb)).unsqueeze(0).to(device)
                    for name, pred in pred_images.items():
                        with torch.no_grad():
                            if name.lower() == 'pix2pix':
                                m = p2p_compute_metrics(pred, target_rgb, image_size=image_size, device=device)
                            else:
                                evaluator = MetricsEvaluator(device)
                                m = evaluator.evaluate_batch(pred.to(device), t)
                                m['l1'] = F.l1_loss(pred.to(device), t).item()
                        rows.append({
                            'Model': name,
                            'PSNR': m['psnr'],
                            'SSIM': m['ssim'],
                            'LPIPS (↓)': m['lpips'],
                            'L1 (↓)': m['l1'],
                        })
                df = pd.DataFrame(rows).set_index('Model')
                st.dataframe(df.style.format({'PSNR': '{:.3f}', 'SSIM': '{:.3f}', 'LPIPS (↓)': '{:.4f}', 'L1 (↓)': '{:.4f}'}), width="stretch")

                # Comparison charts
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    st.markdown('Comparison: PSNR / SSIM (higher is better)')
                    df_higher = df[['PSNR', 'SSIM']].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')
                    fig1 = px.bar(df_higher, x='Model', y='Value', color='Metric', barmode='group', title=None)
                    st.plotly_chart(fig1, width="stretch")
                with chart_cols[1]:
                    st.markdown('Comparison: LPIPS / L1 (lower is better)')
                    df_lower = df[['LPIPS (↓)', 'L1 (↓)']].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')
                    fig2 = px.bar(df_lower, x='Model', y='Value', color='Metric', barmode='group', title=None)
                    st.plotly_chart(fig2, width="stretch")


# --- Training Logs tab
with tabs[3]:
    st.subheader('Metrics across epochs (from logs)')
    log_cols = st.columns(2)

    # UNet logs
    with log_cols[0]:
        st.markdown('UNet validation metrics')
        unet_val_csv = project_path('results/unet/val_metrics.csv')
        if unet_val_csv.exists():
            try:
                df_u = pd.read_csv(unet_val_csv)
                # Expected columns: epoch, val_l1, psnr, ssim, lpips (some names may vary)
                # Normalize column names
                cols = {c.lower(): c for c in df_u.columns}
                # Map possible variants
                epoch_col = cols.get('epoch', 'epoch')
                psnr_col = cols.get('psnr', 'psnr')
                ssim_col = cols.get('ssim', 'ssim')
                lpips_col = cols.get('lpips', 'lpips') if 'lpips' in cols else None
                l1_col = cols.get('val_l1', 'val_l1') if 'val_l1' in cols else (cols.get('l1') if 'l1' in cols else None)

                plots = []
                for metric_name, col in [('PSNR', psnr_col), ('SSIM', ssim_col)]:
                    if col in df_u.columns:
                        fig = px.line(df_u, x=epoch_col, y=col, title=f'UNet {metric_name} vs Epoch')
                        st.plotly_chart(fig, width="stretch")
                if lpips_col and lpips_col in df_u.columns:
                    fig = px.line(df_u, x=epoch_col, y=lpips_col, title='UNet LPIPS vs Epoch')
                    st.plotly_chart(fig, width="stretch")
                if l1_col and l1_col in df_u.columns:
                    fig = px.line(df_u, x=epoch_col, y=l1_col, title='UNet L1 vs Epoch')
                    st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f'Failed to read UNet logs: {e}')
        else:
            st.info('No UNet val_metrics.csv found at results/unet/val_metrics.csv')

    # Pix2Pix logs
    with log_cols[1]:
        st.markdown('Pix2Pix validation metrics')
        # Try to discover a CSV under results/pix2pix/**
        pix2pix_root = project_path('results/pix2pix')
        found_csv = None
        if pix2pix_root.exists():
            for p in sorted(pix2pix_root.rglob('*.csv')):
                # heuristics: pick a csv that has typical metric columns
                try:
                    df_tmp = pd.read_csv(p, nrows=1)
                    cols = set(c.lower() for c in df_tmp.columns)
                    if {'epoch'} & cols and ({'psnr', 'ssim'} & cols or {'val_psnr', 'val_ssim'} & cols or {'lpips', 'val_lpips'} & cols):
                        found_csv = p
                        break
                except Exception:
                    continue
        if found_csv:
            st.caption(f'Using: {found_csv.relative_to(PROJECT_ROOT)}')
            try:
                df_p = pd.read_csv(found_csv)
                cols = {c.lower(): c for c in df_p.columns}
                epoch_col = cols.get('epoch', 'epoch')
                for name_variant in [('PSNR', ('psnr', 'val_psnr')), ('SSIM', ('ssim', 'val_ssim')), ('LPIPS', ('lpips', 'val_lpips')), ('L1', ('l1', 'val_l1'))]:
                    label, choices = name_variant
                    for ch in choices:
                        if ch in cols:
                            colname = cols[ch]
                            fig = px.line(df_p, x=epoch_col, y=colname, title=f'Pix2Pix {label} vs Epoch')
                            st.plotly_chart(fig, width="stretch")
                            break
            except Exception as e:
                st.error(f'Failed to read Pix2Pix logs: {e}')
        else:
            st.info('No Pix2Pix metrics CSV discovered under results/pix2pix/.')

    with st.expander('Generate fresh plots via plotter (optional)'):
        st.caption('This uses scripts/plot_metrics.py for UNet plots and will overwrite images in results/unet/plots.')
        if st.button('Run plotter'):
            try:
                # Best-effort import and run main from plot_metrics
                from scripts.plot_metrics import main as plot_main
                plot_main()
                st.success('Plotting completed. Check results/unet/plots for images.')
            except Exception as e:
                st.error(f'Plotting failed: {e}')


st.caption('Tip: Run with uv: uv run streamlit run scripts/ui.py')

