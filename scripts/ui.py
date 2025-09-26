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
from typing import Any, Tuple, Sequence


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


def resolve_ckpt(ckpt_path: str, secret_key: str | Sequence[str]) -> str:
    """Resolve a checkpoint path.

    Order of operations:
      1. If ckpt_path exists locally -> return it.
      2. Else, try one or more secret keys (string or list of strings). For the first
         secret that exists in st.secrets, download to the desired local path and return it.
      3. If none succeed, raise FileNotFoundError.

    This allows using separate secrets for baseline vs warmstart Pix2Pix (e.g.,
    pix2pix_ckpt_url, pix2pix_warmstart_ckpt_url) while sharing the same function.
    """
    target = project_path(ckpt_path)
    if target.exists():
        return str(target)
    keys: Sequence[str] = [secret_key] if isinstance(secret_key, str) else secret_key
    for k in keys:
        url = st.secrets.get(k)
        if url:
            target.parent.mkdir(parents=True, exist_ok=True)
            with st.spinner(f"Downloading checkpoint from Hugging Face ({k})..."):
                urllib.request.urlretrieve(url, str(target))
            return str(target)
    raise FileNotFoundError(
        f"Checkpoint '{ckpt_path}' not found locally and none of the provided secrets ({', '.join(keys)}) are configured.")


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
        # Decide secret key based on path hint (warmstart vs baseline)
        path_norm = ckpt_path.replace('\\', '/').lower()
        # If ambiguous, pass both keys so either baseline or warmstart secret can satisfy the request
        if "warmstart" in path_norm:
            resolved = resolve_ckpt(ckpt_path, ["pix2pix_warmstart_ckpt_url", "pix2pix_ckpt_url"])  # prefer warmstart
        else:
            resolved = resolve_ckpt(ckpt_path, ["pix2pix_ckpt_url", "pix2pix_warmstart_ckpt_url"])  # prefer baseline
        pred_t, rgb = p2p_infer_single(Path(resolved), sar_pil, image_size=image_size, device=device)
        if gt_rgb is not None:
            metrics = p2p_compute_metrics(pred_t, gt_rgb, image_size=image_size, device=device)
        return rgb, metrics
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# --- Sidebar controls
st.sidebar.header('Model Checkpoints')
default_unet_ckpt = project_path('results/unet/checkpoints/epoch_100.pt')
default_p2p_baseline_ckpt = project_path('results/pix2pix/pix2pix_20250920_005609/checkpoints/best_by_val_lpips.pt')
default_p2p_warm_ckpt = project_path('results/pix2pix/warmstart/checkpoints/best_by_val_lpips.pt')

unet_ckpt = st.sidebar.text_input('UNet checkpoint path', str(default_unet_ckpt))
pix2pix_baseline_ckpt = st.sidebar.text_input('Pix2Pix (baseline) checkpoint path', str(default_p2p_baseline_ckpt))
pix2pix_warm_ckpt = st.sidebar.text_input('Pix2Pix (warmstart) checkpoint path', str(default_p2p_warm_ckpt))

# Optional prefetch button for all three (downloads if missing)
if st.sidebar.button('Prefetch (download) checkpoints'):
    prefetch_msgs = []
    for path, keys in [
        (unet_ckpt, ["unet_ckpt_url"]),
        (pix2pix_baseline_ckpt, ["pix2pix_ckpt_url", "pix2pix_warmstart_ckpt_url"]),
        (pix2pix_warm_ckpt, ["pix2pix_warmstart_ckpt_url", "pix2pix_ckpt_url"]),
    ]:
        try:
            resolved = resolve_ckpt(path, keys)
            prefetch_msgs.append(f"✔ {Path(resolved).name}")
        except Exception as e:
            prefetch_msgs.append(f"✖ {Path(path).name}: {e}")
    st.sidebar.write('\n'.join(prefetch_msgs))

image_size = st.sidebar.number_input('Image size', min_value=64, max_value=1024, value=256, step=32)


######### Unified Page (Inputs + Inference + Metrics) #########
st.header('Upload & Inference')
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_sar = st.file_uploader('SAR image (PNG/JPG/TIF)', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='sar_upload')
    st.caption('Converted to 1-channel (L), normalized to [-1,1].')
with col_up2:
    uploaded_rgb = st.file_uploader('Optional Ground Truth RGB (PNG/JPG)', type=['png', 'jpg', 'jpeg'], key='rgb_upload')
    st.caption('Provide to enable PSNR / SSIM / LPIPS / L1 metrics.')

if not uploaded_sar:
    st.info('Upload a SAR image to run inference.')
else:
    pred_images = {}
    base_t, norm_sar, norm_rgb = build_transforms(image_size)
    sar_img = Image.open(uploaded_sar).convert('L')
    has_gt = uploaded_rgb is not None
    # Columns: Input | (GT) | UNet | Pix2Pix Baseline | Pix2Pix Warmstart
    num_cols = 5 if has_gt else 4
    cols = st.columns(num_cols)
    with cols[0]:
        st.image(sar_img.resize((image_size, image_size)), caption='Input SAR', use_container_width=True)
    if has_gt:
        try:
            gt_img_disp = Image.open(uploaded_rgb).convert('RGB').resize((image_size, image_size))
            with cols[1]:
                st.image(gt_img_disp, caption='Ground Truth RGB', use_container_width=True)
        except Exception as e:
            with cols[1]:
                st.error(f'GT load error: {e}')
    unet_col_idx = 2 if has_gt else 1
    p2p_base_col_idx = unet_col_idx + 1
    p2p_warm_col_idx = unet_col_idx + 2

    # UNet inference
    unet_ok = False
    if unet_ckpt and Path(unet_ckpt).suffix in {'.pt', '.pth'}:
        try:
            with st.spinner('UNet inference...'):
                pred_t_unet, img_unet = run_unet_inference(unet_ckpt, sar_img, image_size)
                pred_images['UNet'] = pred_t_unet
            with cols[unet_col_idx]:
                st.image(img_unet, caption='UNet Output', use_container_width=True)
            unet_ok = True
        except Exception as e:
            with cols[unet_col_idx]:
                st.error(f'UNet failed: {e}')
    else:
        with cols[unet_col_idx]:
            st.info('Set a valid UNet .pt/.pth path.')

    # Pix2Pix baseline inference
    if pix2pix_baseline_ckpt and Path(pix2pix_baseline_ckpt).suffix in {'.pt', '.pth'}:
        try:
            with st.spinner('Pix2Pix (baseline) inference...'):
                img_p2p_base, m_p2p_base = run_inference('pix2pix', pix2pix_baseline_ckpt, sar_img, image_size, gt_rgb=Image.open(uploaded_rgb).convert('RGB') if has_gt else None)
            with cols[p2p_base_col_idx]:
                st.image(img_p2p_base, caption='Pix2Pix Baseline', use_container_width=True)
            if m_p2p_base is not None:
                st.session_state.setdefault('metrics_by_model', {})['Pix2Pix Baseline'] = m_p2p_base
        except Exception as e:
            with cols[p2p_base_col_idx]:
                st.error(f'Pix2Pix Baseline failed: {e}')
    else:
        with cols[p2p_base_col_idx]:
            st.info('Set baseline Pix2Pix .pt/.pth path.')

    # Pix2Pix warmstart inference
    if pix2pix_warm_ckpt and Path(pix2pix_warm_ckpt).suffix in {'.pt', '.pth'}:
        try:
            with st.spinner('Pix2Pix (warmstart) inference...'):
                img_p2p_warm, m_p2p_warm = run_inference('pix2pix', pix2pix_warm_ckpt, sar_img, image_size, gt_rgb=Image.open(uploaded_rgb).convert('RGB') if has_gt else None)
            with cols[p2p_warm_col_idx]:
                st.image(img_p2p_warm, caption='Pix2Pix Warmstart', use_container_width=True)
            if m_p2p_warm is not None:
                st.session_state.setdefault('metrics_by_model', {})['Pix2Pix Warmstart'] = m_p2p_warm
        except Exception as e:
            with cols[p2p_warm_col_idx]:
                st.error(f'Pix2Pix Warmstart failed: {e}')
    else:
        with cols[p2p_warm_col_idx]:
            st.info('Set warmstart Pix2Pix .pt/.pth path.')

    # Compute UNet metrics if GT available
    if unet_ok and has_gt:
        try:
            evaluator = MetricsEvaluator(device)
            target_rgb = Image.open(uploaded_rgb).convert('RGB')
            _, _, norm_rgb_local = build_transforms(image_size)
            t = norm_rgb_local(base_t(target_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                m = evaluator.evaluate_batch(pred_images['UNet'].to(device), t)
                m['l1'] = F.l1_loss(pred_images['UNet'].to(device), t).item()
            st.session_state.setdefault('metrics_by_model', {})['UNet'] = m
        except Exception:
            pass

    # Metrics section
    st.subheader('Per-image Metrics')
    if not has_gt:
        st.info('Provide a ground truth RGB image to compute metrics.')
    else:
        metrics_cached = st.session_state.get('metrics_by_model')
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
        if rows:
            df = pd.DataFrame(rows).set_index('Model')
            st.dataframe(df.style.format({'PSNR': '{:.3f}', 'SSIM': '{:.3f}', 'LPIPS (↓)': '{:.4f}', 'L1 (↓)': '{:.4f}'}), width="stretch")
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown('PSNR / SSIM (higher better)')
                df_higher = df[['PSNR', 'SSIM']].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')
                fig1 = px.bar(df_higher, x='Model', y='Value', color='Metric', barmode='group', title=None)
                st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})
            with chart_cols[1]:
                st.markdown('LPIPS / L1 (lower better)')
                df_lower = df[['LPIPS (↓)', 'L1 (↓)']].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')
                fig2 = px.bar(df_lower, x='Model', y='Value', color='Metric', barmode='group', title=None)
                st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
        else:
            st.info('Run inference with checkpoints to populate metrics.')

st.caption('Tip: Run with: uv run streamlit run scripts/ui.py')

