import sys
from pathlib import Path
import argparse
import csv

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_loader import Sentinel
from src.metrics import MetricsEvaluator
from models.unet import UNet


def build_transforms(image_size: int):
    base = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    denorm_gray = lambda t: (t.clamp(-1, 1) + 1) / 2
    denorm_rgb = lambda t: (t.clamp(-1, 1) + 1) / 2
    return base, norm_sar, norm_rgb, denorm_gray, denorm_rgb


def parse_args():
    p = argparse.ArgumentParser(description='Train UNet for SAR->RGB colorization')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--save-dir', type=str, default='results/unet')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--save-interval', type=int, default=5)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'plateau'])
    p.add_argument('--step-size', type=int, default=30)
    p.add_argument('--gamma', type=float, default=0.5)
    return p.parse_args()


def run_training(
    data_root: str,
    save_dir: str = 'results/unet',
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 8,
    lr: float = 2e-4,
    save_interval: int = 5,
    resume: str = '',
    scheduler: str = 'none',
    step_size: int = 30,
    gamma: float = 0.5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Resolve paths relative to project root if not absolute
    save_dir_path = Path(save_dir)
    save_path = save_dir_path if save_dir_path.is_absolute() else (project_root / save_dir_path)
    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (save_path / 'samples').mkdir(parents=True, exist_ok=True)

    base_t, norm_sar, norm_rgb, denorm_gray, denorm_rgb = build_transforms(image_size)

    dataset = Sentinel(
        root_dir=data_root,
        split_type='train',
        transform=base_t,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05),
    )
    valset = Sentinel(
        root_dir=data_root,
        split_type='val',
        transform=base_t,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=1,persistent_workers=False)
    vloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=1,persistent_workers=False)

    model = UNet(in_channels=1, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    # Scheduler
    if scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
    else:
        lr_scheduler = None
    criterion_l1 = torch.nn.L1Loss()
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    metrics = MetricsEvaluator(device)

    start_epoch = 1
    if resume:
        print(f"Loading checkpoint: {resume}")
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(ckpt['scheduler'])
            except Exception:
                print("Warning: Failed to load scheduler state from checkpoint; continuing with fresh scheduler.")
        if 'scaler' in ckpt and isinstance(scaler, GradScaler):
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                print("Warning: Failed to load AMP scaler state; continuing with fresh scaler.")
        if 'epoch' in ckpt:
            resumed_epoch = int(ckpt['epoch'])
            start_epoch = resumed_epoch + 1
            print(f"Resuming training from epoch {start_epoch} (checkpoint epoch {resumed_epoch})")
        else:
            print("Warning: 'epoch' not found in checkpoint. Starting from epoch 1.")

    # If the checkpoint is already at or beyond the requested total epochs, there is nothing to train
    if start_epoch > epochs:
        print(f"No training to run: start_epoch {start_epoch} > target epochs {epochs}. Increase --epochs to continue training.")
        return

    log_csv = (save_path / 'training_log.csv')
    if not log_csv.exists():
        with log_csv.open('w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'iter', 'loss_l1'])
    val_csv = (save_path / 'val_metrics.csv')
    if not val_csv.exists():
        with val_csv.open('w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'psnr', 'ssim', 'lpips', 'val_l1'])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}/{epochs}')
        for i, (sar, rgb) in pbar:
            sar = sar.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            sar = norm_sar(sar)
            rgb = norm_rgb(rgb)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred = model(sar)
                loss = criterion_l1(pred, rgb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss_l1': f'{loss.item():.4f}'})
            with log_csv.open('a', newline='') as f:
                csv.writer(f).writerow([epoch, i, f'{loss.item():.6f}'])

        # Validation metrics
        model.eval()
        val_losses = []
        agg = {"psnr": [], "ssim": [], "lpips": []}
        with torch.no_grad():
            for sar, rgb in vloader:
                sar = norm_sar(sar.to(device, non_blocking=True))
                rgb = norm_rgb(rgb.to(device, non_blocking=True))
                pred = model(sar)
                vloss = criterion_l1(pred, rgb).item()
                val_losses.append(vloss)
                m = metrics.evaluate_batch(pred, rgb)
                for k in agg:
                    agg[k].append(m[k])
        val_l1 = sum(val_losses) / max(1, len(val_losses))
        psnr_e = sum(agg['psnr']) / max(1, len(agg['psnr']))
        ssim_e = sum(agg['ssim']) / max(1, len(agg['ssim']))
        lpips_e = sum(agg['lpips']) / max(1, len(agg['lpips']))
        with val_csv.open('a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{psnr_e:.6f}', f'{ssim_e:.6f}', f'{lpips_e:.6f}', f'{val_l1:.6f}'])

        if lr_scheduler is not None:
            if scheduler == 'plateau':
                lr_scheduler.step(val_l1)
            else:
                lr_scheduler.step()

        if epoch % save_interval == 0:
            ckpt_path = save_path / 'checkpoints' / f'epoch_{epoch}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': (lr_scheduler.state_dict() if lr_scheduler is not None else None),
                'scaler': (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
                'epoch': epoch
            }, ckpt_path)

            with torch.no_grad():
                sar, rgb = next(iter(loader))
                sar = sar.to(device)
                rgb = rgb.to(device)
                sar_n = norm_sar(sar)
                rgb_n = norm_rgb(rgb)
                pred = model(sar_n)
                pred = denorm_rgb(pred).cpu()
                sar_v = denorm_gray(sar_n).cpu()
                rgb_v = denorm_rgb(rgb_n).cpu()

                fig, axs = plt.subplots(3, 4, figsize=(12, 8))
                for j in range(4):
                    axs[0, j].imshow(sar_v[j].squeeze(), cmap='gray')
                    axs[1, j].imshow(rgb_v[j].permute(1, 2, 0).numpy())
                    axs[2, j].imshow(pred[j].permute(1, 2, 0).numpy())
                plt.tight_layout()
                plt.savefig(save_path / 'samples' / f'epoch_{epoch}.png')
                plt.close()


if __name__ == '__main__':
    args = parse_args()
    run_training(
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        lr=args.lr,
        save_interval=args.save_interval,
        resume=args.resume,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
    )

