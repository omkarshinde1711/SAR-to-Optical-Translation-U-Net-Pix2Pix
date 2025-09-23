import sys
import os
import yaml
import argparse
import csv
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import torch.cuda.amp as amp

from src.data_loader import Sentinel
from models.pix2pix import create_pix2pix_model, VGGPerceptualLoss
from src.metrics import MetricsEvaluator


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model for SAR to optical translation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_data_loaders(config, data_root):
    """Create training and validation data loaders"""
    # Transforms
    base_transform = v2.Compose([
        v2.Resize((config['image_size'], config['image_size'])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    # Normalization
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Datasets
    train_dataset = Sentinel(
        root_dir=data_root,
        split_type='train',
        transform=base_transform,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05)
    )
    
    val_dataset = Sentinel(
        root_dir=data_root,
        split_type='val',
        transform=base_transform,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05)
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, norm_sar, norm_rgb


def create_optimizers(model, config):
    """Create optimizers for generator and discriminator"""
    g_optimizer = optim.Adam(
        model.generator.parameters(),
        lr=config['lr'],
        betas=config['betas']
    )
    
    d_optimizer = optim.Adam(
        model.discriminator.parameters(),
        lr=config['lr'],
        betas=config['betas']
    )
    
    return g_optimizer, d_optimizer


def create_schedulers(g_optimizer, d_optimizer, config, train_loader):
    """Create learning rate schedulers"""
    if config['scheduler'] == 'linear_decay':
        g_scheduler = optim.lr_scheduler.LinearLR(
            g_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config['epochs']
        )
        d_scheduler = optim.lr_scheduler.LinearLR(
            d_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config['epochs']
        )
    elif config['scheduler'] == 'plateau':
        g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        g_scheduler = None
        d_scheduler = None
    
    return g_scheduler, d_scheduler


def compute_generator_loss(pred, target, fake_logits, config, perceptual_loss_fn=None):
    """Compute generator loss"""
    # Adversarial loss
    if config['adv_loss'] == 'bce':
        adv_loss = nn.BCEWithLogitsLoss()(fake_logits, torch.ones_like(fake_logits))
    elif config['adv_loss'] == 'hinge':
        adv_loss = -torch.mean(fake_logits)
    else:
        raise ValueError(f"Unknown adversarial loss: {config['adv_loss']}")
    
    # Reconstruction loss (L1)
    recon_loss = nn.L1Loss()(pred, target)
    
    # Total loss
    total_loss = adv_loss + config['lambda_l1'] * recon_loss
    
    # Perceptual loss (optional)
    if config.get('use_perceptual_loss', False) and perceptual_loss_fn is not None:
        perc_loss = perceptual_loss_fn(pred, target)
        total_loss += config['lambda_perc'] * perc_loss
        return total_loss, adv_loss, recon_loss, perc_loss
    
    return total_loss, adv_loss, recon_loss, None


def compute_discriminator_loss(real_logits, fake_logits, config):
    """Compute discriminator loss"""
    if config['adv_loss'] == 'bce':
        real_loss = nn.BCEWithLogitsLoss()(real_logits, torch.ones_like(real_logits))
        fake_loss = nn.BCEWithLogitsLoss()(fake_logits, torch.zeros_like(fake_logits))
    elif config['adv_loss'] == 'hinge':
        real_loss = -torch.mean(torch.min(real_logits - 1, torch.zeros_like(real_logits)))
        fake_loss = -torch.mean(torch.min(-fake_logits - 1, torch.zeros_like(fake_logits)))
    else:
        raise ValueError(f"Unknown adversarial loss: {config['adv_loss']}")
    
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss, real_loss, fake_loss


def train_epoch(model, train_loader, g_optimizer, d_optimizer, norm_sar, norm_rgb,
                config, device, scaler, perceptual_loss_fn, metrics_evaluator, logger):
    """Train for one epoch"""
    model.train()
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_perc_loss = 0.0
    epoch_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    num_batches = len(train_loader)
    
    for batch_idx, (sar, rgb) in enumerate(train_loader):
        sar = norm_sar(sar.to(device, non_blocking=True))
        rgb = norm_rgb(rgb.to(device, non_blocking=True))
        
        batch_size = sar.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        with amp.autocast():
            # Real samples
            real_logits = model.discriminate(sar, rgb)
            
            # Fake samples
            with torch.no_grad():
                fake_rgb = model.generator(sar)
            fake_logits = model.discriminate(sar, fake_rgb.detach())
            
            d_loss, d_real_loss, d_fake_loss = compute_discriminator_loss(
                real_logits, fake_logits, config
            )
        
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)
        
        # Train Generator
        g_optimizer.zero_grad()
        
        with amp.autocast():
            fake_rgb = model.generator(sar)
            fake_logits = model.discriminate(sar, fake_rgb)
            
            g_loss, adv_loss, recon_loss, perc_loss = compute_generator_loss(
                fake_rgb, rgb, fake_logits, config, perceptual_loss_fn
            )
        
        scaler.scale(g_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()
        
        # Update metrics
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_adv_loss += adv_loss.item()
        epoch_recon_loss += recon_loss.item()
        if perc_loss is not None:
            epoch_perc_loss += perc_loss.item()
        
        # Compute metrics on generated images
        with torch.no_grad():
            batch_metrics = metrics_evaluator.evaluate_batch(fake_rgb, rgb)
            for key in epoch_metrics:
                val = batch_metrics[key]
                if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                    epoch_metrics[key].extend(val)
                else:
                    epoch_metrics[key].append(float(val))
        
        # Log progress
        if batch_idx % config.get('log_interval', 100) == 0:
            logger.info(
                f'Batch {batch_idx}/{num_batches} - '
                f'G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}, '
                f'Adv: {adv_loss.item():.4f}, Recon: {recon_loss.item():.4f}'
            )
    
    # Average metrics
    avg_metrics = {}
    for key in epoch_metrics:
        avg_metrics[key] = sum(epoch_metrics[key]) / len(epoch_metrics[key]) if epoch_metrics[key] else 0.0
    
    return {
        'g_loss': epoch_g_loss / num_batches,
        'd_loss': epoch_d_loss / num_batches,
        'adv_loss': epoch_adv_loss / num_batches,
        'recon_loss': epoch_recon_loss / num_batches,
        'perc_loss': epoch_perc_loss / num_batches if config.get('use_perceptual_loss', False) else 0.0,
        **avg_metrics
    }


def validate_epoch(model, val_loader, norm_sar, norm_rgb, config, device, 
                  perceptual_loss_fn, metrics_evaluator):
    """Validate for one epoch"""
    model.eval()
    
    val_g_loss = 0.0
    val_d_loss = 0.0
    val_adv_loss = 0.0
    val_recon_loss = 0.0
    val_perc_loss = 0.0
    val_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    with torch.no_grad():
        for sar, rgb in val_loader:
            sar = norm_sar(sar.to(device, non_blocking=True))
            rgb = norm_rgb(rgb.to(device, non_blocking=True))
            
            # Generator forward pass
            fake_rgb = model.generator(sar)
            fake_logits = model.discriminate(sar, fake_rgb)
            
            # Compute losses
            g_loss, adv_loss, recon_loss, perc_loss = compute_generator_loss(
                fake_rgb, rgb, fake_logits, config, perceptual_loss_fn
            )
            
            # Discriminator losses
            real_logits = model.discriminate(sar, rgb)
            d_loss, _, _ = compute_discriminator_loss(real_logits, fake_logits, config)
            
            # Update metrics
            val_g_loss += g_loss.item()
            val_d_loss += d_loss.item()
            val_adv_loss += adv_loss.item()
            val_recon_loss += recon_loss.item()
            if perc_loss is not None:
                val_perc_loss += perc_loss.item()
            
            # Compute metrics
            batch_metrics = metrics_evaluator.evaluate_batch(fake_rgb, rgb)
            for key in val_metrics:
                val = batch_metrics[key]
                if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                    val_metrics[key].extend(val)
                else:
                    val_metrics[key].append(float(val))
    
    num_batches = len(val_loader)
    avg_metrics = {}
    for key in val_metrics:
        avg_metrics[key] = sum(val_metrics[key]) / len(val_metrics[key]) if val_metrics[key] else 0.0
    
    return {
        'val_g_loss': val_g_loss / num_batches,
        'val_d_loss': val_d_loss / num_batches,
        'val_adv_loss': val_adv_loss / num_batches,
        'val_recon_loss': val_recon_loss / num_batches,
        'val_perc_loss': val_perc_loss / num_batches if config.get('use_perceptual_loss', False) else 0.0,
        **{f'val_{k}': v for k, v in avg_metrics.items()}
    }


def save_checkpoint(model, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                   epoch, metrics, config, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_scheduler_state_dict': g_scheduler.state_dict() if g_scheduler else None,
        'd_scheduler_state_dict': d_scheduler.state_dict() if d_scheduler else None,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f'best_{save_path.stem}.pt'
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer, g_scheduler, d_scheduler):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    if g_scheduler and checkpoint['g_scheduler_state_dict']:
        g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
    if d_scheduler and checkpoint['d_scheduler_state_dict']:
        d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics'], checkpoint['config']


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"pix2pix_{timestamp}"
    
    exp_dir = project_root / 'results' / 'pix2pix' / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(exp_dir)
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Save config
    save_config(config, exp_dir / 'config.yaml')
    
    # Create data loaders
    train_loader, val_loader, norm_sar, norm_rgb = create_data_loaders(config, args.data_root)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_pix2pix_model(config).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizers and schedulers
    g_optimizer, d_optimizer = create_optimizers(model, config)
    g_scheduler, d_scheduler = create_schedulers(g_optimizer, d_optimizer, config, train_loader)
    
    # Create loss functions
    perceptual_loss_fn = None
    if config.get('use_perceptual_loss', False):
        perceptual_loss_fn = VGGPerceptualLoss(device)
    
    # Create metrics evaluator
    metrics_evaluator = MetricsEvaluator(device)
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # TensorBoard writer
    writer = SummaryWriter(exp_dir / 'tensorboard')
    
    # Training state
    start_epoch = 0
    best_val_lpips = float('inf')
    best_val_l1 = float('inf')
    patience_counter = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, _, _ = load_checkpoint(
            args.resume, model, g_optimizer, d_optimizer, g_scheduler, d_scheduler
        )
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, g_optimizer, d_optimizer, norm_sar, norm_rgb,
            config, device, scaler, perceptual_loss_fn, metrics_evaluator, logger
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, norm_sar, norm_rgb, config, device,
            perceptual_loss_fn, metrics_evaluator
        )
        
        # Update schedulers
        if g_scheduler:
            if config['scheduler'] == 'plateau':
                g_scheduler.step(val_metrics['val_lpips'])
                d_scheduler.step(val_metrics['val_lpips'])
            else:
                g_scheduler.step()
                d_scheduler.step()
        
        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}
        logger.info(f"Train - G Loss: {train_metrics['g_loss']:.4f}, D Loss: {train_metrics['d_loss']:.4f}")
        logger.info(f"Val - G Loss: {val_metrics['val_g_loss']:.4f}, D Loss: {val_metrics['val_d_loss']:.4f}")
        logger.info(f"Val - PSNR: {val_metrics['val_psnr']:.3f}, SSIM: {val_metrics['val_ssim']:.3f}, LPIPS: {val_metrics['val_lpips']:.3f}")
        
        # TensorBoard logging
        for key, value in all_metrics.items():
            writer.add_scalar(key, value, epoch)
        
        # Save checkpoint
        checkpoint_dir = exp_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Regular checkpoint
        if (epoch + 1) % config.get('checkpoint_interval', 5) == 0:
            save_checkpoint(
                model, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                epoch, all_metrics, config, checkpoint_dir / f'epoch_{epoch+1}.pt'
            )
        
        # Best checkpoints
        is_best_lpips = val_metrics['val_lpips'] < best_val_lpips
        is_best_l1 = val_metrics['val_recon_loss'] < best_val_l1
        
        if is_best_lpips:
            best_val_lpips = val_metrics['val_lpips']
            save_checkpoint(
                model, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                epoch, all_metrics, config, checkpoint_dir / 'best_by_val_lpips.pt', is_best=True
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        if is_best_l1:
            best_val_l1 = val_metrics['val_recon_loss']
            save_checkpoint(
                model, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                epoch, all_metrics, config, checkpoint_dir / 'best_by_val_l1.pt', is_best=True
            )
        
        # Early stopping
        if config.get('early_stopping', False) and patience_counter >= config.get('patience', 10):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final checkpoint
    save_checkpoint(
        model, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
        epoch, all_metrics, config, checkpoint_dir / 'final.pt'
    )
    
    logger.info("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()