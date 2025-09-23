import sys
import argparse
import csv
import glob
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.utils as vutils

from src.data_loader import Sentinel
from models.pix2pix import create_pix2pix_model
from src.metrics import MetricsEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Pix2Pix model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-samples', action='store_true', help='Save sample images')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of sample images to save')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint if available
    config = checkpoint.get('config', {})
    
    return checkpoint, config


def create_data_loader(data_root, split, batch_size, num_workers, image_size=256):
    """Create data loader for evaluation"""
    # Transforms
    base_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    # Normalization
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Dataset
    dataset = Sentinel(
        root_dir=data_root,
        split_type=split,
        transform=base_transform,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05)
    )
    
    # Data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader, norm_sar, norm_rgb


def evaluate_model(model, data_loader, norm_sar, norm_rgb, device, metrics_evaluator):
    """Evaluate model on dataset"""
    model.eval()
    
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    l1_losses = []
    
    with torch.no_grad():
        for batch_idx, (sar, rgb) in enumerate(data_loader):
            sar = norm_sar(sar.to(device, non_blocking=True))
            rgb = norm_rgb(rgb.to(device, non_blocking=True))
            
            # Generate prediction
            pred = model.generator(sar)
            
            # Compute L1 loss
            l1_loss = nn.L1Loss()(pred, rgb)
            l1_losses.append(l1_loss.item())
            
            # Compute metrics
            batch_metrics = metrics_evaluator.evaluate_batch(pred, rgb)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx}/{len(data_loader)} batches")
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics:
        avg_metrics[key] = sum(all_metrics[key]) / len(all_metrics[key]) if all_metrics[key] else 0.0
    
    avg_metrics['l1'] = sum(l1_losses) / len(l1_losses)
    
    return avg_metrics


def save_sample_images(model, data_loader, norm_sar, norm_rgb, device, output_dir, num_samples=20):
    """Save sample images for visual inspection"""
    model.eval()
    
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, (sar, rgb) in enumerate(data_loader):
            if saved_count >= num_samples:
                break
                
            sar = norm_sar(sar.to(device, non_blocking=True))
            rgb = norm_rgb(rgb.to(device, non_blocking=True))
            
            # Generate prediction
            pred = model.generator(sar)
            
            # Denormalize for visualization
            sar_vis = (sar + 1) / 2  # [-1, 1] -> [0, 1]
            rgb_vis = (rgb + 1) / 2
            pred_vis = (pred + 1) / 2
            
            # Clamp values
            sar_vis = torch.clamp(sar_vis, 0, 1)
            rgb_vis = torch.clamp(rgb_vis, 0, 1)
            pred_vis = torch.clamp(pred_vis, 0, 1)
            
            # Ensure SAR has 3 channels for visualization
            if sar_vis.size(1) == 1:
                sar_vis = sar_vis.repeat(1, 3, 1, 1)

            # Save individual images
            for i in range(min(sar.size(0), num_samples - saved_count)):
                # Concatenate side by side: [SAR | RGB | Prediction]
                comparison = torch.cat([
                    sar_vis[i],   # shape [3, H, W]
                    rgb_vis[i],
                    pred_vis[i]
                ], dim=2)  # along width
                
                # Add batch dimension back so vutils.save_image works
                comparison = comparison.unsqueeze(0)  # [1, 3, H, 3W]
                
                vutils.save_image(
                    comparison,
                    samples_dir / f'sample_{saved_count:03d}.png',
                    nrow=1,  # keep in one row
                    padding=2,
                    normalize=False
                )
                
                saved_count += 1


    
    print(f"Saved {saved_count} sample images to {samples_dir}")


def save_results(metrics, output_dir, checkpoint_path, config):
    """Save evaluation results to CSV"""
    results_file = output_dir / 'evaluation_results.csv'
    
    # Prepare results
    results = {
        'checkpoint': str(checkpoint_path),
        'psnr': f"{metrics['psnr']:.6f}",
        'ssim': f"{metrics['ssim']:.6f}",
        'lpips': f"{metrics['lpips']:.6f}",
        'l1': f"{metrics['l1']:.6f}",
        'config': str(config) if config else 'unknown'
    }
    
    # Write to CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    print(f"Results saved to {results_file}")
    
    # Also print results
    print("\nEvaluation Results:")
    print(f"PSNR: {metrics['psnr']:.3f}")
    print(f"SSIM: {metrics['ssim']:.3f}")
    print(f"LPIPS: {metrics['lpips']:.3f}")
    print(f"L1 Loss: {metrics['l1']:.6f}")


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint, config = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Override config with command line args if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
        config.update(file_config)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_dir = Path(args.checkpoint).parent / f'eval_{checkpoint_name}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loader
    data_loader, norm_sar, norm_rgb = create_data_loader(
        args.data_root, args.split, args.batch_size, args.num_workers,
        config.get('image_size', 256)
    )
    print(f"Created {args.split} data loader with {len(data_loader.dataset)} samples")
    
    # Create model
    model = create_pix2pix_model(config).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully")
    
    # Create metrics evaluator
    metrics_evaluator = MetricsEvaluator(device)
    
    # Evaluate model
    print(f"Evaluating on {args.split} split...")
    metrics = evaluate_model(model, data_loader, norm_sar, norm_rgb, device, metrics_evaluator)
    
    # Save results
    save_results(metrics, output_dir, args.checkpoint, config)
    
    # Save sample images if requested
    if args.save_samples:
        print("Saving sample images...")
        save_sample_images(model, data_loader, norm_sar, norm_rgb, device, output_dir, args.num_samples)
    
    # Save config for reference
    if config:
        config_file = output_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()