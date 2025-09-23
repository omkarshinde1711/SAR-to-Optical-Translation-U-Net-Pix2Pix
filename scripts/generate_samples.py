import sys
import argparse
from pathlib import Path
import torch
import torchvision.utils as vutils
from torchvision.transforms import v2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import Sentinel
from models.pix2pix import create_pix2pix_model


def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample images from Pix2Pix model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create model
    model = create_pix2pix_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create data loader
    base_transform = v2.Compose([
        v2.Resize((config.get('image_size', 256), config.get('image_size', 256))),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    dataset = Sentinel(
        root_dir=args.data_root,
        split_type=args.split,
        transform=base_transform,
        split_mode='random',
        split_ratio=(0.85, 0.10, 0.05)
    )
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / 'samples'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        for i in range(0, min(args.num_samples, len(dataset)), args.batch_size):
            batch_end = min(i + args.batch_size, args.num_samples)
            batch_size = batch_end - i
            
            # Get batch
            sar_batch = []
            rgb_batch = []
            for j in range(i, batch_end):
                sar, rgb = dataset[j]
                sar_batch.append(sar)
                rgb_batch.append(rgb)
            
            sar = torch.stack(sar_batch)
            rgb = torch.stack(rgb_batch)
            
            # Normalize and move to device
            sar = norm_sar(sar.to(device))
            rgb = norm_rgb(rgb.to(device))
            
            # Generate
            pred = model.generator(sar)
            
            # Denormalize for visualization
            sar_vis = (sar + 1) / 2
            rgb_vis = (rgb + 1) / 2
            pred_vis = (pred + 1) / 2
            
            # Clamp values
            sar_vis = torch.clamp(sar_vis, 0, 1)
            rgb_vis = torch.clamp(rgb_vis, 0, 1)
            pred_vis = torch.clamp(pred_vis, 0, 1)
            
            # Save individual comparisons
            # for j in range(batch_size):
            #     comparison = torch.cat([
            #         sar_vis[j:j+1],  # SAR input
            #         rgb_vis[j:j+1],  # Ground truth
            #         pred_vis[j:j+1]  # Generated
            #     ], dim=0)
                
            #     vutils.save_image(
            #         comparison,
            #         output_dir / f'sample_{i+j:03d}.png',
            #         nrow=3,
            #         padding=2,
            #         normalize=False
            #     )
            # Save individual comparisons
            for j in range(batch_size):
                sar_img = sar_vis[j:j+1]
                rgb_img = rgb_vis[j:j+1]
                pred_img = pred_vis[j:j+1]

                # Expand SAR to 3 channels
                if sar_img.shape[1] == 1:
                    sar_img = sar_img.repeat(1, 3, 1, 1)

                comparison = torch.cat([sar_img, rgb_img, pred_img], dim=0)

                vutils.save_image(
                    comparison,
                    output_dir / f'sample_{i+j:03d}.png',
                    nrow=3,
                    padding=2,
                    normalize=False
                )
    
    print(f"Generated {args.num_samples} samples in {output_dir}")


if __name__ == '__main__':
    main()