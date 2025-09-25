import sys
from pathlib import Path
import argparse
import csv
import glob

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data_loader import Sentinel
from models.unet import UNet
from src.metrics import MetricsEvaluator


def parse_args():
    p = argparse.ArgumentParser(description='Compute metrics across checkpoints on validation split')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--checkpoints-glob', type=str, default='results/unet/checkpoints/epoch_*.pt')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--out-csv', type=str, default='results/unet/metrics_by_epoch.csv')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_t = v2.Compose([v2.Resize((args.image_size, args.image_size)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    norm_rgb = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    valset = Sentinel(root_dir=args.data_root, split_type='val', transform=base_t, split_mode='random', split_ratio=(0.85, 0.10, 0.05))
    vloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    metrics = MetricsEvaluator(device)

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'psnr', 'ssim', 'lpips', 'val_l1','batch_size'])

        for ckpt_path in sorted(glob.glob(args.checkpoints_glob)):
            epoch_num = Path(ckpt_path).stem.split('_')[-1]
            model = UNet(in_channels=1, out_channels=3).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model'] if 'model' in state else state)
            model.eval()

            agg = {"psnr": [], "ssim": [], "lpips": []}
            val_losses = []
            with torch.no_grad():
                for sar, rgb in vloader:
                    sar = norm_sar(sar.to(device, non_blocking=True))
                    rgb = norm_rgb(rgb.to(device, non_blocking=True))
                    pred = model(sar)
                    val_losses.append(torch.nn.functional.l1_loss(pred, rgb).item())
                    m = metrics.evaluate_batch(pred, rgb)
                    for k in agg:
                        agg[k].append(m[k])
            psnr_e = sum(agg['psnr']) / max(1, len(agg['psnr']))
            ssim_e = sum(agg['ssim']) / max(1, len(agg['ssim']))
            lpips_e = sum(agg['lpips']) / max(1, len(agg['lpips']))
            val_l1 = sum(val_losses) / max(1, len(val_losses))
            writer.writerow([epoch_num, f'{psnr_e:.6f}', f'{ssim_e:.6f}', f'{lpips_e:.6f}', f'{val_l1:.6f}'])


if __name__ == '__main__':
    main()

