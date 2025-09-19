import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


project_root = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(description='Plot training and validation metrics')
    p.add_argument('--train-log', type=str, default='results/unet/training_log.csv')
    p.add_argument('--val-metrics', type=str, default='results/unet/val_metrics.csv')
    p.add_argument('--out-dir', type=str, default='results/unet/plots')
    return p.parse_args()


def main():
    args = parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)

    train_log = Path(args.train_log)
    if not train_log.is_absolute():
        train_log = project_root / train_log
    val_metrics = Path(args.val_metrics)
    if not val_metrics.is_absolute():
        val_metrics = project_root / val_metrics

    train_df = pd.read_csv(train_log)
    val_df = pd.read_csv(val_metrics)

    # Plot training loss vs iterations
    plt.figure(figsize=(8,4))
    plt.plot(train_df.index, train_df['loss_l1'])
    plt.title('Training L1 Loss vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('L1 Loss')
    plt.tight_layout()
    plt.savefig(out / 'train_loss.png')
    plt.close()

    # Metrics vs epochs
    for col in ['psnr', 'ssim', 'lpips', 'val_l1']:
        plt.figure(figsize=(6,4))
        
        # Raw curve
        plt.plot(val_df['epoch'], val_df[col], label="Raw", alpha=0.5, linewidth=1)
        
        # Smoothed curve (window=5 is common, can tweak)
        smooth = val_df[col].rolling(window=5, center=True).mean()
        plt.plot(val_df['epoch'], smooth, label="Smoothed", linewidth=2)
        
        # Mark best epoch
        if col in ['psnr', 'ssim']:
            # For PSNR and SSIM, higher is better
            best_idx = val_df[col].idxmax()
            best_epoch = val_df.loc[best_idx, 'epoch']
            best_value = val_df.loc[best_idx, col]
        else:
            # For LPIPS and val_l1, lower is better
            best_idx = val_df[col].idxmin()
            best_epoch = val_df.loc[best_idx, 'epoch']
            best_value = val_df.loc[best_idx, col]
        
        plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: Epoch {best_epoch} ({best_value:.3f})')
        
        plt.title(f'{col.upper()} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(col.upper())
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f'{col}_vs_epoch.png')
        plt.close()


if __name__ == '__main__':
    main()

