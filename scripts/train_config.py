import sys
from pathlib import Path

# Ensure project root is on sys.path so 'scripts' and 'models' imports work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.train import run_training


def main():
    # Edit these values and run this script directly.
    CONFIG = {
        'data_root': r"D:\COLLEGE\SAR\2.7 Gb V_2\v_2",
        'save_dir': 'results/unet',
        'epochs': 100,
        'batch_size': 12,  #16,
        'image_size': 256,
        'num_workers': 4,
        'lr': 2e-4,
        'save_interval': 1,
        'resume': r"D:\COLLEGE\SAR_Analysis_Project\scripts\results\unet\checkpoints\epoch_65.pt",  # '' to start fresh
        'scheduler': 'plateau',  # 'none' | 'step' | 'plateau'
        'step_size': 20,
        'gamma': 0.5
    }

    run_training(**CONFIG)


if __name__ == '__main__':
    main()

