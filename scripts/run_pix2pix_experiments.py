import subprocess
import sys
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_experiment(config_path, data_root, experiment_name=None, resume=None):
    """Run a single Pix2Pix experiment"""
    cmd = [
        sys.executable, 'scripts/train_pix2pix.py',
        '--config', config_path,
        '--data-root', data_root
    ]
    
    if experiment_name:
        cmd.extend(['--experiment-name', experiment_name])
    
    if resume:
        cmd.extend(['--resume', resume])
    
    print(f"Running experiment: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"Experiment completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Pix2Pix ablation experiments')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--experiments', nargs='+', default=['all'], 
                       choices=['all', 'default', 'warmstart', 'l1_50', 'l1_200', 
                               'perceptual', 'perceptual_5', 'lr_1e4', 'batch16', 'spectral_norm'],
                       help='Which experiments to run')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Define experiment configurations
    experiments = {
        'default': ('configs/pix2pix_default.yaml', 'default'),
        'warmstart': ('configs/pix2pix_warmstart.yaml', 'warmstart'),
        'l1_50': ('configs/pix2pix_l1_50.yaml', 'l1_50'),
        'l1_200': ('configs/pix2pix_l1_200.yaml', 'l1_200'),
        'perceptual': ('configs/pix2pix_perceptual.yaml', 'perceptual'),
        'perceptual_5': ('configs/pix2pix_perceptual_5.yaml', 'perceptual_5'),
        'lr_1e4': ('configs/pix2pix_lr_1e4.yaml', 'lr_1e4'),
        'batch16': ('configs/pix2pix_batch16.yaml', 'batch16'),
        'spectral_norm': ('configs/pix2pix_spectral_norm.yaml', 'spectral_norm')
    }
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        exp_list = list(experiments.keys())
    else:
        exp_list = args.experiments
    
    print(f"Running {len(exp_list)} experiments: {exp_list}")
    
    # Run experiments
    successful = 0
    failed = 0
    
    for exp_name in exp_list:
        if exp_name not in experiments:
            print(f"Unknown experiment: {exp_name}")
            continue
            
        config_path, exp_id = experiments[exp_name]
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*50}")
        
        success = run_experiment(config_path, args.data_root, exp_id, args.resume)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Experiment Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()