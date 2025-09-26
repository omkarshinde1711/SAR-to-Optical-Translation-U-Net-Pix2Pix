import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import gaussian_filter1d
import os

def export_tensorboard_plots(logdir, output_dir):
    """
    Export TensorBoard plots with professional styling similar to your PSNR plot
    """
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get scalar tags
    tags = event_acc.Tags()['scalars']
    
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        
        # Extract data
        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])
        
        # Skip if no data
        if len(values) == 0:
            continue
        
        # Create smoothed version (similar to your orange line)
        smoothed_values = gaussian_filter1d(values, sigma=2)
        
        # Find best value and its position
        if 'loss' in tag.lower():
            best_idx = np.argmin(smoothed_values)
            best_value = np.min(smoothed_values)
            best_text = f"Best: Epoch {steps[best_idx]} ({best_value:.4f})"
        else:
            best_idx = np.argmax(smoothed_values)
            best_value = np.max(smoothed_values)
            best_text = f"Best: Epoch {steps[best_idx]} ({best_value:.3f})"
        
        # Create the plot with your exact styling
        plt.figure(figsize=(10, 6))
        
        # Plot raw data (light blue line)
        plt.plot(steps, values, color='lightblue', linewidth=1, alpha=0.7, label='Raw')
        
        # Plot smoothed data (orange line)
        plt.plot(steps, smoothed_values, color='#ff7f0e', linewidth=2.5, label='Smoothed')
        
        # Add best value horizontal line (pink dashed)
        plt.axhline(y=best_value, color='#ff69b4', linestyle='--', linewidth=1.5, 
                   label=best_text)
        
        # Styling to match your plot
        plt.title(f'{tag.replace("_", " ").title()} vs Epoch', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(tag.replace("_", " ").title(), fontsize=12)
        
        # Grid styling
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Legend styling (matching your plot)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=False, 
                  fontsize=10, framealpha=0.9)
        
        # Set axis limits with some padding
        y_min, y_max = values.min(), values.max()
        y_range = y_max - y_min
        plt.ylim(y_min - y_range*0.05, y_max + y_range*0.05)
        
        # Make it look clean like your plot
        plt.tight_layout()
        
        # Remove top and right spines for cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        
        # Save with high quality
        filename = tag.replace('/', '_').replace('\\', '_') + '.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved plot: {filename}")

def create_summary_report(logdir, output_dir):
    """
    Create a simple text summary of training results
    """
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    
    summary = []
    summary.append("Training Results Summary")
    summary.append("=" * 30)
    summary.append("")
    
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        if len(scalar_events) == 0:
            continue
            
        values = np.array([e.value for e in scalar_events])
        steps = np.array([e.step for e in scalar_events])
        
        if 'loss' in tag.lower():
            best_idx = np.argmin(values)
            best_value = np.min(values)
            final_value = values[-1]
        else:
            best_idx = np.argmax(values)
            best_value = np.max(values)
            final_value = values[-1]
        
        summary.append(f"{tag.replace('_', ' ').title()}:")
        summary.append(f"  Best: {best_value:.4f} (Epoch {steps[best_idx]})")
        summary.append(f"  Final: {final_value:.4f} (Epoch {steps[-1]})")
        summary.append("")
    
    # Save summary
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print("Saved training summary")

# Usage examples
if __name__ == "__main__":
    # Replace with your actual paths
    logdir = 'results/pix2pix/pix2pix_20250920_005609/tensorboard'
    output_dir = 'results/pix2pix/pix2pix_20250920_005609/plots'
    
    # Export individual plots only
    export_tensorboard_plots(logdir, output_dir)
    
    # Create a simple summary report
    create_summary_report(logdir, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Files created:")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                print(f"  - {file}")