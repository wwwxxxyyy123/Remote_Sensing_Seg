import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(csv_path, save_dir=None, best_epoch=None):
    """
    Plot training curves from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing training metrics
        save_dir: Directory to save the plot image (default: same as CSV file directory)
        best_epoch: Epoch number where best model was saved (auto-detected if None)
    """
    # Read CSV file
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Auto-detect best epoch if not provided
    if best_epoch is None:
        best_epoch = df.loc[df['val_mIoU'].idxmax(), 'epoch']
        best_epoch = int(best_epoch)
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot train_loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    axes[0, 0].scatter([best_epoch], [df.loc[df['epoch'] == best_epoch, 'train_loss'].values[0]], 
                       color='r', s=100, zorder=5)
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('Train Loss', fontsize=10)
    axes[0, 0].set_title('Train Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot val_loss
    axes[0, 1].plot(df['epoch'], df['val_loss'], 'g-', linewidth=2, label='Val Loss')
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    axes[0, 1].scatter([best_epoch], [df.loc[df['epoch'] == best_epoch, 'val_loss'].values[0]], 
                       color='r', s=100, zorder=5)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('Val Loss', fontsize=10)
    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot val_mIoU
    axes[0, 2].plot(df['epoch'], df['val_mIoU'], 'm-', linewidth=2, label='Val mIoU')
    axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    best_miou = df.loc[df['epoch'] == best_epoch, 'val_mIoU'].values[0]
    axes[0, 2].scatter([best_epoch], [best_miou], color='r', s=100, zorder=5)
    axes[0, 2].set_xlabel('Epoch', fontsize=10)
    axes[0, 2].set_ylabel('Val mIoU', fontsize=10)
    axes[0, 2].set_title(f'Validation mIoU (Best: {best_miou:.4f})', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot val_MPA
    axes[1, 0].plot(df['epoch'], df['val_MPA'], 'c-', linewidth=2, label='Val MPA')
    axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    axes[1, 0].scatter([best_epoch], [df.loc[df['epoch'] == best_epoch, 'val_MPA'].values[0]], 
                       color='r', s=100, zorder=5)
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('Val MPA', fontsize=10)
    axes[1, 0].set_title('Validation MPA', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot val_mDice
    axes[1, 1].plot(df['epoch'], df['val_mDice'], 'orange', linewidth=2, label='Val mDice')
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    axes[1, 1].scatter([best_epoch], [df.loc[df['epoch'] == best_epoch, 'val_mDice'].values[0]], 
                       color='r', s=100, zorder=5)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Val mDice', fontsize=10)
    axes[1, 1].set_title('Validation mDice', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot lr
    # 分层训练时 CSV 含 shallow_lr/deep_lr 两列, 在同一子图中分别绘制两条学习率曲线;
    # 常规训练时仅含 lr 单列, 绘制单一学习率曲线(保持向后兼容)。
    if 'shallow_lr' in df.columns and 'deep_lr' in df.columns:
        axes[1, 2].plot(df['epoch'], df['shallow_lr'], 'blue', linewidth=2, label='Shallow LR (MobileNetV2)')
        axes[1, 2].plot(df['epoch'], df['deep_lr'], 'red', linewidth=2, label='Deep LR')
        axes[1, 2].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
        axes[1, 2].set_xlabel('Epoch', fontsize=10)
        axes[1, 2].set_ylabel('Learning Rate', fontsize=10)
        axes[1, 2].set_title('Learning Rate (Layered)', fontsize=12, fontweight='bold')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].plot(df['epoch'], df['lr'], 'purple', linewidth=2, label='Learning Rate')
        axes[1, 2].axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
        axes[1, 2].scatter([best_epoch], [df.loc[df['epoch'] == best_epoch, 'lr'].values[0]],
                           color='r', s=100, zorder=5)
        axes[1, 2].set_xlabel('Epoch', fontsize=10)
        axes[1, 2].set_ylabel('Learning Rate', fontsize=10)
        axes[1, 2].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Training curves plot saved to: {plot_path}")
    return plot_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training curves from CSV file")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plot')
    parser.add_argument('--best_epoch', type=int, default=None, help='Best epoch number')
    
    args = parser.parse_args()
    
    plot_training_curves(args.csv_path, args.save_dir, args.best_epoch)