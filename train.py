import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import numpy as np
import random
import datetime
import time
import csv

from utils.dataset import LoveDADataSet
from utils.evaluate import evaluate
from utils.loss import CombinedLoss
from utils.plot import plot_training_curves

from models.Unet import UNet
from models.DeepLabV3Plus import DeepLabV3Plus
from models.SegFormer import SegFormer_init
from models.MFLNet import MFLNet
from models.C2MFLNet import C2MFLNet
from models.C3MFLNet import C3MFLNet



def log_info(msg):   print(f"[INFO] {msg}")
def log_save(msg): print(f"[SAVE] {msg}")
def print_sep(char='=', length=60): print(char * length)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_dataset(data_root, img_size, batch_size, num_workers=4):
    # train dataset
    train_dataset = LoveDADataSet(
        img_dir=os.path.join(data_root, 'Train/images'),
        mask_dir=os.path.join(data_root, 'Train/masks'),
        img_size=img_size,
        mode='train',
        augment=True
    )

    val_dataset  = LoveDADataSet(
        img_dir=os.path.join(data_root, 'Val/images'),
        mask_dir=os.path.join(data_root, 'Val/masks'),
        img_size=img_size,
        mode='val',
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def load_model(model_name, input_channels=3, cls_num=7, drop_rate=0.1):
    if model_name == 'unet':
        model = UNet(input_channels, cls_num)
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(input_channels, cls_num)
    elif model_name == 'segformer':
        model = SegFormer_init("b0", input_channels, cls_num, drop_rate=drop_rate)
    elif model_name == 'mflnet':
        model = MFLNet(input_channels, cls_num)
    elif model_name == 'c2mflnet':
        model = C2MFLNet(input_channels, cls_num)
    elif model_name == 'c3mflnet':
        model = C3MFLNet(input_channels, cls_num)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def get_layered_param_groups(model, shallow_lr, deep_lr):
    """
    将模型参数按是否属于MobileNetV2骨干网络分为浅层组和深层组。

    分层训练策略:
        - 浅层组(shallow): MobileNetV2模块的所有参数, 使用较小的shallow_lr
        - 深层组(deep): 模型中除MobileNetV2外的所有剩余参数, 使用较大的deep_lr

    Args:
        model: 待训练的分割模型
        shallow_lr: 浅层参数(MobileNetV2骨干)的初始学习率
        deep_lr: 深层参数(非骨干部分)的初始学习率

    Returns:
        tuple: (param_groups, has_mobilenet_v2)
            - param_groups: 包含两个参数组的列表; 若未检测到MobileNetV2则返回None
            - has_mobilenet_v2: 模型中是否包含MobileNetV2模块
    """
    # 遍历模型子模块, 检测是否存在MobileNetV2并收集其参数id
    mobilenet_param_ids = set()
    has_mobilenet_v2 = False
    for module in model.modules():
        if module.__class__.__name__ == 'MobileNetV2':
            has_mobilenet_v2 = True
            for param in module.parameters():
                mobilenet_param_ids.add(id(param))

    # 未检测到MobileNetV2模块, 返回None以回退到常规训练
    if not has_mobilenet_v2:
        return None, False

    # 按参数id将所有可训练参数划分到浅层组或深层组
    shallow_params = []
    deep_params = []
    for param in model.parameters():
        if id(param) in mobilenet_param_ids:
            shallow_params.append(param)
        else:
            deep_params.append(param)

    param_groups = [
        {'params': shallow_params, 'lr': shallow_lr},
        {'params': deep_params, 'lr': deep_lr},
    ]
    return param_groups, True


def compute_class_weights(dataloader, num_classes=7, ignore_index=255):
    class_counts = np.zeros(num_classes)
    total_pixels = 0

    for batch in dataloader:
        masks = batch['mask'].numpy()
        valid_mask = (masks != ignore_index) & (masks >= 0) & (masks < num_classes)
        total_pixels += valid_mask.sum()
        for cls in range(num_classes):
            class_counts[cls] += ((masks == cls) & valid_mask).sum()
    
    weights = np.ones(num_classes)
    if total_pixels > 0:
        for cls in range(num_classes):
            if class_counts[cls] > 0:
                weights[cls] = total_pixels / (class_counts[cls] * num_classes)
    
    return torch.tensor(weights, dtype=torch.float32)

    
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, clip_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training Script")

    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'deeplabv3plus', 'segformer', 'mflnet', 'c2mflnet', 'c3mflnet'],
                        help='Model name to train')
    parser.add_argument('--data_root', type=str, default='./LoveDA',
                        help='Data root directory')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay rate')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes')
    parser.add_argument('--ignore_index', type=int, default=255,
                        help='Ignore label value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='CSV log directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--drop_rate', type=float, default=0.3,
                        help='Dropout rate for SegFormer')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the model')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Custom path to pretrained weights')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (number of epochs)')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss in CombinedLoss')

    # ===== 分层训练参数 =====
    parser.add_argument('--enable_layered_training', action='store_true',
                        help='启用分层训练: MobileNetV2骨干(浅层)与其余参数(深层)使用不同学习率')
    parser.add_argument('--shallow_lr', type=float, default=1e-5,
                        help='浅层参数学习率(MobileNetV2骨干网络), 默认1e-5')
    parser.add_argument('--deep_lr', type=float, default=1e-4,
                        help='深层参数学习率(非骨干网络部分), 默认1e-4')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print_sep()
    log_info(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Make save directory
    model_save_dir = os.path.join(args.save_dir, f"{args.model}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, f"{args.model}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_info(f"Checkpoint save path: {model_save_dir}")
    log_info(f"Log save path: {log_dir}")

    # Load dataset
    log_info("Loading dataset...")
    train_loader, val_loader = load_dataset(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print_sep()
    log_info(f"Train set: {len(train_loader.dataset)} samples")
    log_info(f"Validation set: {len(val_loader.dataset)} samples")

    # Load model
    log_info("Loading model...")
    model = load_model(args.model, input_channels=3, cls_num=args.num_classes, 
                       drop_rate=args.drop_rate)
    model.to(device)
    log_info(f"Loading model: {args.model}")

    # Load pretrained weights if specified
    if args.pretrained:
        from utils.pretrained import find_pretrained_weights, download_pretrained_weights, load_pretrained_weights
        
        pretrained_path = args.pretrained_path
        
        if pretrained_path is not None:
            found_path = find_pretrained_weights(args.model, pretrained_path)
            if found_path is not None:
                log_info(f"Found pretrained weights at: {found_path}")
                pretrained_path = found_path
            else:
                log_info(f"Pretrained weights not found in {pretrained_path}, attempting to download...")
                downloaded_path = download_pretrained_weights(args.model, pretrained_path)
                if downloaded_path is not None:
                    log_info(f"Pretrained weights downloaded to: {downloaded_path}")
                    pretrained_path = downloaded_path
                else:
                    log_info(f"Failed to download pretrained weights, training from scratch")
                    pretrained_path = None
        else:
            log_info(f"Downloading pretrained weights for {args.model}...")
            pretrained_path = download_pretrained_weights(args.model)
            if pretrained_path is not None:
                log_info(f"Pretrained weights downloaded to: {pretrained_path}")
            else:
                log_info(f"Failed to download pretrained weights, training from scratch")
        
        if pretrained_path is not None:
            model = load_pretrained_weights(model, args.model, pretrained_path)

    # Calculate class weights
    log_info("Calculating class weights...")
    class_weights = compute_class_weights(train_loader, num_classes=args.num_classes, ignore_index=args.ignore_index)
    log_info(f"Class weights: {class_weights}")

    # Loss function
    criterion = CombinedLoss(
        num_classes=args.num_classes,
        class_weights=class_weights.to(device),
        ignore_index=args.ignore_index,
        ce_weight=1.0,
        dice_weight=args.dice_weight
    )
    criterion.to(device)

    # ===== 分层训练: 参数分组与优化器构建 =====
    # 当启用分层训练且模型包含MobileNetV2骨干时, 将参数分为浅层组(骨干)与深层组(其余),
    # 两组分别使用 shallow_lr / deep_lr 作为初始学习率; 调度器(SequentialLR)会基于
    # 各 param group 的 base_lr 同步执行"预热+余弦退火", 保证两组仅初始学习率不同。
    layered_active = False
    if args.enable_layered_training:
        param_groups, has_mobilenet = get_layered_param_groups(
            model, args.shallow_lr, args.deep_lr
        )
        if has_mobilenet:
            layered_active = True
            optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
            shallow_cnt = sum(p.numel() for p in param_groups[0]['params'])
            deep_cnt = sum(p.numel() for p in param_groups[1]['params'])
            log_info(f"Layered training enabled: MobileNetV2 backbone detected")
            log_info(f"  Shallow group (MobileNetV2): {shallow_cnt:,} params, lr={args.shallow_lr}")
            log_info(f"  Deep group (rest):          {deep_cnt:,} params, lr={args.deep_lr}")
        else:
            log_info(f"Layered training requested but no MobileNetV2 module found in model '{args.model}', "
                     f"falling back to uniform learning rate (lr={args.lr})")
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler: LinearLR warmup + CosineAnnealingLR decay (epoch-level)
    # SequentialLR 内部基于各 param group 的 base_lr 计算当前 lr, 因此分层训练时浅层/深层
    # 两组会各自从其初始学习率出发, 执行相同的预热与余弦退火曲线(仅初始值不同)。
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0 / args.warmup_epochs,
            total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mix precision
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Save metrics to CSV
    # 分层训练时额外记录 shallow_lr / deep_lr 两列, 供 plot.py 绘制双学习率曲线
    csv_path = os.path.join(log_dir, f'result.csv')
    if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                if layered_active:
                    csv_writer.writerow([
                        'epoch', 'train_loss', 'val_loss', 'val_mIoU', 'val_MPA', 'val_mDice',
                        'shallow_lr', 'deep_lr'
                    ])
                else:
                    csv_writer.writerow([
                        'epoch', 'train_loss', 'val_loss', 'val_mIoU', 'val_MPA', 'val_mDice', 'lr'
                    ])

    start_epoch = 0
    best_miou = 0.0
    best_epoch = 0
    last_miou = 0.0
    early_stopping_counter = 0
    train_start_time = time.time()

    print_sep()
    if args.resume is not None and os.path.isfile(args.resume):
        log_info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_miou = checkpoint.get('miou', 0.0)
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        log_info(f"Resuming from epoch {start_epoch}, best mIoU: {best_miou:.4f}, early_stopping_counter: {early_stopping_counter}")
        print_sep('-')
    else:
        log_info("Starting training from scratch")
        print_sep('-')

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        # 分层训练时两组学习率分别记录; 常规训练时仅记录单一学习率
        if layered_active:
            current_shallow_lr = optimizer.param_groups[0]['lr']
            current_deep_lr = optimizer.param_groups[1]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            clip_grad_norm=args.clip_grad_norm
        )

        # Update learning rate at epoch level
        scheduler.step()

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, args.num_classes,
            ignore_index=args.ignore_index, desc=f"Epoch {epoch+1} [Val]"
        )

        print_sep('-')
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['Loss']:.4f}, mIoU: {val_metrics['mIoU']:.4f}, MPA: {val_metrics['MPA']:.4f}, mDice: {val_metrics['mDice']:.4f}")
        if layered_active:
            print(f"  LR (shallow/deep): {current_shallow_lr:.2e} / {current_deep_lr:.2e}")
        else:
            print(f"  LR: {current_lr:.2e}")
        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.2f}s")

        # Save metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            if layered_active:
                csv_writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_metrics['Loss'],
                    val_metrics['mIoU'],
                    val_metrics['MPA'],
                    val_metrics['mDice'],
                    current_shallow_lr,
                    current_deep_lr
                ])
            else:
                csv_writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_metrics['Loss'],
                    val_metrics['mIoU'],
                    val_metrics['MPA'],
                    val_metrics['mDice'],
                    current_lr
                ])

        # Save best model
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'miou': best_miou,
                'args': args,
                'early_stopping_counter': early_stopping_counter
            }, os.path.join(model_save_dir, 'best.pth'))
            log_save(f"New best model saved (mIoU: {best_miou:.4f})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter > 0 and early_stopping_counter % 5 == 0:
                log_info(f"Early stopping counter: {early_stopping_counter}/{args.patience}")
        
        print_sep('-')

        if epoch == args.epochs - 1:
            last_miou = val_metrics['mIoU']

        # Early stopping
        if args.patience > 0 and early_stopping_counter >= args.patience:
            log_info(f"Early stopping triggered after {args.patience} epochs without improvement")
            last_miou = val_metrics['mIoU']

            break

    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'miou': last_miou,
        'args': args,
        'early_stopping_counter': early_stopping_counter
    }, os.path.join(model_save_dir, 'last.pth'))

    total_train_time = time.time() - train_start_time
    print_sep()
    log_info(f"Training completed! Total time: {total_train_time/60:.2f} min")
    log_info(f"Best mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    log_info(f"Last mIoU: {last_miou:.4f}")
    log_info(f"Checkpoints saved in: {model_save_dir}")
    log_info(f"Log saved in: {log_dir}")
    
    # Plot training curves
    plot_training_curves(csv_path, best_epoch=best_epoch)
    print_sep()


if __name__ == "__main__":
    main()