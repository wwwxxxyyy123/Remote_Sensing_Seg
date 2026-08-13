import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from .metrics import miou, mean_pixel_acc, mean_dice

def evaluate(model, dataloader, criterion, device, num_classes: int, ignore_index: int = 255, desc="Val"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # 初始化混淆矩阵（CPU 上累积，避免显存开销）
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device='cpu')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            pred = torch.argmax(outputs, dim=1)  # [B, H, W]

            # 只保留有效像素（忽略 ignore_index 且值在 0~num_classes-1）
            valid_mask = (masks != ignore_index) & (masks >= 0) & (masks < num_classes)
            pred_valid = pred[valid_mask]
            mask_valid = masks[valid_mask]

            # 将预测和标签展平（一维），并转移到 CPU 进行混淆矩阵累加
            pred_cpu = pred_valid.cpu()
            mask_cpu = mask_valid.cpu()

            # 使用 bincount 计算当前 batch 的混淆矩阵增量
            indices = mask_cpu * num_classes + pred_cpu
            counts = torch.bincount(indices, minlength=num_classes * num_classes)
            counts = counts.view(num_classes, num_classes)
            conf_matrix += counts

    # 从混淆矩阵计算指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 调用以下基于混淆矩阵的计算函数（后文给出）
    miou_val = miou(conf_matrix)
    mpa_val = mean_pixel_acc(conf_matrix)
    mdice_val = mean_dice(conf_matrix)

    return {
        'Loss': avg_loss,
        'mIoU': miou_val,
        'MPA': mpa_val,
        'mDice': mdice_val,
    }