import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np


# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = F.softmax(pred, dim=1)  # (B, C, H, W)
        valid_mask = (target != self.ignore_index)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Flatten predicted probabilities and labels
        B, C, H, W = pred.shape
        pred_flat = pred_prob.permute(0, 2, 3, 1).reshape(-1, C) # (N, C), N=B*H*W
        target_flat = target.reshape(-1) # (N,)

        # Filter valid positions
        valid_mask_flat = valid_mask.reshape(-1) # (N,)
        pred_valid = pred_flat[valid_mask_flat] # (M, C)
        target_valid = target_flat[valid_mask_flat] # (M,)

        target_one_hot = F.one_hot(target_valid, num_classes=self.num_classes).float() # (M, num_classes)

        intersection = (pred_valid * target_one_hot).sum(dim=0)
        union = pred_valid.sum(dim=0) + target_one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        class_present = target_one_hot.sum(dim=0) > 0
        if class_present.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return 1 - dice[class_present].mean()



# Combined loss function (CE + Dice)
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = DiceLoss(num_classes, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice