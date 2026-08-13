import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class LoveDADataSet(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, mode="train", augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mode = mode
        self.augment = augment and (mode == 'train')

        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
        self.mask_files = []
        for f in self.img_files:
            mask_path = os.path.join(mask_dir, f)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            self.mask_files.append(f)
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks must match"

        if mode == 'train' and augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(self.img_size[0], self.img_size[1]), scale=(0.5, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)
        image_t = transformed['image']
        mask_t = transformed['mask']

        mask_t = mask_t.long()
        mask_t = torch.where(mask_t == 0, torch.tensor(255, dtype=torch.long), mask_t - 1)
        
        return {'image': image_t, 'mask': mask_t}