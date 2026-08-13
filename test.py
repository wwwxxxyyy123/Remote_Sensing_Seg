import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

# 项目根目录注入到sys.path，确保模型模块可正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.Unet import UNet
from models.DeepLabV3Plus import DeepLabV3Plus
from models.SegFormer import SegFormer_init
from models.MFLNet import MFLNet
from models.C2MFLNet import C2MFLNet
from models.C3MFLNet import C3MFLNet

# ========== 路径常量 ==========
DATA_PATH = "./LoveDA/Test/images"
MODLE_PATH = "./checkpoints/c2mflnet_20260811_131440/best.pth"
RESULT_PATH = "./results/c2mflnet_20260811_131440/Result"

def make_dir(path):
    """创建文件夹，不存在则创建，存在则不创建"""
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] 结果保存路径: {path}")


def load_model(model_path):
    """
    从.pth文件加载模型
    
    Args:
        model_path: 模型权重文件路径(.pth)
    
    Returns:
        tuple: (model, args) - 加载好权重的模型和训练参数
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint中获取训练参数 (args为argparse.Namespace对象，转为dict统一访问)
    args = checkpoint.get('args', None)
    args = vars(args) if args is not None else {}
    
    model_name = args.get('model', 'deeplabv3plus')
    num_classes = args.get('num_classes', 7)
    input_channels = args.get('input_channels', 3)
    drop_rate = args.get('drop_rate', 0.3)
    
    print(f"[INFO] 检测到模型类型: {model_name}")
    print(f"[INFO] 类别数: {num_classes}, 输入通道: {input_channels}")
    
    # 构建模型
    if model_name == 'unet':
        model = UNet(input_channels, num_classes)
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(input_channels, num_classes)
    elif model_name == 'segformer':
        model = SegFormer_init("b0", input_channels, num_classes, drop_rate=drop_rate)
    elif model_name == 'mflnet':
        model = MFLNet(input_channels, num_classes)
    elif model_name == 'c2mflnet':
        model = C2MFLNet(input_channels, num_classes)
    elif model_name == 'c3mlfnet':
        model = C3MFLNet(input_channels, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 加载权重
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型加载完成，参数量: {total_params:,}")
    print(f"[INFO] 使用设备: {device}")
    
    return model, args


def load_data(data_path):
    """
    加载数据集目录下的全部图片
    
    Args:
        data_path: 图片目录路径
    
    Returns:
        tuple: (图片文件列表, 完整路径列表)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    
    # 支持的图片扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # 获取所有图片文件
    img_files = sorted([
        f for f in os.listdir(data_path)
        if f.lower().endswith(valid_extensions)
    ])
    
    if len(img_files) == 0:
        raise ValueError(f"目录下未找到图片: {data_path}")
    
    img_paths = [os.path.join(data_path, f) for f in img_files]
    print(f"[INFO] 加载测试图片: {len(img_files)} 张")
    
    return img_files, img_paths


def predict(model, img_paths, img_size=512):
    """
    使用模型对图片进行预测
    
    Args:
        model: 加载好权重的模型
        img_paths: 图片完整路径列表
        img_size: 输入图像尺寸
    
    Returns:
        list: 预测结果列表，每个元素为单通道掩码numpy数组
    """
    device = next(model.parameters()).device
    
    # 图像预处理管线：Resize + Normalize + ToTensorV2 (与训练一致)
    transform = Compose([
        Resize(height=img_size, width=img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    results = []
    
    pbar = tqdm(img_paths, desc="预测进度", ncols=80)
    for img_path in pbar:
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        tensor = transform(image=image)['image'].unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).squeeze(0)
            pred = pred.cpu().numpy().astype(np.uint8)
        
        # 还原到原始图像尺寸（使用最近邻插值避免标签失真）
        pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        results.append(pred_resized)
        pbar.set_postfix({'当前': os.path.basename(img_path)})
    
    return results


def save_result(results, img_files, save_path):
    """
    保存预测结果为单通道掩码图
    
    Args:
        results: 预测结果列表
        img_files: 原始图片文件名列表
        save_path: 保存目录路径
    """
    save_dir = save_path
    os.makedirs(save_dir, exist_ok=True)
    
    pbar = tqdm(zip(img_files, results), total=len(results), desc="保存结果", ncols=80)
    for img_name, pred_mask in pbar:
        save_file = os.path.join(save_dir, img_name)
        cv2.imwrite(save_file, pred_mask)
        pbar.set_postfix({'文件': img_name})
    
    print(f"[INFO] 预测结果已保存至: {save_dir}")
    print(f"[INFO] 共保存 {len(results)} 张掩码图")


if __name__ == "__main__":
    print("=" * 60)
    print("语义分割预测脚本")
    print("=" * 60)
    
    # 1. 创建结果目录
    make_dir(RESULT_PATH)
    
    # 2. 加载模型
    print("-" * 40)
    print("[STEP 1] 加载模型...")
    model, args = load_model(MODLE_PATH)
    
    # 3. 加载数据
    print("-" * 40)
    print("[STEP 2] 加载数据集...")
    img_files, img_paths = load_data(DATA_PATH)
    
    # 4. 模型推理
    print("-" * 40)
    print("[STEP 3] 开始预测...")
    
    # 从checkpoint参数读取img_size
    img_size = 1024
    print(f"[INFO] 输入尺寸: {img_size}")
    
    results = predict(model, img_paths, img_size=img_size)
    
    # 5. 保存结果
    print("-" * 40)
    print("[STEP 4] 保存结果...")
    save_result(results, img_files, RESULT_PATH)
    
    print("=" * 60)
    print("[完成] 所有预测任务已完成！")
    print("=" * 60)
