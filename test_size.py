# -*- coding: utf-8 -*-
"""
计算 UNet、SegFormer、DeepLabV3+、MFLNet 的参数量、FLOPs 与 FPS。

统计口径：
- 参数量：模型所有可学习参数（Params），单位 M（百万）
- FLOPs：使用 thop 统计 MACs（乘加运算数），FLOPs = 2 × MACs，单位 G（十亿）
- FPS：batch size = 1，先预热若干次再计时若干次取平均，FPS = 1 / 平均单图耗时
- 模型构建方式与 train.py 的 load_model 保持一致（SegFormer 使用 MiT-b0）

用法：
    python test_size.py                     # 默认 3×512×512 输入、7 个类别
    python test_size.py --img_size 1024     # 指定输入分辨率（与训练默认一致）
    python test_size.py --num_classes 21    # 指定类别数
    python test_size.py --warmup 50 --iters 200   # 自定义 FPS 预热与计时次数
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from thop import profile

from models.Unet import UNet
from models.SegFormer import SegFormer_init
from models.DeepLabV3Plus import DeepLabV3Plus
from models.MFLNet import MFLNet


def build_models(input_channels, num_classes):
    """与 train.py 中 load_model 的构建方式保持一致。"""
    return {
        "UNet":       UNet(input_channels, num_classes),
        "SegFormer":  SegFormer_init("b0", input_channels, num_classes, drop_rate=0.0),
        "DeepLabV3+": DeepLabV3Plus(input_channels, num_classes),
        "MFLNet":     MFLNet(input_channels, num_classes),
    }


def count_parameters(model):
    """统计模型总参数量与可训练参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure(model, input_tensor):
    """返回 (参数量, 可训练参数量, MACs, FLOPs)。"""
    model.eval()
    total_params, trainable_params = count_parameters(model)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2.0 * macs
    return total_params, trainable_params, macs, flops


def measure_fps(model, input_tensor, warmup, iters):
    """测量 batch size=1 的平均推理耗时与 FPS。

    返回 (平均单图耗时 ms, FPS)。
    CUDA 下每次前后都做 synchronize，保证计时包含完整的 GPU 计算。
    """
    use_cuda = input_tensor.is_cuda
    model.eval()
    with torch.no_grad():
        # 预热：触发 cuDNN 算子选择、显存分配等，避免计入首次开销
        for _ in range(warmup):
            model(input_tensor)
        if use_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            model(input_tensor)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / iters * 1000.0
    fps = iters / elapsed
    return latency_ms, fps


def main():
    parser = argparse.ArgumentParser(description="统计模型参数量、FLOPs 与 FPS")
    parser.add_argument("--img_size", type=int, default=512,
                        help="输入图像分辨率（正方形边长），默认 512；训练默认为 1024")
    parser.add_argument("--input_channels", type=int, default=3,
                        help="输入通道数，默认 3（RGB）")
    parser.add_argument("--num_classes", type=int, default=7,
                        help="类别数，默认 7（与 train.py 默认一致）")
    parser.add_argument("--warmup", type=int, default=50,
                        help="FPS 测试前的预热次数，默认 50")
    parser.add_argument("--iters", type=int, default=200,
                        help="FPS 测试的计时推理次数，默认 200")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_models(args.input_channels, args.num_classes)
    dummy_input = torch.randn(1, args.input_channels, args.img_size, args.img_size,
                              device=device)

    print("=" * 88)
    print(f"输入尺寸: (1, {args.input_channels}, {args.img_size}, {args.img_size})"
          f"  |  类别数: {args.num_classes}  |  设备: {device}")
    print("FLOPs = 2 × MACs（thop 统计）；FPS 为 batch size=1，"
          f"预热 {args.warmup} 次、计时 {args.iters} 次的平均值")
    print("=" * 88)
    header = (f"{'Model':<12}{'Params (M)':>12}{'MACs (G)':>11}{'FLOPs (G)':>11}"
              f"{'Latency (ms)':>14}{'FPS':>10}")
    print(header)
    print("-" * 88)

    results = []
    for name, model in models.items():
        model.to(device)
        total_params, trainable_params, macs, flops = measure(model, dummy_input)
        latency_ms, fps = measure_fps(model, dummy_input, args.warmup, args.iters)
        results.append((name, total_params, trainable_params, macs, flops,
                        latency_ms, fps))
        print(f"{name:<12}{total_params / 1e6:>12.3f}{macs / 1e9:>11.3f}"
              f"{flops / 1e9:>11.3f}{latency_ms:>14.2f}{fps:>10.2f}")

    print("=" * 88)


if __name__ == "__main__":
    main()
