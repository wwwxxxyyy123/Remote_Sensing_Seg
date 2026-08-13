import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from onnxsim import simplify

# -------------------- 模型定义导入（与训练保持一致） --------------------
from models.Unet import UNet
from models.DeepLabV3Plus import DeepLabV3Plus
from models.SegFormer import SegFormer_init
from models.MFLNet import MFLNet
from models.C2MFLNet import C2MFLNet
from models.C3MFLNet import C3MFLNet

# ==================== 配置常量 ====================
MODEL_NAME = 'mflnet'                      # 模型名称
CHECKPOINT_PATH = './checkpoints/mflnet_20260810_170418/best.pth'   # 训练好的 .pth 权重路径
OUTPUT_PATH = './checkpoints/mflnet_20260810_170418/best.onnx'       # 导出的 ONNX 文件名
NUM_CLASSES = 7                          # 类别数
DROP_RATE = 0.3                          # SegFormer 专用 dropout 率
SAMPLE_SIZE = 1024                       # 追踪时的示例输入尺寸
OPSET_VERSION = 13                       # ONNX opset 版本
USE_SIMPLIFY = True                      # 是否使用 onnx-simplifier 优化
# ======================================================================


def load_model(model_name, num_classes=7, drop_rate=0.3):
    """与训练脚本完全一致的模型加载函数"""
    if model_name == 'unet':
        return UNet(3, num_classes)
    elif model_name == 'deeplabv3plus':
        return DeepLabV3Plus(3, num_classes)
    elif model_name == 'segformer':
        return SegFormer_init("b0", 3, num_classes, drop_rate=drop_rate)
    elif model_name == 'mflnet':
        return MFLNet(3, num_classes)
    elif model_name == 'c2mflnet':
        return C2MFLNet(3, num_classes)
    elif model_name == 'c3mflnet':
        return C3MFLNet(3, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def export_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 构建模型
    model = load_model(MODEL_NAME, num_classes=NUM_CLASSES, drop_rate=DROP_RATE)
    model.to(device)

    # 2. 加载权重
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # 处理 DataParallel 保存时的 "module." 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # 3. 评估模式
    model.eval()

    # 4. 构造示例输入（尺寸由 SAMPLE_SIZE 决定，导出后任意尺寸均可）
    dummy_input = torch.randn(1, 3, SAMPLE_SIZE, SAMPLE_SIZE, device=device)

    # 5. 动态轴定义：batch 和空间尺寸可变
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }

    # 6. 导出 ONNX
    print(f"Exporting ONNX to {OUTPUT_PATH} (opset={OPSET_VERSION}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("Export completed.")

    # 7. 使用 onnx-simplifier 简化 ONNX 模型
    if USE_SIMPLIFY:
        print("Simplifying ONNX model ...")
        try:
            onnx_model = onnx.load(OUTPUT_PATH)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, OUTPUT_PATH)
                print("Simplification succeeded.")
            else:
                print("Simplification failed, keeping original.")
        except Exception as e:
            print(f"Simplification error: {e}, skipping.")

    # 8. 验证模型结构
    print("Validating ONNX model ...")
    try:
        onnx_model = onnx.load(OUTPUT_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 9. 使用 ONNXRuntime 测试多种尺寸推理
    print("Testing inference with ONNXRuntime ...")
    sess = ort.InferenceSession(OUTPUT_PATH)
    input_name = sess.get_inputs()[0].name

    test_shapes = [(1, 3, 512, 512), (1, 3, 1024, 1024), (1, 3, 800, 600)]
    for shape in test_shapes:
        test_input = np.random.randn(*shape).astype(np.float32)
        output = sess.run(None, {input_name: test_input})
        print(f"Input shape {shape} -> Output shape {output[0].shape}")

    print("All tests passed. ONNX export finished successfully.")


if __name__ == "__main__":
    export_onnx()