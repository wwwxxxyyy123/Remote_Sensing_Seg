import os
import torch
import torch.nn as nn
from typing import Dict, Optional

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained')

def ensure_pretrained_dir():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    return PRETRAINED_DIR

def _hf_load_any(repo_id: str, save_dir: str, save_path: str):
    """Try to download backbone weights from HuggingFace Hub in various formats.

    On success moves / copies the weights to save_path and returns save_path.
    Supports pytorch_model.bin, model.safetensors, and backbone-only repos.
    """
    from huggingface_hub import hf_hub_download, list_repo_files, HfFileSystem

    files = set()
    try:
        files = set(list_repo_files(repo_id))
    except Exception:
        pass

    # Ordered candidate (filename, loader) pairs. We prefer the raw MiT backbone.
    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
    ]
    chosen = None
    for fn in candidates:
        if fn in files:
            chosen = fn
            break
    # Fallback: try without listing (may be a private / restricted repo)
    if chosen is None:
        chosen = "pytorch_model.bin"

    try:
        temp_path = hf_hub_download(repo_id=repo_id, filename=chosen, local_dir=save_dir)
    except Exception:
        # try the safetensors variant as a last resort
        alt = "model.safetensors" if chosen != "model.safetensors" else "pytorch_model.bin"
        temp_path = hf_hub_download(repo_id=repo_id, filename=alt, local_dir=save_dir)

    # If it is safetensors, convert to a .pth for uniform loading
    if temp_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            sd = load_file(temp_path)
        except Exception as e:
            raise RuntimeError(f"safetensors not available or corrupt: {e}")
        torch.save(sd, save_path)
    else:
        # Copy / rename into the canonical save_path location
        try:
            if os.path.realpath(temp_path) != os.path.realpath(save_path):
                import shutil
                shutil.copy2(temp_path, save_path)
        except Exception:
            os.replace(temp_path, save_path)

    return save_path


def download_mit_b0_pretrained(force_download=False, save_dir=None) -> str:
    if save_dir is None:
        save_dir = PRETRAINED_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mit_b0.pth')

    if not force_download and os.path.exists(save_path):
        return save_path

    print("[INFO] Downloading MiT-b0 pretrained weights...")

    # 1) NVlabs GitHub release (official raw MixTransformer weights)
    try:
        import urllib.request
        url = "https://github.com/NVlabs/SegFormer/releases/download/v0.1/mit_b0.pth"
        urllib.request.urlretrieve(url, save_path)
        print(f"[INFO] MiT-b0 weights saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Failed to download MiT-b0 from GitHub release: {e}")

    # 2) HuggingFace Hub: nvidia/mit-b0 pure MixTransformer weights
    try:
        _hf_load_any("nvidia/mit-b0", save_dir, save_path)
        print(f"[INFO] MiT-b0 weights downloaded from HuggingFace Hub (nvidia/mit-b0): {save_path}")
        return save_path
    except Exception as e2:
        print(f"[WARN] Failed to download nvidia/mit-b0 from HuggingFace Hub: {e2}")

    # 3) HuggingFace Hub: segformer-b0-finetuned-ade (extract encoder only when loaded)
    try:
        _hf_load_any("nvidia/segformer-b0-finetuned-ade-512-512", save_dir, save_path)
        print(f"[INFO] SegFormer-b0 weights downloaded from HuggingFace Hub "
              f"(nvidia/segformer-b0-finetuned-ade-512-512); encoder-only will be extracted at load time.")
        return save_path
    except Exception as e3:
        print(f"[WARN] Also failed to download segformer-b0-finetuned variant: {e3}")

    return None

def download_unet_pretrained(force_download=False, save_dir=None) -> str:
    if save_dir is None:
        save_dir = PRETRAINED_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'unet_carvana.pth')
    
    if not force_download and os.path.exists(save_path):
        return save_path
    
    print("[INFO] Downloading UNet pretrained weights...")
    try:
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] UNet weights saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Failed to download UNet pretrained weights: {e}")
        return None

def download_mobilenet_v2_pretrained(force_download=False, save_dir=None) -> str:
    if save_dir is None:
        save_dir = PRETRAINED_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mobilenet_v2.pth')

    if not force_download and os.path.exists(save_path):
        return save_path

    print("[INFO] Downloading MobileNetV2 pretrained weights...")
    try:
        model = torch.hub.load('pytorch/vision:v0.16.0', 'mobilenet_v2', pretrained=True)
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] MobileNetV2 weights saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Failed to download MobileNetV2 pretrained weights: {e}")
        return None


def _remap_mit_key(k: str) -> str:
    """Remap a pretrained MiT/SegFormer key to our model's naming convention.

    Handles:
      - Various prefixes: segformer.encoder. / encoder. / backbone. / (none, NVlabs)
      - NVlabs typo: patch_embedings (missing 'd')
      - HF patch embed naming: projection -> proj, layer_norm -> norm
      - HF attention naming: attention.self.query/key/value -> attn.q/k/v, etc.
      - Block norms: layer_norm_1 / layer_norm_2 -> norm1 / norm2
      - MLP naming: dense1 / dense2 -> fc1 / fc2
      - dwconv wrapper: ensures nested .mlp.dwconv.dwconv. form matches our DWConv class
    """
    key = k

    # ---- 1) Strip common outer prefixes ------------------------------------
    for prefix in ('segformer.encoder.', 'encoder.', 'backbone.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    # ---- 2) Patch embedding fixes -----------------------------------------
    # NVlabs typo: patch_embedings (missing 'd') -> patch_embeddings
    key = key.replace('patch_embedings.', 'patch_embeddings.')

    # HF SegFormer patch embed uses "projection" instead of "proj"
    # We match: patch_embeddings.N.projection.(weight|bias) -> patch_embeddings.N.proj.$1
    if 'patch_embeddings.' in key:
        key = key.replace('.projection.weight', '.proj.weight')
        key = key.replace('.projection.bias',   '.proj.bias')
        # HF also uses "layer_norm" for the patch embed norm; ours is now "norm"
        key = key.replace('.layer_norm.weight', '.norm.weight')
        key = key.replace('.layer_norm.bias',   '.norm.bias')

    # HF SegFormer encoder uses "layer_norms.N" (plural) or NVlabs uses
    # "layer_norm.N" (singular) for per-stage output LayerNorms.
    # Ours is: norm.N.weight/bias
    if key.startswith('layer_norms.'):
        key = 'norm.' + key[len('layer_norms.'):]
    elif key.startswith('layer_norm.'):
        key = 'norm.' + key[len('layer_norm.'):]

    # ---- 3) Attention module naming ---------------------------------------
    # 3a) Sub-field renames first (because they include ".attention." in the HF keys)
    key = key.replace('.attention.self.query.weight', '.attn.q.weight')
    key = key.replace('.attention.self.query.bias',   '.attn.q.bias')
    key = key.replace('.attention.self.key.weight',   '.attn.k.weight')
    key = key.replace('.attention.self.key.bias',     '.attn.k.bias')
    key = key.replace('.attention.self.value.weight', '.attn.v.weight')
    key = key.replace('.attention.self.value.bias',   '.attn.v.bias')
    key = key.replace('.attention.output.dense.weight', '.attn.proj.weight')
    key = key.replace('.attention.output.dense.bias',   '.attn.proj.bias')
    key = key.replace('.attention.self.sr.weight',    '.attn.sr.weight')
    key = key.replace('.attention.self.sr.bias',      '.attn.sr.bias')
    key = key.replace('.attention.self.norm.weight',  '.attn.norm.weight')
    key = key.replace('.attention.self.norm.bias',    '.attn.norm.bias')
    key = key.replace('.attention.self.layer_norm.weight', '.attn.norm.weight')
    key = key.replace('.attention.self.layer_norm.bias',   '.attn.norm.bias')
    # 3b) Fallback: rename any remaining ".attention." module -> ".attn."
    key = key.replace('.attention.', '.attn.')

    # ---- 3c) Attention module fallback: after .attention. -> .attn., we
    #         may still have .attn.layer_norm.(weight|bias) from old local
    #         checkpoints or HF variants -> map to .attn.norm.$1
    key = key.replace('.attn.layer_norm.weight', '.attn.norm.weight')
    key = key.replace('.attn.layer_norm.bias',   '.attn.norm.bias')

    # ---- 4) Block norms ---------------------------------------------------
    key = key.replace('.layer_norm_1.weight', '.norm1.weight')
    key = key.replace('.layer_norm_1.bias',   '.norm1.bias')
    key = key.replace('.layer_norm_2.weight', '.norm2.weight')
    key = key.replace('.layer_norm_2.bias',   '.norm2.bias')

    # ---- 4b) Block-level norms that used a different (non-indexed) name --
    # e.g. some HF/mmcv variants use just ".layer_norm.weight" at the block
    # level.  Map those conservatively: if after ".block.X.Y." we see a
    # ".layer_norm." we try to split it by context.  Since we already
    # handled .layer_norm_1 and .layer_norm_2, any remaining plain
    # ".layer_norm." inside a block path is unexpected; leave as-is so the
    # diagnostic report surfaces it (no silent mis-mapping).

    # ---- 5) MLP naming ----------------------------------------------------
    key = key.replace('.mlp.dense1.weight', '.mlp.fc1.weight')
    key = key.replace('.mlp.dense1.bias',   '.mlp.fc1.bias')
    key = key.replace('.mlp.dense2.weight', '.mlp.fc2.weight')
    key = key.replace('.mlp.dense2.bias',   '.mlp.fc2.bias')

    # ---- 6) dwconv wrapper depth ------------------------------------------
    # Our MixFFN now uses DWConv class -> weights live under .mlp.dwconv.dwconv.(weight|bias)
    # Flat form (from older local code): .mlp.dwconv.(weight|bias) -> wrap once
    if key.endswith('.mlp.dwconv.weight') or '.mlp.dwconv.weight' in key and '.mlp.dwconv.dwconv.' not in key:
        key = key.replace('.mlp.dwconv.weight', '.mlp.dwconv.dwconv.weight')
    if key.endswith('.mlp.dwconv.bias') or '.mlp.dwconv.bias' in key and '.mlp.dwconv.dwconv.' not in key:
        key = key.replace('.mlp.dwconv.bias',   '.mlp.dwconv.dwconv.bias')

    # ---- 7) Final encoder prefix ------------------------------------------
    if not key.startswith('encoder.'):
        key = 'encoder.' + key

    return key


def _load_state_dict_any(path: str) -> dict:
    """Load a state dict from .pth, .bin, or .safetensors."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".safetensors",):
        try:
            from safetensors.torch import load_file
            return load_file(path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load safetensors file {path}. "
                f"Install safetensors: pip install safetensors. Cause: {e}"
            )
    # .pth, .bin, anything else -> try torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        # Older / various files may fail with weights_only=True
        return torch.load(path, map_location='cpu')


def load_mit_b0_weights(model: nn.Module, pretrained_path: str) -> nn.Module:
    print("[INFO] Loading MiT-b0 pretrained weights...")
    pretrained_dict = _load_state_dict_any(pretrained_path)

    # Unwrap common container formats
    while isinstance(pretrained_dict, dict):
        for key in ('state_dict', 'model', 'module', 'net'):
            if key in pretrained_dict and isinstance(pretrained_dict[key], dict):
                pretrained_dict = pretrained_dict[key]
                break
        else:
            break
    # Ensure we actually have a dict of tensors
    if not isinstance(pretrained_dict, dict):
        raise RuntimeError(f"Pretrained file at {pretrained_path} did not yield a state dict.")

    model_dict = model.state_dict()
    model_keys = set(model_dict.keys())

    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if not hasattr(v, 'shape'):
            continue
        new_key = _remap_mit_key(k)
        if new_key in model_keys and tuple(model_dict[new_key].shape) == tuple(v.shape):
            filtered_dict[new_key] = v.detach().to('cpu').clone()

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    # ---- Concise summary ---------------------------------------------------
    total_encoder_params = len([k for k in model_keys if k.startswith('encoder.')])
    loaded = len(filtered_dict)

    if loaded == total_encoder_params:
        print(f"[INFO] MiT-b0 pretrained: {loaded}/{total_encoder_params} encoder params loaded (100%)")
    else:
        print(f"[INFO] MiT-b0 pretrained: {loaded}/{total_encoder_params} encoder params loaded")

    return model

def load_unet_weights(model: nn.Module, pretrained_path: str) -> nn.Module:
    print("[INFO] Loading UNet pretrained weights...")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    
    model_dict = model.state_dict()
    filtered_dict = {}
    
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            if k.startswith('outc'):
                continue
            filtered_dict[k] = v
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    loaded_keys = len(filtered_dict)
    print(f"[INFO] Loaded {loaded_keys} UNet pretrained parameters")

    return model

def load_mobilenet_v2_weights(model: nn.Module, pretrained_path: str) -> nn.Module:
    print("[INFO] Loading MobileNetV2 pretrained weights...")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)

    model_dict = model.state_dict()
    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if not k.startswith('features.'):
                continue
        # DeepLabV3Plus: encoder.mobilenet_v2.features.*
        new_key = 'encoder.mobilenet_v2.' + k

        if new_key in model_dict and model_dict[new_key].shape == v.shape:
            filtered_dict[new_key] = v

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    total_keys = len([k for k in pretrained_dict if k.startswith('features.')])
    loaded_keys = len(filtered_dict)
    print(f"[INFO] MobileNetV2 pretrained: {loaded_keys}/{total_keys} backbone params loaded")

    return model

def get_model_weight_filename(model_name: str) -> str:
    if model_name == 'unet':
        return 'unet_carvana.pth'
    elif model_name in ['deeplabv3plus', 'mflnet', 'c2mflnet', 'c3mflnet']:
        return 'mobilenet_v2.pth'
    elif model_name == 'segformer':
        return 'mit_b0.pth'
    else:
        return f'{model_name}.pth'

def find_pretrained_weights(model_name: str, search_path: str) -> Optional[str]:
    weight_filename = get_model_weight_filename(model_name)
    
    if os.path.isfile(search_path):
        return search_path
    
    candidate_path = os.path.join(search_path, weight_filename)
    if os.path.isfile(candidate_path):
        return candidate_path
    
    return None

def download_pretrained_weights(model_name: str, save_dir: str = None, force_download=False) -> Optional[str]:
    if model_name == 'unet':
        return download_unet_pretrained(force_download, save_dir)
    elif model_name in ['deeplabv3plus', 'mflnet', 'c2mflnet', 'c3mflnet']:
        return download_mobilenet_v2_pretrained(force_download, save_dir)
    elif model_name == 'segformer':
        return download_mit_b0_pretrained(force_download, save_dir)
    else:
        print(f"[WARN] No pretrained weights available for model: {model_name}")
        return None

def load_pretrained_weights(model: nn.Module, model_name: str, pretrained_path: str) -> nn.Module:
    if model_name == 'unet':
        return load_unet_weights(model, pretrained_path)
    elif model_name in ['deeplabv3plus', 'mflnet', 'c2mflnet', 'c3mflnet']:
        return load_mobilenet_v2_weights(model, pretrained_path)
    elif model_name == 'segformer':
        return load_mit_b0_weights(model, pretrained_path)
    else:
        print(f"[WARN] No pretrained weight loader for model: {model_name}")
        return model