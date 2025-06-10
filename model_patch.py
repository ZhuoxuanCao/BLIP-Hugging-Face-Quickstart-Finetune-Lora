# model_patch.py
import torch
import torch.nn as nn
from lora_layer import LoRALinear

def find_and_patch_linear_layers(
    model,
    target_keywords=None,  # 只有名字包含这些关键词的Linear才patch。None表示全部patch
    lora_r=4,
    lora_alpha=1.0,
    lora_dropout=0.0,
    verbose=True
):
    """
    递归地遍历model，将匹配的nn.Linear替换为LoRALinear。
    默认仅patch包含target_keywords的Linear层。
    """
    num_patched = 0
    for name, module in model.named_children():
        # 递归对子模块做patch
        patched = find_and_patch_linear_layers(
            module, target_keywords, lora_r, lora_alpha, lora_dropout, verbose=False
        )
        num_patched += patched

        # 仅对nn.Linear做替换
        if isinstance(module, nn.Linear):
            if (target_keywords is None) or any([kw in name for kw in target_keywords]):
                # 记录原始参数
                old_linear = module
                # 新建LoRA线性层
                lora_linear = LoRALinear(
                    in_features=old_linear.in_features,
                    out_features=old_linear.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=old_linear.bias is not None
                )
                # 拷贝原始权重
                lora_linear.weight.data = old_linear.weight.data.clone()
                if old_linear.bias is not None:
                    lora_linear.bias.data = old_linear.bias.data.clone()
                # 冻结原始权重
                lora_linear.weight.requires_grad = False
                if lora_linear.bias is not None:
                    lora_linear.bias.requires_grad = False
                # LoRA部分参数可训练（默认即可）
                # 替换
                setattr(model, name, lora_linear)
                num_patched += 1
                if verbose:
                    print(f"[LoRA Patch] Patched: {model.__class__.__name__}.{name}  ({old_linear.in_features}→{old_linear.out_features})")
    return num_patched

def patch_blip1_visual_transformer(model, r=4, alpha=1.0, dropout=0.0, verbose=True):
    """
    只 patch BLIP1 的视觉主干部分（视觉Transformer）。
    兼容 blip-image-captioning-large（BLIP1）
    """
    # 自动寻找视觉主干（blip1: model.visual_encoder, blip2: model.vision_model, etc）
    visual_attr_candidates = ['visual_encoder', 'vision_model']
    visual_encoder = None
    for attr in visual_attr_candidates:
        if hasattr(model, attr):
            visual_encoder = getattr(model, attr)
            if verbose:
                print(f"[LoRA Patch] Find visual backbone: {attr}")
            break
    assert visual_encoder is not None, "Cannot find visual backbone in model! Please manually set attribute."

    # 通常视觉主干都是ViT/Transformer结构，遍历patch所有Attention/Linear
    # 建议只patch 'qkv', 'proj', 'fc1', 'fc2' 等关键层
    keywords = ['qkv', 'proj', 'fc1', 'fc2']
    total_patched = find_and_patch_linear_layers(
        visual_encoder, target_keywords=keywords, lora_r=r, lora_alpha=alpha, lora_dropout=dropout, verbose=verbose
    )
    print(f"[LoRA Patch] Total {total_patched} Linear layers patched in {attr}.")

# Example usage:
if __name__ == "__main__":
    from transformers import BlipForConditionalGeneration
    import torch

    # 加载 BLIP1 (image captioning large) 示例
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("Before LoRA patch:")
    for n, m in model.visual_encoder.named_modules():
        if isinstance(m, nn.Linear):
            print(n, m)

    patch_blip1_visual_transformer(model, r=8, alpha=16, dropout=0.05)

    print("\nAfter LoRA patch:")
    for n, m in model.visual_encoder.named_modules():
        if isinstance(m, LoRALinear):
            print(n, m)
