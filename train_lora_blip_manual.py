# train_lora_blip_manual.py  （只保留这一份为最终版本）
import os, json, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    get_cosine_schedule_with_warmup,  # ⬅ 新增：余弦调度器
)

from lora_layer import LoRALinear
from model_patch import patch_blip1_visual_transformer

# =========== 1. 配置参数 ===========
MODEL_NAME   = "Salesforce/blip-image-captioning-large"
JSONL_PATH   = "./annotations.jsonl"
IMAGE_ROOT   = "./"
OUTPUT_DIR   = "./lora_blip_output"

BATCH_SIZE   = 2          # 单次显存可承受 batch
NUM_EPOCHS   = 50          # 训练轮数
LR           = 1e-4       # 初始学习率

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 新增混合精度 / 调度 / 梯度累积 ====
USE_FP16          = True          # 一键切换 FP16
GRAD_ACCUM_STEPS  = 4             # 累 n个 小 batch 再 step → 等效大 batch
WARMUP_STEPS      = 100           # 线性 warm-up 步数

# =========== 2. 数据集类 ===========
class BlockDataset(Dataset):
    def __init__(self, jsonl_path, image_root, processor):
        self.items, self.processor, self.image_root = [], processor, image_root
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                img = data["image"] if "image" in data else data["image_path"]
                cap = data["caption"] if "caption" in data else data["text"]
                self.items.append((os.path.join(image_root, img), cap))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        proc = self.processor(
            images=image, text=caption,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in proc.items()}
        item["labels"] = item["input_ids"].clone()
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

# =========== 3. 训练主逻辑 ===========
def main():
    print(f"加载模型: {MODEL_NAME}")
    model      = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor  = BlipProcessor.from_pretrained(MODEL_NAME)

    # 2. 打 LoRA Patch
    print(f"打LoRA Patch到视觉主干（r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}）")
    patch_blip1_visual_transformer(
        model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, verbose=True
    )

    # 3. 冻结非 LoRA 参数
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n or "lora_B" in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数量: {sum(p.numel() for p in trainable):,}")

    # 4. 数据加载
    dataset   = BlockDataset(JSONL_PATH, IMAGE_ROOT, processor)
    dataloader= DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, collate_fn=collate_fn)

    # ================== 优化器 & 调度器 ==================
    optimizer  = torch.optim.AdamW(trainable, lr=LR)
    total_steps= (len(dataloader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    print(f"总优化步数: {total_steps}  |  warm-up 步数: {WARMUP_STEPS}")

    # AMP 设置
    if USE_FP16 and DEVICE == "cuda":
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print(">> 已启用混合精度 (FP16)")
    else:
        scaler = None

    # ================== 训练循环 ==================
    model.to(DEVICE)
    model.train()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader)):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids    = batch["input_ids"].to(DEVICE)
            attn_mask    = batch["attention_mask"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            if scaler:  # FP16
                with autocast():
                    out  = model(pixel_values=pixel_values,
                                 input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 labels=labels)
                    loss = out.loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:       # FP32
                out  = model(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attn_mask,
                             labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS
                loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            # === 梯度累积到指定步再更新 ===
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")

    # ================== 保存 LoRA 权重 ==================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"lora_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_lr{LR}_r{LORA_R}_a{LORA_ALPHA}_{ts}.pth"
    save_path = os.path.join(OUTPUT_DIR, name)
    torch.save({k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora_A" in k or "lora_B" in k},
               save_path)
    print(f"[✓] LoRA Adapter 已保存到: {save_path}")

if __name__ == "__main__":
    main()


#
# # train_lora_blip_manual.py
# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm
# from datetime import datetime  # 新增
#
# from transformers import BlipForConditionalGeneration, BlipProcessor
#
# from lora_layer import LoRALinear
# from model_patch import patch_blip1_visual_transformer
#
# # =========== 1. 配置参数 ===========
# MODEL_NAME = "Salesforce/blip-image-captioning-large"
# JSONL_PATH = "./annotations.jsonl"        # 你的训练集，内容如：{"image": "xxx.png", "caption": "a blue block on a red block"}
# IMAGE_ROOT = "./"                         # 图片文件夹根目录
# OUTPUT_DIR = "./lora_blip_output"         # 保存LoRA权重的目录
# # BATCH_SIZE = 8
# # NUM_EPOCHS = 10
# BATCH_SIZE = 2
# NUM_EPOCHS = 1
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # LORA_R = 8
# # LORA_ALPHA = 16
# LORA_R = 16
# LORA_ALPHA = 32
# LORA_DROPOUT = 0.05
#
# # =========== 2. 数据集类 ===========
# class BlockDataset(Dataset):
#     def __init__(self, jsonl_path, image_root, processor):
#         self.items = []
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 # 支持单/多字段
#                 image_path = data["image"] if "image" in data else data["image_path"]
#                 caption = data["caption"] if "caption" in data else data["text"]
#                 self.items.append((os.path.join(image_root, image_path), caption))
#         self.processor = processor
#
#     def __len__(self):
#         return len(self.items)
#
#     def __getitem__(self, idx):
#         image_path, caption = self.items[idx]
#         image = Image.open(image_path).convert("RGB")
#         processed = self.processor(images=image, text=caption, padding="max_length", truncation=True, return_tensors="pt")
#         # squeeze batch维
#         item = {k: v.squeeze(0) for k, v in processed.items()}
#         item["labels"] = item["input_ids"].clone()
#         return item
#
# def collate_fn(batch):
#     # 自动pad到最大长度
#     keys = batch[0].keys()
#     result = {}
#     for k in keys:
#         if isinstance(batch[0][k], torch.Tensor):
#             result[k] = torch.stack([item[k] for item in batch])
#     return result
#
# # =========== 3. 训练主逻辑 ===========
# def main():
#     # 1. 加载模型和processor
#     print(f"加载模型: {MODEL_NAME}")
#     model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
#     processor = BlipProcessor.from_pretrained(MODEL_NAME)
#
#     # 2. 打 LoRA Patch
#     print(f"打LoRA Patch到视觉主干（r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}）")
#     patch_blip1_visual_transformer(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, verbose=True)
#
#     # 3. 冻结所有非 LoRA 参数（只训练LoRA参数）
#     for name, param in model.named_parameters():
#         if not ("lora_A" in name or "lora_B" in name):
#             param.requires_grad = False
#     lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
#     print(f"可训练参数量: {sum(p.numel() for p in lora_params):,}")
#
#     # 4. 加载数据
#     dataset = BlockDataset(JSONL_PATH, IMAGE_ROOT, processor)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
#
#     model.to(DEVICE)
#     model.train()
#
#     optimizer = torch.optim.AdamW(lora_params, lr=LR)
#
#     # =========== 5. 训练循环 ===========
#     for epoch in range(NUM_EPOCHS):
#         print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#         running_loss = 0.0
#         for batch in tqdm(dataloader):
#             pixel_values = batch["pixel_values"].to(DEVICE)
#             input_ids = batch["input_ids"].to(DEVICE)
#             labels = batch["labels"].to(DEVICE)
#             # BLIP1 forward 不会有 inputs_embeds 问题
#             outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
#             loss = outputs.loss
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         avg_loss = running_loss / len(dataloader)
#         print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")
#
#     # =========== 6. 保存LoRA权重 ===========
#     # 构造带时间戳、BATCH_SIZE、NUM_EPOCHS、LR 信息的文件名
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"lora_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_lr{LR}_{timestamp}.pth"
#     output_path = os.path.join(OUTPUT_DIR, filename)
#
#     # 只保存 lora_A/lora_B 参数字典，部署时合成/加载即可
#     lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
#     torch.save(lora_state, output_path)
#     print(f"LoRA Adapter 权重已保存到: {output_path}")
#
# if __name__ == "__main__":
#     main()


# # train_lora_blip_manual.py
# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm
# from datetime import datetime
#
# from transformers import BlipForConditionalGeneration, BlipProcessor
#
# from lora_layer import LoRALinear
# from model_patch import patch_blip1_visual_transformer
#
# # =========== 1. 配置参数 ===========
# MODEL_NAME = "Salesforce/blip-image-captioning-large"
# JSONL_PATH = "./annotations.jsonl"        # 你的训练集，内容如：{"image": "xxx.png", "caption": "a blue block on a red block"}
# IMAGE_ROOT = "./"              # 图片文件夹根目录
# OUTPUT_LORA_WEIGHTS = "./lora_blip_output/"
# # BATCH_SIZE = 8
# # NUM_EPOCHS = 10
# BATCH_SIZE = 2
# NUM_EPOCHS = 1
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LORA_R = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.05
#
# # =========== 2. 数据集类 ===========
# class BlockDataset(Dataset):
#     def __init__(self, jsonl_path, image_root, processor):
#         self.items = []
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 # 支持单/多字段
#                 image_path = data["image"] if "image" in data else data["image_path"]
#                 caption = data["caption"] if "caption" in data else data["text"]
#                 self.items.append((os.path.join(image_root, image_path), caption))
#         self.processor = processor
#
#     def __len__(self):
#         return len(self.items)
#
#     def __getitem__(self, idx):
#         image_path, caption = self.items[idx]
#         image = Image.open(image_path).convert("RGB")
#         processed = self.processor(images=image, text=caption, padding="max_length", truncation=True, return_tensors="pt")
#         # squeeze batch维
#         item = {k: v.squeeze(0) for k, v in processed.items()}
#         item["labels"] = item["input_ids"].clone()
#         return item
#
# def collate_fn(batch):
#     # 自动pad到最大长度
#     keys = batch[0].keys()
#     result = {}
#     for k in keys:
#         if isinstance(batch[0][k], torch.Tensor):
#             result[k] = torch.stack([item[k] for item in batch])
#     return result
#
# # =========== 3. 训练主逻辑 ===========
# def main():
#     # 1. 加载模型和processor
#     print(f"加载模型: {MODEL_NAME}")
#     model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
#     processor = BlipProcessor.from_pretrained(MODEL_NAME)
#
#     # 2. 打 LoRA Patch
#     print(f"打LoRA Patch到视觉主干（r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}）")
#     patch_blip1_visual_transformer(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, verbose=True)
#
#     # 3. 冻结所有非 LoRA 参数（只训练LoRA参数）
#     for name, param in model.named_parameters():
#         if not ("lora_A" in name or "lora_B" in name):
#             param.requires_grad = False
#     lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
#     print(f"可训练参数量: {sum(p.numel() for p in lora_params):,}")
#
#     # 4. 加载数据
#     dataset = BlockDataset(JSONL_PATH, IMAGE_ROOT, processor)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
#
#     model.to(DEVICE)
#     model.train()
#
#     optimizer = torch.optim.AdamW(lora_params, lr=LR)
#
#     # =========== 5. 训练循环 ===========
#     for epoch in range(NUM_EPOCHS):
#         print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#         running_loss = 0.0
#         for batch in tqdm(dataloader):
#             pixel_values = batch["pixel_values"].to(DEVICE)
#             input_ids = batch["input_ids"].to(DEVICE)
#             labels = batch["labels"].to(DEVICE)
#             # BLIP1 forward不会有inputs_embeds问题！
#             # outputs = model(pixel_values=pixel_values, labels=labels)
#             outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
#             loss = outputs.loss
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         avg_loss = running_loss / len(dataloader)
#         print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")
#
#     # =========== 6. 保存LoRA权重 ===========
#     # 只保存 lora_A/lora_B 参数字典，部署时合成/加载即可
#     lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
#     torch.save(lora_state, OUTPUT_LORA_WEIGHTS)
#     print(f"LoRA Adapter 权重已保存到: {OUTPUT_LORA_WEIGHTS}")
#
# if __name__ == "__main__":
#     main()
#
