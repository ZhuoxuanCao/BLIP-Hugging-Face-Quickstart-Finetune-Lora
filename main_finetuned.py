import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
from lora_layer import LoRALinear   # 你自己的LoRA层实现
from model_patch import patch_blip1_visual_transformer  # 你自己的模型patch工具

# =========== 配置 ===========
MODEL_NAME = "Salesforce/blip-image-captioning-large"
LORA_PATH = "./lora_blip_output/lora_bs2_ep50_lr0.0001_r16_a32_20250607_184641.pth"  # 你的LoRA权重文件
IMAGE_PATH = "./image_test/Snipaste_2025-06-06_15-54-15.png"  # 待推理的图片路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========== 1. 加载模型 ===========
print("加载BLIP1大模型...")
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
processor = BlipProcessor.from_pretrained(MODEL_NAME)

# =========== 2. Patch LoRA ===========
print("给视觉主干打LoRA Patch...")
patch_blip1_visual_transformer(model, r=16, alpha=32, dropout=0.05, verbose=True)

# =========== 3. 加载LoRA权重 ===========
print(f"加载LoRA Adapter权重: {LORA_PATH}")
lora_state = torch.load(LORA_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(lora_state, strict=False)
if len(missing) > 0:
    print("[警告] 有参数未被加载:", missing)
if len(unexpected) > 0:
    print("[警告] 有多余参数:", unexpected)

model = model.to(DEVICE)
model.eval()

# =========== 4. 推理 ===========
# 示例：对一张图片进行caption生成
image = Image.open(IMAGE_PATH).convert("RGB")
prompt = ""  # 可以为空，或自定义问题

inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    output_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"] if "input_ids" in inputs else None,
        attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
        max_new_tokens=30
    )
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("\n>> 生成的描述:", caption)
