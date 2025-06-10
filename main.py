# main.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def main():
    # 1. 加载 Hugging Face 的 BLIP1 Large 模型与处理器
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # 2. 指定待推理图片路径
    image_path = "./image_test/BlueUp5.jpg"  # 替换为你的图片路径
    image = Image.open(image_path).convert("RGB")

    # 3. 图像预处理
    inputs = processor(images=image, return_tensors="pt")

    # 4. 推理，生成描述
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)

    # 5. 解码并输出结果
    caption = processor.decode(output[0], skip_special_tokens=True)
    print("图片描述：", caption)

if __name__ == "__main__":
    main()
