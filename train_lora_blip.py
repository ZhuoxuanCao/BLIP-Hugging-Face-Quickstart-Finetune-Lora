# '''我们的的过滤没问题，inputs_embeds不是手动传的，是PEFT自动带上的。
#
# PEFT当前对BLIP这种V+L结构模型支持不完善，包装forward时可能出错。
#
# 这也是为什么你在社区和github上找不到很多LoRA微调BLIP1 Large的实战案例——它并不像LoRA之于LLM/RoBERTa/BERT那样天然兼容。
#
# 如果你只是想让BLIP1 Large能微调解决你的任务，强烈建议直接进行全量微调（不是LoRA）。
#
# 如果你执着于LoRA，可以考虑：
#
# 将LoRA只注入视觉部分（不注入文本部分），或者
#
# 只用LoRA Adapter对单独的视觉transformer注入，而文本decoder不动，这种方式要定制代码，不能直接用peft工具链。
#
# 也可以尝试在github/peft repo发issue寻求官方支持，说明你的报错和需求。'''
#
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import os
# from datasets import load_dataset
# from transformers import (
#     BlipProcessor,
#     BlipForConditionalGeneration,
#     TrainingArguments,
#     Trainer
# )
# from peft import LoraConfig, get_peft_model, TaskType
# import torch
# from PIL import Image
#
# # ==== 用户可修改的配置部分 ====
# MODEL_NAME = "Salesforce/blip-image-captioning-large"  # HF 上的 BLIP1 Large 模型名
# JSONL_PATH = "./annotations.jsonl"                    # 已转换为标准 jsonl 格式的标注文件
# IMG_ROOT = "./"                                       # 图片根目录（jsonl 里 image 字段相对这一目录）
# OUTPUT_DIR = "./lora_blip_output"                     # 保存 LoRA 微调后权重的目录
# BATCH_SIZE = 2                                        # 每卡 Batch Size
# EPOCHS = 5                                            # 微调轮数
# LR = 1e-4                                             # 学习率
# MAX_SEQ_LENGTH = 32                                   # 文本最大长度（生成时最长 Token 数）
# # ================================
#
#
# class FilteredTrainer(Trainer):
#     """
#     自定义 Trainer：覆盖 compute_loss，
#     1. 只保留模型真正需要的 pixel_values 和 labels，
#     2. 并打印当前 batch 里所有传递字段和过滤后实际传给模型的字段，方便调试。
#     """
#
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         # 先打印当前 batch 传进来的所有字段
#         print(f"\n>> 当前 batch 传递给 Trainer 的字段: {list(inputs.keys())}")
#
#         # 只保留 BLIP1 forward 真正需要的两个字段：pixel_values 和 labels
#         filtered_inputs = {}
#         if "pixel_values" in inputs:
#             filtered_inputs["pixel_values"] = inputs["pixel_values"]
#         if "labels" in inputs:
#             filtered_inputs["labels"] = inputs["labels"]
#
#         # 再打印一次，确认模型最终收到哪些字段
#         print(f">> 过滤后真正传入模型的字段: {list(filtered_inputs.keys())}")
#
#         # 调用模型，计算 loss
#         outputs = model(**filtered_inputs)
#         loss = outputs.loss
#
#         return (loss, outputs) if return_outputs else loss
#
#
# def main():
#     # 1. 加载 jsonl 格式的数据集（每行一个 {"image": "...", "text": "..."}）
#     dataset = load_dataset("json", data_files=JSONL_PATH, split="train")
#
#     # 2. 加载 BLIP1 Large 对应的 Processor 和 基础模型
#     processor = BlipProcessor.from_pretrained(MODEL_NAME)
#     base_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
#
#     # 3. 配置 LoRA 参数（使用“self_attn.qkv”和“self_attn.projection”来匹配所有层）
#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias="none",
#         task_type=TaskType.SEQ_2_SEQ_LM,
#         target_modules=[
#             "self_attn.qkv",        # 匹配视觉编码器所有层的 qkv 线性层
#             "self_attn.projection", # 匹配视觉编码器所有层的 projection 线性层
#             # 如需同时微调 Text Decoder 的 Query/Value，可打开下面两行：
#             # "attention.self.query",
#             # "attention.self.value",
#         ]
#     )
#     model = get_peft_model(base_model, lora_config)
#     model.print_trainable_parameters()  # 输出所有可训练的 LoRA 参数，确认已生效
#
#     # 4. 定义单个样本的预处理函数：加载图片并与对应的文本一起处理
#     def preprocess(example):
#         image_path = os.path.join(IMG_ROOT, example["image"])
#         image = Image.open(image_path).convert("RGB")
#
#         # 使用 processor 同时处理图像和文本
#         proc_outputs = processor(
#             images=image,
#             text=example["text"],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=MAX_SEQ_LENGTH
#         )
#
#         # 这里只保留 pixel_values。 不再显式获取 input_ids 或 attention_mask
#         inputs = {}
#         inputs["pixel_values"] = proc_outputs["pixel_values"].squeeze(0)
#
#         # 将 input_ids 复制一份作为 labels，模型内部会自动生成 decoder_input_ids
#         inputs["labels"] = proc_outputs["input_ids"].squeeze(0).clone()
#
#         return inputs
#
#     # 5. 对整个数据集做 map 预处理，生成模型可直接输入的张量
#     dataset = dataset.map(
#         preprocess,
#         remove_columns=dataset.column_names,
#         num_proc=1  # 若 CPU 核心较多，可设成 2~4 提升速度
#     )
#
#     # 6. 配置训练参数
#     args = TrainingArguments(
#         output_dir=OUTPUT_DIR,                       # 微调后权重保存路径
#         per_device_train_batch_size=BATCH_SIZE,
#         num_train_epochs=EPOCHS,
#         learning_rate=LR,
#         save_strategy="epoch",                       # 每个 epoch 保存一次
#         save_total_limit=1,                          # 最多保留 1 个 checkpoint
#         logging_steps=10,                            # 每 10 步打印一次日志
#         fp16=True if torch.cuda.is_available() else False,
#         report_to="none"                             # 关闭 wandb 等日志上报
#     )
#
#     # 7. 创建自定义 Trainer，并开始训练
#     trainer = FilteredTrainer(
#         model=model,
#         args=args,
#         train_dataset=dataset
#     )
#
#     print("\n>>> LoRA Adapter 参数如下（可训练参数）:")
#     model.print_trainable_parameters()
#
#     print("\n>>> 开始 LoRA 微调训练 ...")
#     trainer.train()
#
#     # 8. 训练结束后保存 LoRA 微调后的权重
#     model.save_pretrained(OUTPUT_DIR)
#     print(f"\n>>> 训练完成，LoRA 微调后的模型权重已保存至：{OUTPUT_DIR}")
#
#
# if __name__ == "__main__":
#     main()
