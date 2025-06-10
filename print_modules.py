# from transformers import BlipForConditionalGeneration
#
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
#
# for name, module in model.named_modules():
#     print(name)
#
# # python print_modules.py > all_modules.txt

import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
print("CUDA device count:", torch.cuda.device_count())

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
