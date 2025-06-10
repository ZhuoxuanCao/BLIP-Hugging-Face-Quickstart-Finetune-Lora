# BLIP1-HF-Captioning

**Modular BLIP-1 Fine-Tuning Framework with LoRA (Lightweight Adapter Injection)** using Hugging Face Transformers.

## 1. Project Description

This project reimplements the **BLIP-1 fine-tuning pipeline from scratch**, offering full control over data loading, model patching, and training — **without relying on the original [LAVIS](https://github.com/salesforce/LAVIS) framework** or any high-level wrappers such as `peft`.

We directly build on the Hugging Face Transformers implementation of `blip-image-captioning-large`, and manually inject **LoRA adapters** into the vision backbone (ViT-based) for parameter-efficient fine-tuning.

> ⚠️ Unlike typical LoRA implementations using the [`peft`](https://github.com/huggingface/peft) library, we encountered parameter incompatibility issues.  
> Instead, we implement LoRA manually using `torch.nn.Parameter` and `torch.nn.functional.linear`, and inject the low-rank adaptation weights into selected `Linear` layers via **PyTorch forward hooks**. This gives us complete control and compatibility with BLIP-1's internal transformer design.

<div align="center">
  <img src="./img/blip1_arch.png" alt="BLIP-1 Architecture" width="700"/>
  <br />
  <em>Figure: BLIP-1 model architecture </em>
</div>

### Key Highlights

- **Full Code Ownership**: Everything — from dataset loading to loss calculation and training logic — is built manually for maximal flexibility.
- **LoRA Injection without PEFT**: Automatically patches 96 `nn.Linear` layers in the visual encoder via forward hooks.  
  LoRA parameters (`r`, `alpha`, `dropout`) are fully customizable and saved independently.
- **Modern Training Tricks**: Built-in support for cosine learning rate decay, linear warm-up, FP16 mixed precision, and gradient accumulation.
- **Prompt-Driven Captioning**: Supports free-form prompts and generates structured captions like *"A red block on a green block"*.

Whether you're a researcher exploring multimodal adaptation, or an engineer deploying captioning systems in robotics and smart manufacturing — this framework enables **modular, transparent and efficient fine-tuning** of BLIP-1.


## 2. LoRA Implementation Details

Instead of relying on external packages like `peft`, we manually implement **LoRA injection** in the vision backbone of BLIP-1 (ViT-based encoder). The process involves:

- Dynamically locating all `nn.Linear` layers inside the visual encoder (`vision_model`).
- Wrapping each Linear layer with LoRA adapters:  
  Each adapter is composed of two trainable matrices: **A (down-projection)** and **B (up-projection)**, injected into the forward pass.
- Applying **forward hooks** to add the LoRA term during inference and training:
  
$$
\text{Output} = W x + \alpha \cdot BAx
$$

Where:
- `r`: Rank of the adapter bottleneck
- `alpha`: Scaling factor for the LoRA update
- `dropout`: Optional dropout between A and B

```python
# simplified version
def apply_lora_to_linear(layer, r=8, alpha=16):
    A = nn.Parameter(torch.randn((r, in_features)))
    B = nn.Parameter(torch.randn((out_features, r)))
    ...
    def hook_fn(input, output):
        return output + (B @ (A @ input.T)).T * scale
    layer.register_forward_hook(hook_fn)
````

* This approach enables **adapter-style fine-tuning**, requiring only \~3M trainable parameters (vs. 300M+ in full fine-tuning).
* The adapter weights are saved separately and can be loaded into a frozen BLIP-1 model for downstream inference.

> See `model_patch.py` for full implementation.


## 3. Project Structure

This repository is a minimal and modular implementation for fine-tuning BLIP-1 with custom LoRA injection and Hugging Face support.

```

BLIP_HF/
├── image_train/                 # Training images
├── image_test/                  # Test images
├── annotations.jsonl            # Captions and image paths in JSONL format
├── lora_layer.py                # Core LoRA layer implementation (torch-based, no PEFT dependency)
├── model_patch.py               # Visual transformer patching logic for LoRA injection
├── train_lora_blip_manual.py    # Main training loop (manual training + LoRA injection + FP16 support)
├── main_finetuned.py            # Inference with fine-tuned LoRA adapters
├── main.py                      # Optional standalone inference without fine-tuned
├── requirements.txt             
├── README.md
└── LICENSE

```

This layout reflects a fully working BLIP fine-tuning and inference pipeline with LoRA injection, gradient accumulation, learning rate scheduling, and optional mixed-precision (fp16) support.

## 4. Installation and Environment Setup
This project relies on the latest source version of Hugging Face Transformers to support BLIP models such as blip-image-captioning-large, and requires a GPU-compatible installation of PyTorch.

⚠️ **It is strongly recommended to follow the steps below to manually configure your environment, especially to avoid errors related to missing arguments like num_query_tokens that may not be supported in pre-built transformers wheels.**

Create a Python environment (Recommended: Python 3.9 or 3.10)

With Conda：

```bash
conda create -n blip_env python=3.9 -y
conda activate blip_env
```

Install PyTorch (choose CUDA version based on your server; example: CUDA 11.8)

```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, refer to: https://pytorch.org/get-started/locally/

Install Transformers from Source + Other Dependencies

```bash
git clone https://github.com/huggingface/transformers.git ./src/transformers
pip install -e ./src/transformers[torch]

pip install -r requirements.txt
```

Verify Your Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```


## 5. Usage & Training

### Zero-Shot Inference (No Fine-Tuning)

To use the BLIP-1 model for image captioning without any fine-tuning (i.e., directly using the pretrained weights), simply run:

```bash
python main.py
````

This will perform image captioning on your test images using the original BLIP-1 model, without loading any LoRA adapters or custom weights.
You can modify the input image paths and prompts directly in `main.py` as needed.

### Training with LoRA

This project enables efficient fine-tuning of BLIP-1 models for image captioning using a **custom LoRA injection** mechanism. LoRA adapters are patched directly into the visual backbone using `model_patch.py` and `lora_layer.py`, without relying on external libraries like `peft`.

To start training, simply run:

```bash
python train_lora_blip_manual.py
````

**Key features:**

* Training arguments (batch size, learning rate, epochs, LoRA rank/alpha, warmup steps, mixed precision, gradient accumulation, etc.) can be modified at the top of `train_lora_blip_manual.py`.
* Training loss and progress are printed at each step.
* LoRA adapter weights are saved with a timestamp and key hyperparameters for easy management.

Example LoRA adapter output:

```
lora_blip_output/lora_bs2_ep4_lr0.0001_r16_a32_20250607_161536.pth
```

### Inference with Fine-Tuned LoRA Adapters

After training, you can perform image captioning with your custom fine-tuned LoRA adapters using:

```bash
python main_finetuned.py
```

**Notes:**

* The script will automatically load the BLIP-1 backbone and apply your chosen LoRA adapter weights.
* You can easily switch between different LoRA checkpoints by modifying the adapter path in `main_finetuned.py`.

### Example Output

After running inference, you should see outputs like:

```
Image: image_test/green_on_blue_0005.jpg
Caption: "A green cube on top of a blue cube."
```

---

**Advanced Tips:**

* For large-scale training or more robust results, you can enable mixed precision (FP16), cosine learning rate decay, and gradient accumulation—all implemented in the training script.
* Custom data augmentations, prompt engineering, or additional evaluation scripts can be added based on your research needs.

## 6. Custom Dataset Format

We use a [JSON Lines](https://jsonlines.org/) (`.jsonl`) file to define the training data.  
Each line is a JSON object containing an image path and a natural-language caption.

**Example (`annotations.jsonl`):**
```jsonl
{"image": "image_train/blue_0001.jpg", "text": "A single blue cube"}
{"image": "image_train/green_0001.jpg", "text": "A single green cube"}
{"image": "image_train/blue_on_green_0001.jpg", "text": "A blue cube stacked on a green cube"}
{"image": "image_train/none_0001.jpg", "text": "No block is present in the image"}
````

### Recommended Captioning Strategies

* **Be explicit and consistent**: Use clear, descriptive sentences (“A blue cube on a green cube”).
* **Encourage compositional generalization**:
  If your downstream task may include new combinations (e.g., “red on blue” when “red” and “blue” exist in training but not their combination), use systematic and compositional captions.
* **Negative samples**: If needed, include images with no target object and captions like `"No block is present in the image"` to teach the model to reject irrelevant or empty images.
* **Single and multi-object**: Describe both single objects and various stacking/combination scenarios.

> The more systematically and compositionally you annotate, the stronger the generalization to novel combinations in downstream inference.

### File Organization

* All image paths in the `.jsonl` should be **relative to the project root or the folder containing the images**.
* For best results, keep file names and captions informative and consistent.

```
image_train/
  blue_0001.jpg
  green_0001.jpg
  blue_on_green_0001.jpg
  none_0001.jpg
annotations.jsonl
```

### Open Dataset

We provide the full image dataset and the complete `annotations.jsonl` used for training in this repository.
You can find them in the [`image_train/`](./image_train/) and [`annotations.jsonl`](./annotations.jsonl) files.  
Feel free to use them for benchmarking, re-training, or method comparison.

> If you use our dataset, please cite this repository.

## 6. License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

---

## 🔍 Keywords

BLIP captioning, image-text dataset, Hugging Face Transformers, fine-tuning vision-language models, Lora, PyTorch training, beginner-friendly multimodal project, inference-ready pipeline, GPU-compatible training, custom annotation format, blip-image-captioning-large.
