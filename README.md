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
  <em>Figure: BLIP-1 model architecture (Image credit: Salesforce Research)</em>
</div>

### Key Highlights

- 🧠 **Full Code Ownership**: Everything — from dataset loading to loss calculation and training logic — is built manually for maximal flexibility.
- 🔌 **LoRA Injection without PEFT**: Automatically patches 96 `nn.Linear` layers in the visual encoder via forward hooks.  
  LoRA parameters (`r`, `alpha`, `dropout`) are fully customizable and saved independently.
- ⚙️ **Modern Training Tricks**: Built-in support for cosine learning rate decay, linear warm-up, FP16 mixed precision, and gradient accumulation.
- 🧪 **Prompt-Driven Captioning**: Supports free-form prompts and generates structured captions like *"A red block on a green block"*.

Whether you're a researcher exploring multimodal adaptation, or an engineer deploying captioning systems in robotics and smart manufacturing — this framework enables **modular, transparent and efficient fine-tuning** of BLIP-1.
