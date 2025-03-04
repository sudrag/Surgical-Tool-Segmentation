# Surgical Tool Segmentation

This project implements and compares two deep learning models for **semantic segmentation** of surgical tools in real-world and synthetic surgical images. The goal is to accurately segment **surgical instruments and organs**, enabling applications such as **real-time tool tracking in surgery, robotic-assisted surgeries, and surgical simulation**.

## Overview

Two **state-of-the-art** semantic segmentation models are trained and evaluated on the **SISVSE dataset**:

1. **DeepLabV3-ResNet50** – A CNN-based segmentation model that uses **atrous convolutions** for multi-scale context aggregation.
2. **SegFormer (B2 Variant)** – A **transformer-based** segmentation model that captures long-range dependencies using a hierarchical attention mechanism.

Each model was **fine-tuned** on a **custom dataset** and evaluated on unseen images. Performance is measured using the **Intersection over Union (IoU) metric**.

---

## Dataset

This project uses the **SISVSE Dataset**, available on **Kaggle**:

- [SISVSE Dataset - Kaggle](https://www.kaggle.com/datasets/yjh4374/sisvse-dataset)

The dataset contains **synthetic and real** surgical images, each with corresponding **semantic segmentation masks**.

### Data Split:
- **Training Data:**
  - **Manual Synthetic Images:** 3,400 image-mask pairs
  - **Domain-Randomized Synthetic Images:** 1,236 image-mask pairs
  - **Real Surgical Images:** Three subsets with **3,375, 3,355, and 3,377** image-mask pairs each.

- **Test Data:**
  - **Real Validation Images:** Three subsets with **1,135, 1,155, and 1,133** image-mask pairs.

### Segmentation Labels:
Each pixel in the mask is remapped to:
- **0:** Background
- **1:** Surgical Tools
- **2:** Organs

### Preprocessing:
- **Masks are resized to 256×256 pixels** for evaluation.
- Image transformations (e.g., **resizing, flipping, normalization**) are applied.

---

## Model Architectures

### 1. DeepLabV3-ResNet50
- **CNN-based architecture** using **atrous spatial pyramid pooling (ASPP)**.
- **Pretrained on COCO/VOC datasets**.
- **Modified final layer:** Outputs **3 classes** (background, tools, organs).

#### Training Parameters:
- **Batch Size:** 4
- **Learning Rate:** 0.001 (Adam optimizer)
- **Weight Decay:** 1e-5
- **Scheduler:** StepLR (Step size = 10 epochs, Decay factor γ = 0.8)
- **Epochs:** 30

#### Results:
- **Final Test IoU:** **0.7927**

---

### 2. SegFormer (B2 Variant)
- **Transformer-based segmentation model** with **hierarchical multi-scale attention**.
- **Pretrained on ADE20K dataset**.
- **Modified classifier head:** Outputs **3 classes** (background, tools, organs).

#### Training Parameters:
- **Batch Size:** 2 (with gradient accumulation every 2 steps, effective batch size = 4)
- **Learning Rate:** 5e-5 (AdamW optimizer)
- **Weight Decay:** 1e-5
- **Scheduler:** CosineAnnealingLR (T_max = 30, η_min = 1e-6)
- **Epochs:** 30

#### Results:
- **Final Test IoU:** **0.7463**

---

## Experimental Setup

### Hardware:
- Training was conducted on an **NVIDIA GPU** (e.g., NVIDIA **RTX 3080 or better**).
- Users must ensure **sufficient GPU memory (≥ 8GB VRAM)** for training SegFormer.

### Dataset Organization:
Ensure the dataset is structured as follows:

```
sisvse_dataset/
├── miccai2022_sisvse_dataset/
│   ├── images/
│   │   ├── manual_syn/
│   │   ├── domain_random_syn/
│   │   ├── real/
│   ├── semantic_masks/
│   │   ├── manual_syn/
│   │   ├── domain_random_syn/
│   │   ├── real_train_1/
│   │   ├── real_train_2/
│   │   ├── real_train_3/
│   │   ├── real_val_1/
│   │   ├── real_val_2/
│   │   ├── real_val_3/
```

Ensure all directories contain **matching image-mask pairs**.

---

## Installation & Reproducibility

### Dependencies
To set up the environment, install the required packages:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install tqdm numpy scikit-learn pillow
```

### Running the Code
Clone the repository and run the script:

```bash
git clone https://github.com/your-repo/surgical-tool-segmentation.git
cd surgical-tool-segmentation
python DeepLab-Segformer_NNSegmentation.py
```

---

## Citations & References

1. **SISVSE Dataset:**
   - Y. Huang et al., "SISVSE Dataset: Surgical Instrument Segmentation and Variability Estimation," [Kaggle Dataset](https://www.kaggle.com/datasets/yjh4374/sisvse-dataset).

2. **DeepLabV3:**
   - Chen, Liang-Chieh, et al. *"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation."* ECCV 2018.
   - [Official PyTorch Implementation](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)

3. **SegFormer:**
   - Xie, Enze, et al. *"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers."* NeurIPS 2021.
   - [Hugging Face Model](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.


