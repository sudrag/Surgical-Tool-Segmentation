#!/usr/bin/env python
"""
segmentation_experiment.py

This script trains and evaluates two semantic segmentation models on a custom dataset.
Models used:
    1. DeepLabV3-ResNet50 (a convolutional segmentation model)
    2. SegFormer (a transformer-based segmentation model fine-tuned on ADE20K)

The dataset comprises synthetic and real images with corresponding masks.
Masks are remapped to 3 classes:
    0: Background, 1: Tools, 2: Organs

DeepLabV3 uses a transform that produces 256×256 tensor inputs.
SegFormer fine-tuning uses a transform that produces 512×512 inputs (converted to PIL images for processing).

Both models are evaluated using the macro Intersection over Union (IoU) metric.
"""

import sys
print("sys.path:", sys.path)
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.transforms.functional as TF
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import RandomHorizontalFlip  # using only horizontal flip for simplicity

# ------------------ Additional Imports for SegFormer Fine-Tuning ------------------
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ------------------ Global Variables and Category Map ------------------
# Mapping pixel values in masks to semantic classes:
# 0: Background, 1: Tools, 2: Organs.
CATEGORY_MAP = {
    **{i: 1 for i in range(1, 24)},  # Tools
    **{i: 2 for i in range(24, 29)},  # Organs
}

# ------------------ Dataset Class ------------------
class SISVSEDataset(Dataset):
    """
    Custom dataset for loading images and semantic masks.
    Expects images in JPEG format and masks in PNG format.
    Applies a transformation (if provided) to both image and mask.
    Masks are remapped to have:
        0: Background, 1: Tools, 2: Organs.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Only include images that have a corresponding mask
        self.images = [img for img in os.listdir(image_dir)
                       if os.path.exists(os.path.join(mask_dir, img.replace(".jpg", ".png")))]
        print(f"Found {len(self.images)} valid image-mask pairs in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            # Apply transformation to image (and convert to tensor)
            image = self.transform(image)
            # Convert mask to numpy array and remap labels
            mask = np.array(mask)
            remapped_mask = np.zeros_like(mask)
            for key, value in CATEGORY_MAP.items():
                remapped_mask[mask == key] = value
            # Resize mask to 256x256 using nearest neighbor interpolation to preserve labels
            mask = TF.resize(Image.fromarray(remapped_mask), (256, 256), interpolation=TF.InterpolationMode.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        return image, mask

# ------------------ Custom Collate Function for PIL Images ------------------
def pil_collate_fn(batch):
    """
    Collate function for datasets that return PIL images (for SegFormer fine-tuning).
    Returns a list of images and a list of masks.
    """
    images, masks = zip(*batch)
    return list(images), list(masks)

# ------------------ Visualization Function (Optional) ------------------
def visualize_batch(images, masks):
    """
    Visualize a batch of images and corresponding masks.
    Converts tensors to numpy arrays if needed.
    """
    import matplotlib.pyplot as plt
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0).numpy() if torch.is_tensor(images[i]) else np.array(images[i])
        mask = masks[i].numpy() if torch.is_tensor(masks[i]) else np.array(masks[i])
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")
        plt.show()

# ------------------ Training and Evaluation Functions for DeepLabV3 ------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    """
    Train DeepLabV3 model for semantic segmentation.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"DeepLab Training Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.long())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        print(f"DeepLab Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        val_iou = evaluate_model(model, val_loader)
        print(f"DeepLab Epoch [{epoch+1}/{num_epochs}], Val IoU: {val_iou:.4f}")
        scheduler.step()

def evaluate_model(model, dataloader):
    """
    Evaluate DeepLabV3 model using the macro IoU metric.
    """
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            for pred, mask in zip(preds, masks):
                iou_scores.append(jaccard_score(mask.flatten(), pred.flatten(), average="macro"))
    return sum(iou_scores) / len(iou_scores)

# ------------------ Training and Evaluation Functions for SegFormer Fine-Tuning ------------------
def train_segformer(model, train_loader, val_loader, criterion, optimizer, scheduler, image_processor, num_epochs=10):
    """
    Fine-tune the SegFormer model.
    Uses gradient accumulation every 2 steps to simulate a larger batch size.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (images, masks) in enumerate(tqdm(train_loader, desc=f"SegFormer Training Epoch {epoch+1}/{num_epochs}")):
            # Convert image tensors to PIL images for proper processing by the image processor
            images_pil = [TF.to_pil_image(img.cpu()) if torch.is_tensor(img) else img for img in images]
            inputs = image_processor(images=images_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Resize ground-truth masks to match model output size using nearest-neighbor interpolation
            import torch.nn.functional as F
            gt_masks = torch.stack(masks).to(device)  # shape: (batch, 256, 256)
            outputs = model(**inputs).logits  # shape: (batch, num_labels, H_out, W_out)
            _, _, H_out, W_out = outputs.shape
            gt_masks_resized = F.interpolate(gt_masks.unsqueeze(1).float(), size=(H_out, W_out), mode='nearest').squeeze(1).long()

            loss = criterion(outputs, gt_masks_resized)
            loss.backward()
            if (step + 1) % 2 == 0:  # Gradient accumulation every 2 steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
        print(f"SegFormer Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        val_iou = evaluate_segformer(model, val_loader, image_processor)
        print(f"SegFormer Epoch [{epoch+1}/{num_epochs}], Val IoU: {val_iou:.4f}")
        scheduler.step()

def evaluate_segformer(model, test_loader, feature_extractor):
    """
    Evaluate the SegFormer model using the macro IoU metric.
    Images are processed by the feature extractor before inference.
    """
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in test_loader:
            for img, gt_mask in zip(images, masks):
                if not torch.is_tensor(img):
                    img = ToTensor()(img)
                image_np = img.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                mean = np.array(feature_extractor.image_mean)
                std = np.array(feature_extractor.image_std)
                image_np = (image_np * std + mean)
                image_np = np.clip(image_np, 0, 1) * 255
                image_pil = Image.fromarray(image_np.astype(np.uint8))
                inputs = feature_extractor(images=image_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs).logits  # shape: (1, num_labels, H_out, W_out)
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                # Resize prediction to match ground-truth shape (256x256)
                preds_resized = np.array(Image.fromarray(preds.astype(np.uint8)).resize(gt_mask.shape, resample=Image.NEAREST))
                gt = gt_mask.cpu().numpy()
                iou = jaccard_score(gt.flatten(), preds_resized.flatten(), average="macro")
                iou_scores.append(iou)
    return np.mean(iou_scores)

# ------------------ Path Setup ------------------
manual_syn_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/manual_syn"
manual_syn_mask_dir = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/manual_syn"
synthetic_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/domain_random_syn"
synthetic_mask_dir = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/domain_random_syn"
real_train_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/real"
real_train_mask_dir1 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_1"
real_train_mask_dir2 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_2"
real_train_mask_dir3 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_3"
test_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/real"
test_mask_dir1 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_1"
test_mask_dir2 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_2"
test_mask_dir3 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_3"

# ------------------ Define Transforms ------------------
# DeepLabV3 transform: outputs tensor.
transform = Compose([
    Resize((256, 256)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# SegFormer training transform: outputs tensor, then converted to PIL in collate.
segformer_train_transform = Compose([
    Resize((512, 512)),
    RandomHorizontalFlip(p=0.5),
    ToTensor()
])

# ------------------ Load Datasets ------------------
# For DeepLabV3 training
manual_syn_dataset = SISVSEDataset(manual_syn_image_dir, manual_syn_mask_dir, transform=transform)
synthetic_dataset = SISVSEDataset(synthetic_image_dir, synthetic_mask_dir, transform=transform)
real_train_dataset1 = SISVSEDataset(real_train_image_dir, real_train_mask_dir1, transform=transform)
real_train_dataset2 = SISVSEDataset(real_train_image_dir, real_train_mask_dir2, transform=transform)
real_train_dataset3 = SISVSEDataset(real_train_image_dir, real_train_mask_dir3, transform=transform)
train_dataset = ConcatDataset([manual_syn_dataset, synthetic_dataset,
                                real_train_dataset1, real_train_dataset2, real_train_dataset3])
test_dataset1 = SISVSEDataset(test_image_dir, test_mask_dir1, transform=transform)
test_dataset2 = SISVSEDataset(test_image_dir, test_mask_dir2, transform=transform)
test_dataset3 = SISVSEDataset(test_image_dir, test_mask_dir3, transform=transform)
test_dataset = ConcatDataset([test_dataset1, test_dataset2, test_dataset3])

# For SegFormer fine-tuning, use segformer_train_transform.
manual_syn_dataset_seg = SISVSEDataset(manual_syn_image_dir, manual_syn_mask_dir, transform=segformer_train_transform)
synthetic_dataset_seg = SISVSEDataset(synthetic_image_dir, synthetic_mask_dir, transform=segformer_train_transform)
real_train_dataset1_seg = SISVSEDataset(real_train_image_dir, real_train_mask_dir1, transform=segformer_train_transform)
real_train_dataset2_seg = SISVSEDataset(real_train_image_dir, real_train_mask_dir2, transform=segformer_train_transform)
real_train_dataset3_seg = SISVSEDataset(real_train_image_dir, real_train_mask_dir3, transform=segformer_train_transform)
train_dataset_seg = ConcatDataset([manual_syn_dataset_seg, synthetic_dataset_seg,
                                    real_train_dataset1_seg, real_train_dataset2_seg, real_train_dataset3_seg])
# For SegFormer fine-tuning, create a test dataset that returns PIL images.
test_dataset_seg = ConcatDataset([
    SISVSEDataset(test_image_dir, test_mask_dir1, transform=segformer_train_transform),
    SISVSEDataset(test_image_dir, test_mask_dir2, transform=segformer_train_transform),
    SISVSEDataset(test_image_dir, test_mask_dir3, transform=segformer_train_transform)
])

# ------------------ Create DataLoaders ------------------
# DeepLabV3 DataLoaders (returning tensors)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
# SegFormer DataLoaders (returning PIL images via custom collate)
train_loader_seg = DataLoader(train_dataset_seg, batch_size=2, shuffle=True, collate_fn=pil_collate_fn)
test_loader_seg = DataLoader(test_dataset_seg, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

# ------------------ Model Setup for DeepLabV3 ------------------
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
deeplab_model = deeplabv3_resnet50(weights=weights)
deeplab_model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deeplab_model = deeplab_model.to(device)

# ------------------ Loss, Optimizer, Scheduler for DeepLabV3 ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deeplab_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# ------------------ (Optional) Train DeepLabV3 ------------------
train_model(deeplab_model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=30)
test_iou = evaluate_model(deeplab_model, test_loader)
print(f"DeepLabV3 Test IoU: {test_iou:.4f}")
torch.save(deeplab_model.state_dict(), "deepLab_model.pth")

# ------------------ Model Setup for SegFormer Fine-Tuning ------------------
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
segformer_model.config.num_labels = 3
hidden_dim = segformer_model.config.hidden_sizes[-1] if hasattr(segformer_model.config, "hidden_sizes") else 256
segformer_model.classifier = nn.Conv2d(hidden_dim, 3, kernel_size=1)
segformer_model = segformer_model.to(device)

# ------------------ Loss, Optimizer, Scheduler for SegFormer ------------------
criterion_seg = nn.CrossEntropyLoss()
optimizer_seg = optim.AdamW(segformer_model.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=30, eta_min=1e-6)

# ------------------ Train SegFormer Fine-Tuning ------------------
train_segformer(segformer_model, train_loader_seg, test_loader_seg, criterion_seg, optimizer_seg, scheduler_seg, image_processor, num_epochs=30)
segformer_iou = evaluate_segformer(segformer_model, test_loader_seg, image_processor)
print(f"SegFormer Test IoU after fine-tuning: {segformer_iou:.4f}")
torch.save(segformer_model.state_dict(), "segformer_model.pth")
