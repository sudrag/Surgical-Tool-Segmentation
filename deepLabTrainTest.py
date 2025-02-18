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
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter  # New import

# Mapping categories to tools (1) or organs (2)
CATEGORY_MAP = {
    **{i: 1 for i in range(1, 24)},  # Tools
    **{i: 2 for i in range(24, 29)},  # Organs
}


# Dataset Class
class SISVSEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Include only images with corresponding masks
        self.images = [
            img for img in os.listdir(image_dir)
            if os.path.exists(os.path.join(mask_dir, img.replace(".jpg", ".png")))
        ]
        print(f"Found {len(self.images)} valid image-mask pairs in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

            # Map mask categories to valid indices (0: background, 1: tools, 2: organs)
            mask = np.array(mask)
            remapped_mask = np.zeros_like(mask)
            for key, value in CATEGORY_MAP.items():
                remapped_mask[mask == key] = value
            mask = TF.resize(Image.fromarray(remapped_mask), (256, 256), interpolation=TF.InterpolationMode.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


# Visualization Function (if needed)
def visualize_batch(images, masks):
    import matplotlib.pyplot as plt  # Imported here for clarity
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0).numpy()  # Convert CHW to HWC
        mask = masks[i].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")
        plt.show()


# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.long())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Evaluate on validation data
        val_iou = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation IoU: {val_iou:.4f}")
        scheduler.step()


# Evaluation Function
def evaluate_model(model, dataloader):
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


# ========================
# Path Setup
# ------------------------

# Training data paths
manual_syn_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/manual_syn"
manual_syn_mask_dir = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/manual_syn"

synthetic_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/domain_random_syn"
synthetic_mask_dir = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/domain_random_syn"

# Real training data: images come from the "real" folder; masks are in the three real_train directories.
real_train_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/real"
real_train_mask_dir1 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_1"
real_train_mask_dir2 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_2"
real_train_mask_dir3 = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_train_3"

# Testing data paths: Use sean translation data (for example, the manual_syn version)
# Test dataset: using a validation set
test_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/real"
test_mask_dir1  = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_1"
test_mask_dir2  = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_2"
test_mask_dir3  = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/real_val_3"

# test_image_dir = "sisvse_dataset/miccai2022_sisvse_dataset/images/sean_spade_translation/sean/manual_syn"
# test_mask_dir = "sisvse_dataset/miccai2022_sisvse_dataset/semantic_masks/sean_spade_translation/sean/manual_syn"


# ========================
# Define Transforms
# ------------------------
transform = Compose([
    Resize((256, 256)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========================
# Load Datasets
# ------------------------

manual_syn_dataset = SISVSEDataset(manual_syn_image_dir, manual_syn_mask_dir, transform=transform)
synthetic_dataset   = SISVSEDataset(synthetic_image_dir, synthetic_mask_dir, transform=transform)

real_train_dataset1 = SISVSEDataset(real_train_image_dir, real_train_mask_dir1, transform=transform)
real_train_dataset2 = SISVSEDataset(real_train_image_dir, real_train_mask_dir2, transform=transform)
real_train_dataset3 = SISVSEDataset(real_train_image_dir, real_train_mask_dir3, transform=transform)

# Combine training datasets: synthetic + manual_syn + real training
train_dataset = ConcatDataset([
    manual_syn_dataset,
    synthetic_dataset,
    real_train_dataset1,
    real_train_dataset2,
    real_train_dataset3
])

# Load testing dataset (using sean translation data)
test_dataset1 = SISVSEDataset(test_image_dir, test_mask_dir1, transform=transform)
test_dataset2 = SISVSEDataset(test_image_dir, test_mask_dir2, transform=transform)
test_dataset3 = SISVSEDataset(test_image_dir, test_mask_dir3, transform=transform)


test_dataset = ConcatDataset([
    test_dataset1,
    test_dataset2,
    test_dataset3
])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ========================
# Model Setup
# ------------------------

weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)  # Adjust for 3 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


# Train the Model
train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,num_epochs=20)

# Evaluate on Test Data
test_iou = evaluate_model(model, test_loader)
print(f"Test IoU: {test_iou:.4f}")

# Save the Model
torch.save(model.state_dict(), "sisvse_model.pth")
