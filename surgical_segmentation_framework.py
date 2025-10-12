"""
Patched Surgical Segmentation Framework
- Correct HF preprocessing per model (SegFormer/DeepLab)
- SAM evaluation: oracle-prompted (with GT) vs prompt-free modes, plus efficiency harness safety
- Do NOT skip all-zero predictions in metrics
- Proper HD95 (surface distance 95th percentile) per-class on binary masks
- VideoMAE/VideoSwin placeholder made batch-safe and processor cached
- Robust CUDA/CPU guards for timing/memory
- Dataset pairing: image <-> semantic mask loader supporting the given directory structure

Assumptions
- Images are RGB .png/.jpg; semantic masks are single-channel with integer labels [0..K]
- Background class id = 0

Note: This file is self-contained; replace your previous script with this one.
"""
from __future__ import annotations

import os
import time
import json
import math
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except Exception:
    DeepLabV3_ResNet50_Weights = None

# progress bar (tqdm optional)Note: This file is self-contained; replace your previous script with this one.

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import tempfile, shutil

# =========================
# Dataset utilities
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}



def _atomic_torch_save(obj, path: str):
    """Write to a temp file then atomically replace the target (POSIX-safe)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_ckpt_", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic replace
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def _estimate_model_bytes(state_dict: dict) -> int:
    total = 0
    for t in state_dict.values():
        if torch.is_tensor(t):
            total += t.numel() * t.element_size()
    return total

def _has_free_space(path: str, needed_bytes: int, safety_factor: float = 1.5) -> bool:
    d = os.path.dirname(path) or "."
    usage = shutil.disk_usage(d)
    return usage.free >= int(needed_bytes * safety_factor)



def _list_files(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


class SISVSEDataset(Dataset):
    """
    Pairs images with semantic masks from the provided directory structure.

    Expected structure (root=miccai2022_sisvse_dataset):
      images/
        domain_random_syn, manual_syn, real, sean_spade_translation/... (and subfolders)
      semantic_masks/
        domain_random_syn, manual_syn, real_train_*, real_val_*, sean_spade_translation/...

    We will scan the selected image subdirectories and find corresponding mask paths by mirroring
    the relative path under semantic_masks. You MUST provide which split of masks you intend to use
    (e.g., real_train_1 or real_val_1) when creating the dataset. For synthetic folders, the same
    subfolder names exist under semantic_masks.
    """

    def __init__(
        self,
        root: str,
        image_subroots: List[str],  # e.g., ["images/real", "images/domain_random_syn"]
        mask_root: Optional[str],   # e.g., "semantic_masks/real_train_1" or None for inference
        transform: Optional[callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.mask_root = mask_root

        img_files: List[str] = []
        for sub in image_subroots:
            img_dir = os.path.join(root, sub)
            img_files.extend(_list_files(img_dir))
        img_files.sort()
        self.img_files = img_files

        if mask_root is not None:
            # Build mapping from image path -> mask path by replacing the segment after 'images/'
            masks = []
            for ip in self.img_files:
                rel = os.path.relpath(ip, os.path.join(root, "images"))
                mp = os.path.join(root, mask_root, rel)
                # masks are usually png; if image ext differs, swap to .png if present
                if not os.path.exists(mp):
                    base, _ = os.path.splitext(mp)
                    candidate = base + ".png"
                    if os.path.exists(candidate):
                        mp = candidate
                masks.append(mp)
            self.mask_files = masks
        else:
            self.mask_files = [None] * len(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        mask_path = self.mask_files[idx]
        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            # ensure integer label map
            mask = Image.open(mask_path)
            if mask.mode != "I" and mask.mode != "L":
                mask = mask.convert("L")
        sample = {"image": img, "mask": mask, "image_path": img_path, "mask_path": mask_path}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =========================
# Transforms
# =========================

class ResizePair:
    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size  # (H, W)
    def __call__(self, sample: Dict):
        img: Image.Image = sample["image"]
        mask = sample.get("mask")
        # Resize image with bilinear, mask with nearest
        img = img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.size[1], self.size[0]), resample=Image.NEAREST)
        sample["image"], sample["mask"] = img, mask
        return sample

class ToTensor:
    def __call__(self, sample: Dict):
        img: Image.Image = sample["image"]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # C,H,W
        img_t = torch.from_numpy(arr)
        mask = sample.get("mask")
        if mask is not None:
            m = torch.from_numpy(np.array(mask, dtype=np.int64))  # H,W integers
        else:
            m = None
        sample["image"] = img_t
        sample["mask"] = m
        return sample


class Compose:
    def __init__(self, ts: List[callable]):
        self.ts = ts
    def __call__(self, sample):
        for t in self.ts:
            sample = t(sample)
        return sample


# =========================
# Model Adapters
# =========================

# SegFormer & Mask2Former
try:
    from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, AutoImageProcessor
except Exception:
    SegformerForSemanticSegmentation = None
    Mask2FormerForUniversalSegmentation = None
    AutoImageProcessor = None


class SegFormerAdapter(nn.Module):
    def __init__(self, model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512", num_labels: Optional[int] = None):
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise ImportError("transformers not available")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, use_safetensors=True)
        if num_labels is not None and self.model.config.num_labels != num_labels:
            if hasattr(self.model, "decode_head") and hasattr(self.model.decode_head, "classifier"):
                in_ch = self.model.decode_head.classifier.in_channels
                self.model.decode_head.classifier = nn.Conv2d(in_ch, num_labels, kernel_size=1)
            else:
                replaced = False
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.out_channels == self.model.config.num_labels:
                        in_ch = module.in_channels
                        new_head = nn.Conv2d(in_ch, num_labels, kernel_size=1)
                        parts = name.split('.')
                        obj = self.model
                        for p in parts[:-1]:
                            obj = getattr(obj, p)
                        setattr(obj, parts[-1], new_head)
                        replaced = True
                        break
                if not replaced:
                    raise RuntimeError("Could not locate SegFormer classifier head to resize; check transformers version.")
            self.model.config.num_labels = num_labels

    def forward(self, images: List[Image.Image] | torch.Tensor) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            imgs = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in images]
        else:
            imgs = images
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        logits = out.logits
        if "pixel_values" in inputs:
            H, W = imgs[0].size[1], imgs[0].size[0]
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class Mask2FormerAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-small-ade-semantic",  # semantic!
        num_labels: Optional[int] = None
    ):
        super().__init__()
        if Mask2FormerForUniversalSegmentation is None:
            raise ImportError("transformers not available")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, use_safetensors=True
        )
        if num_labels is not None:
            self.model.config.num_labels = num_labels

    def forward(self, images: List[Image.Image] | torch.Tensor) -> torch.Tensor:
        # Normalize inputs to PIL list
        if isinstance(images, torch.Tensor):
            imgs = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in images]
        else:
            imgs = images

        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Try with explicit task hint when supported
        try:
            out = self.model(**inputs, task_inputs={"task": "semantic"})
        except TypeError:
            out = self.model(**inputs)

        # 1) Direct semantic logits (some checkpoints/versions)
        for key in ("sem_seg", "logits", "pred_semantic_logits"):
            val = getattr(out, key, None) if not isinstance(out, dict) else out.get(key, None)
            if val is not None:
                logits = val
                break
        else:
            # 2) Universal/query outputs → fuse to semantic logits
            cls = getattr(out, "class_queries_logits", None) if not isinstance(out, dict) else out.get("class_queries_logits", None)
            msk = getattr(out, "masks_queries_logits", None) if not isinstance(out, dict) else out.get("masks_queries_logits", None)
            if cls is None or msk is None:
                # Helpful dump for debugging
                try:
                    details = list(out.keys()) if isinstance(out, dict) else [a for a in dir(out) if not a.startswith("_")]
                except Exception:
                    details = [type(out).__name__]
                raise RuntimeError(
                    "Mask2Former forward: could not find semantic logits in output.\n"
                    f"Available fields: {details}\n"
                    "Ensure a semantic checkpoint (…-ade-semantic) or fuse query outputs."
                )

            # Shapes: cls (B, Q, C), msk (B, Q, H, W)
            B, Q, C = cls.shape
            _, _, Hq, Wq = msk.shape

            # If config.num_labels exists and differs, trim/pad classes accordingly
            target_C = getattr(self.model.config, "num_labels", C) or C
            if C > target_C:
                cls = cls[..., :target_C]
                C = target_C

            # Fuse: logits_c(h,w) = logsumexp_q( cls[b,q,c] + msk[b,q,h,w] )
            # Expand and broadcast to (B, Q, C, H, W)
            cls_bc = cls.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)   # (B, C, Q, 1, 1)
            msk_bc = msk.unsqueeze(1)                                   # (B, 1, Q, H, W)
            sum_q = cls_bc + msk_bc                                     # (B, C, Q, H, W)
            logits = torch.logsumexp(sum_q, dim=2)                      # (B, C, H, W)

        # Resize to original image size
        H, W = imgs[0].size[1], imgs[0].size[0]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# DeepLabV3+ (torchvision)
try:
    import torchvision
    from torchvision.models.segmentation import deeplabv3_resnet50
except Exception:
    torchvision = None
    deeplabv3_resnet50 = None


class DeepLabAdapter(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        if deeplabv3_resnet50 is None:
            raise ImportError("torchvision not available")

        # Try to load official weights (for better transfer + right preprocessing)
        self.weights = None
        if DeepLabV3_ResNet50_Weights is not None:
            try:
                self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self.model = deeplabv3_resnet50(weights=self.weights)
            except Exception:
                self.model = deeplabv3_resnet50(weights="DEFAULT")
        else:
            # Older torchvision API fallback
            self.model = deeplabv3_resnet50(weights="DEFAULT")

        # Replace classifier head to match your dataset classes
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

        # Derive mean/std safely with fallback to ImageNet defaults
        if self.weights is not None and hasattr(self.weights, "meta"):
            meta = getattr(self.weights, "meta", {}) or {}
            mean = meta.get("mean", (0.485, 0.456, 0.406))
            std  = meta.get("std",  (0.229, 0.224, 0.225))
        else:
            mean = (0.485, 0.456, 0.406)
            std  = (0.229, 0.224, 0.225)

        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W) in [0,1] float. Apply ImageNet normalization expected by the backbone.
        """
        images = images.float().clamp(0, 1)
        images = (images - self._mean) / self._std
        out = self.model(images)         # keep grads
        logits = out["out"]              # (B,C,H,W)
        return logits

# SAM adapter (oracle vs prompt-free)
try:
    from transformers import SamModel, SamProcessor
except Exception:
    SamModel = None
    SamProcessor = None


class SAMAdapter(nn.Module):
    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        super().__init__()
        if SamModel is None:
            raise ImportError("transformers not available for SAM")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name, use_safetensors=True)

    @torch.no_grad()
    def forward(
        self,
        image: Image.Image,
        oracle_mask: Optional[np.ndarray] = None,
        prompt_free: bool = False,
    ) -> torch.Tensor:
        """Returns mask logits (1,H,W) for single image.
        - oracle_mask: if provided, derive a positive point from GT (centroid) and evaluate oracle-prompted.
        - prompt_free: if True and no oracle_mask, we feed an empty prompt and rely on SAM's no-prompt; returns low-quality masks.
        """
        if oracle_mask is not None and oracle_mask.max() > 0:
            ys, xs = np.where(oracle_mask > 0)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            inputs = self.processor(images=image, input_points=[[[cx, cy]]], return_tensors="pt")
        else:
            if prompt_free:
                inputs = self.processor(images=image, input_points=None, return_tensors="pt")
            else:
                # minimal dummy prompt at image center
                w, h = image.size
                inputs = self.processor(images=image, input_points=[[[w//2, h//2]]], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Pick the highest scoring mask if multiple are returned
        mask_logits = outputs.pred_masks.squeeze(0)  # (N,1,h,w)
        scores = outputs.iou_scores.squeeze(0).mean(dim=1) if hasattr(outputs, "iou_scores") else None
        if scores is not None:
            idx = torch.argmax(scores)
            mask = mask_logits[idx, 0]
        else:
            mask = mask_logits[0, 0]
        # Upsample to image size
        mask = F.interpolate(mask[None, None, ...], size=(image.height, image.width), mode="bilinear", align_corners=False)[0,0]
        return mask[None, ...]  # (1,H,W)


# =========================
# Metrics
# =========================

def per_class_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True) -> Tuple[List[float], float]:
    ious = []
    start_c = 1 if ignore_bg else 0
    for c in range(start_c, num_classes):
        p = (pred == c)
        g = (gt == c)
        inter = (p & g).sum()
        union = (p | g).sum()
        if union == 0:
            iou = 1.0 if g.sum() == 0 and p.sum() == 0 else 0.0
        else:
            iou = inter / union
        ious.append(float(iou))
    return ious, float(np.mean(ious) if ious else 0.0)


def dice_coef(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True) -> Tuple[List[float], float]:
    dices = []
    start_c = 1 if ignore_bg else 0
    for c in range(start_c, num_classes):
        p = (pred == c)
        g = (gt == c)
        denom = p.sum() + g.sum()
        d = (2 * (p & g).sum() / denom) if denom > 0 else (1.0 if g.sum() == 0 and p.sum() == 0 else 0.0)
        dices.append(float(d))
    return dices, float(np.mean(dices) if dices else 0.0)


def _boundary_coords(mask: np.ndarray) -> np.ndarray:
    """Return coordinates (N,2) of boundary pixels of a binary mask using 4-neighborhood.
    Works with numpy only.
    """
    h, w = mask.shape
    pad = np.pad(mask.astype(np.uint8), 1, mode='edge')
    # simple gradient to detect edges
    gx = pad[1:-1, 2:] != pad[1:-1, 1:-1]
    gy = pad[2:, 1:-1] != pad[1:-1, 1:-1]
    edge = (gx | gy)
    ys, xs = np.where(edge)
    return np.stack([ys, xs], axis=1) if len(ys) > 0 else np.zeros((0,2), dtype=int)


def hd95_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True) -> Tuple[List[float], float]:
    def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape[0] == 0 and b.shape[0] == 0:
            return 0.0
        if a.shape[0] == 0 or b.shape[0] == 0:
            return float('inf')
        # pairwise distances (N x M); to keep memory reasonable, do chunks if large
        N, M = a.shape[0], b.shape[0]
        chunk = 4096
        dists_ab = []
        for i in range(0, N, chunk):
            aa = a[i:i+chunk]
            # broadcast (k,1,2) vs (1,m,2)
            d2 = (aa[:, None, :] - b[None, :, :])
            d2 = np.sqrt((d2 ** 2).sum(axis=2))
            dists_ab.append(d2.min(axis=1))
        d_ab = np.concatenate(dists_ab, axis=0)
        dists_ba = []
        for j in range(0, M, chunk):
            bb = b[j:j+chunk]
            d2 = (bb[:, None, :] - a[None, :, :])
            d2 = np.sqrt((d2 ** 2).sum(axis=2))
            dists_ba.append(d2.min(axis=1))
        d_ba = np.concatenate(dists_ba, axis=0)
        sym = np.concatenate([d_ab, d_ba], axis=0)
        return float(np.percentile(sym, 95))

    hd_list = []
    start_c = 1 if ignore_bg else 0
    for c in range(start_c, num_classes):
        pa = (pred == c)
        ga = (gt == c)
        a = _boundary_coords(pa)
        b = _boundary_coords(ga)
        hd = hausdorff95(a, b)
        hd_list.append(hd)
    finite_vals = [d for d in hd_list if np.isfinite(d)]
    mean_hd = float(np.mean(finite_vals)) if finite_vals else 0.0
    return hd_list, mean_hd




# =========================
# Evaluation / Training harness (simplified)
# =========================

# ---- Publishing helpers: plots, tables, markdown ----
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_training_plots(history: List[dict], outdir: str):
    _ensure_dir(outdir)
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    miou = [h["val_mIoU"] for h in history]
    dice = [h["val_Dice"] for h in history]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.title('Training Loss')
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'loss_curve.png'), dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, miou, marker='o', label='mIoU')
    plt.plot(epochs, dice, marker='s', label='Dice')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics')
    plt.legend(); plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'val_metrics.png'), dpi=200); plt.close()

def save_classwise_bars(per_class_iou: List[float], per_class_dice: List[float], outdir: str):
    _ensure_dir(outdir)
    if not per_class_iou or not per_class_dice:
        return
    classes = list(range(len(per_class_iou)))
    plt.figure(figsize=(10,4))
    plt.bar(classes, per_class_iou)
    plt.xlabel('Class (without bg)'); plt.ylabel('IoU'); plt.title('Per-class IoU')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'per_class_iou.png'), dpi=200); plt.close()

    plt.figure(figsize=(10,4))
    plt.bar(classes, per_class_dice)
    plt.xlabel('Class (without bg)'); plt.ylabel('Dice'); plt.title('Per-class Dice')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'per_class_dice.png'), dpi=200); plt.close()

def save_qualitative_samples(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str, num_samples: int = 6, model_type: str = "segformer", sam_mode: str = "oracle"):
    model.eval()
    imgs_vis, gts_vis, preds_vis = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            if model_type == "segformer":
                pil_list = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in imgs]
                logits = model(pil_list)
                logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
                preds = torch.argmax(logits, dim=1)
                for i in range(imgs.size(0)):
                    if len(imgs_vis) >= num_samples: break
                    imgs_vis.append(imgs[i].cpu()); gts_vis.append(gts[i].cpu()); preds_vis.append(preds[i].cpu())
            elif model_type == "deeplab" or model_type == "mask2former":
                if model_type == "mask2former":
                    pil_list = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in imgs]
                    logits = model(pil_list)
                else:
                    logits = model(imgs)
                logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
                preds = torch.argmax(logits, dim=1)
                for i in range(imgs.size(0)):
                    if len(imgs_vis) >= num_samples: break
                    imgs_vis.append(imgs[i].cpu()); gts_vis.append(gts[i].cpu()); preds_vis.append(preds[i].cpu())
            elif model_type == "sam":
                # handle item-wise
                for i in range(imgs.size(0)):
                    if len(imgs_vis) >= num_samples: break
                    img_np = (imgs[i].clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
                    pil = Image.fromarray(img_np)
                    gt_i = gts[i]
                    oracle = gt_i.cpu().numpy() if sam_mode == "oracle" else None
                    mask_logits = model(pil, oracle_mask=oracle, prompt_free=(sam_mode=="prompt_free"))  # (1,H,W)
                    pred = (mask_logits[0] > 0).to(torch.int64)  # binary
                    imgs_vis.append(imgs[i].cpu()); gts_vis.append(gt_i.cpu()); preds_vis.append(pred.cpu())
            if len(imgs_vis) >= num_samples:
                break
    if not imgs_vis:
        return
    cols = 3
    rows = len(imgs_vis)
    plt.figure(figsize=(12, 4*rows))
    for r in range(rows):
        plt.subplot(rows, cols, r*cols+1)
        img = imgs_vis[r].permute(1,2,0).numpy()
        plt.imshow(np.clip(img,0,1));  plt.title('Input' if r==0 else ''); plt.axis('off')
        plt.subplot(rows, cols, r*cols+2)
        plt.imshow(gts_vis[r].numpy(), cmap='tab20'); plt.title('GT' if r==0 else ''); plt.axis('off')
        plt.subplot(rows, cols, r*cols+3)
        # Use tab20 for multiclass, binary will also show
        plt.imshow(preds_vis[r].numpy(), cmap='tab20'); plt.title('Pred' if r==0 else ''); plt.axis('off')
    _ensure_dir(os.path.dirname(out_path) or '.')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def write_markdown_report(outdir: str, split_name: str, num_classes: int, final_res: 'EvalResult'):
    md_path = os.path.join(outdir, 'report.md')
    lines = []
    lines.append(f"# SISVSE Training Report\n")
    lines.append(f"**Split:** `{split_name}`  ")
    lines.append(f"**Classes (incl. bg):** `{num_classes}`  \n")
    lines.append("\n## Final Validation Metrics\n")
    lines.append(f"- **mIoU:** {final_res.mean_iou:.4f}")
    lines.append(f"- **Dice:** {final_res.mean_dice:.4f}")
    lines.append(f"- **HD95:** {final_res.mean_hd95:.4f}\n")
    lines.append("\n## Curves\n")
    lines.append("![Loss](loss_curve.png)")
    lines.append("![Validation Metrics](val_metrics.png)\n")
    lines.append("## Per-class\n")
    lines.append("![Per-class IoU](per_class_iou.png)")
    lines.append("![Per-class Dice](per_class_dice.png)\n")
    lines.append("## Qualitative Examples\n")
    lines.append("![Qualitative Grid](qualitative_grid.png)\n")
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    return md_path
# ---- end publishing helpers ----

# (rest of evaluation harness continues)

@dataclass
class EvalResult:
    per_class_iou: List[float]
    mean_iou: float
    per_class_dice: List[float]
    mean_dice: float
    per_class_hd95: List[float]
    mean_hd95: float


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    model_type: str = "segformer",  # segformer|deeplab|mask2former|sam
    sam_mode: str = "oracle",       # oracle|prompt_free
    use_tqdm: bool = False,
) -> EvalResult:
    model.eval()
    iou_all: List[List[float]] = []
    dice_all: List[List[float]] = []
    hd_all: List[List[float]] = []

    iterator = tqdm(loader, total=len(loader), desc="Val", leave=False) if (use_tqdm and 'tqdm' in globals() and tqdm is not None) else loader
    for batch in iterator:
        if model_type == "sam":
            imgs = batch["image"]
            gts = batch.get("mask")
            for i in range(len(imgs)):
                img = imgs[i]
                if isinstance(img, torch.Tensor):
                    img = Image.fromarray((img.clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8))
                gt_np = None
                if gts is not None and isinstance(gts, torch.Tensor):
                    gt_np = gts[i].numpy()
                oracle = gt_np if (sam_mode == "oracle") else None
                mask_logits = model(img, oracle_mask=oracle, prompt_free=(sam_mode == "prompt_free"))
                pred = (mask_logits[0].cpu().numpy() > 0).astype(np.uint8)
                if gt_np is None:
                    continue
                gt_fg = (gt_np > 0).astype(np.uint8)
                iou_list, _ = per_class_iou(pred, gt_fg, num_classes=2, ignore_bg=False)
                dice_list, _ = dice_coef(pred, gt_fg, num_classes=2, ignore_bg=False)
                hd_list, _ = hd95_per_class(pred, gt_fg, num_classes=2, ignore_bg=False)
                iou_all.append(iou_list); dice_all.append(dice_list); hd_all.append(hd_list)
        else:
            imgs_t = batch["image"].to(device)
            gts_t = batch.get("mask")
            if gts_t is None:
                continue
            gts_t = gts_t.to(device)
            with torch.no_grad():
                if model_type in ("segformer", "mask2former"):
                    pil_list = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in imgs_t]
                    logits = model(pil_list)
                elif model_type == "deeplab":
                    logits = model(imgs_t)
                else:
                    raise ValueError("Unknown model_type")
            logits = F.interpolate(logits, size=gts_t.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)
            for i in range(preds.shape[0]):
                p = preds[i].cpu().numpy().astype(np.int32)
                g = gts_t[i].cpu().numpy().astype(np.int32)
                iou_list, _ = per_class_iou(p, g, num_classes)
                dice_list, _ = dice_coef(p, g, num_classes)
                hd_list, _ = hd95_per_class(p, g, num_classes)
                iou_all.append(iou_list); dice_all.append(dice_list); hd_all.append(hd_list)

    iou_arr = np.array(iou_all, dtype=float)
    dice_arr = np.array(dice_all, dtype=float)
    hd_arr = np.array(hd_all, dtype=float)
    per_class_iou_mean = np.nanmean(iou_arr, axis=0).tolist() if iou_arr.size else []
    per_class_dice_mean = np.nanmean(dice_arr, axis=0).tolist() if dice_arr.size else []
    per_class_hd_mean = np.nanmean(hd_arr, axis=0).tolist() if hd_arr.size else []
    return EvalResult(
        per_class_iou=per_class_iou_mean,
        mean_iou=float(np.nanmean(per_class_iou_mean) if per_class_iou_mean else 0.0),
        per_class_dice=per_class_dice_mean,
        mean_dice=float(np.nanmean(per_class_dice_mean) if per_class_dice_mean else 0.0),
        per_class_hd95=per_class_hd_mean,
        mean_hd95=float(np.nanmean(per_class_hd_mean) if per_class_hd_mean else 0.0),
    )


# =========================
# Efficiency harness (CPU/CUDA safe)
# =========================

def benchmark_latency_memory(run_fn, warmup: int = 3, iters: int = 10) -> Dict[str, Union[float, str]]:
    # Warmup
    for _ in range(warmup):
        _ = run_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = run_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    ms = np.mean(times) * 1000.0
    mem = 0
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()
    else:
        mem = "N/A"
    return {"latency_ms": float(ms), "max_mem_mb": mem}


# =========================
# Video model placeholder (batch-safe)
# =========================

class VideoMAEPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, videos: List[List[Image.Image]]):
        # Simply returns zeros of size (B, C, H, W) as placeholder logits
        B = len(videos)
        H, W = videos[0][0].size[1], videos[0][0].size[0]
        C = 3
        return torch.zeros((B, C, H, W))


# =========================
# Example setup (to be customized)
# =========================

def make_dataloaders(root: str, split: str = "real_train_1", batch_size: int = 2) -> DataLoader:
    # For training/validation, you need MASKS. Use the appropriate semantic_masks split name.
    transform = Compose([ResizePair((512,512)), ToTensor()])
    image_subroots = [
    os.path.join("images", "real"),
    os.path.join("images", "domain_random_syn"),
    os.path.join("images", "manual_syn"),
    os.path.join("images", "sean_spade_translation", "sean", "domain_random_syn"),
    os.path.join("images", "sean_spade_translation", "sean", "manual_syn"),
    os.path.join("images", "sean_spade_translation", "spade", "domain_random_syn"),
    os.path.join("images", "sean_spade_translation", "spade", "manual_syn"),
    ]
    ds = SISVSEDataset(root=root, image_subroots=image_subroots, mask_root=os.path.join("semantic_masks", split), transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=_collate_dict)


def _collate_dict(batch: List[Dict]):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    masks = [b.get("mask") for b in batch]

    # If any mask is None, it's safer to stop early for training:
    if any(m is None for m in masks):
        raise RuntimeError("Batch contains samples without masks; ensure your split only includes pairs with semantic masks.")

    masks_t = torch.stack([m for m in masks], dim=0)  # <-- list, not generator
    return {
        "image": imgs,
        "mask": masks_t,
        "image_path": [b.get("image_path") for b in batch],
        "mask_path": [b.get("mask_path") for b in batch],
    }


# =========================
# Train/Val split utilities
# =========================

def build_dataset_index(
    root: str,
    image_subroots: List[str],
    mask_search_roots: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Return list of (image_path, mask_path) pairs.
    We search for each image's corresponding mask by mirroring the relative path under each
    provided mask_search_root until a match is found. If not provided, we scan *all*
    subdirectories under `semantic_masks/`.
    """
    img_paths: List[str] = []
    for sub in image_subroots:
        img_dir = os.path.join(root, sub)
        img_paths.extend(_list_files(img_dir))
    img_paths.sort()

    if mask_search_roots is None:
        sem_root = os.path.join(root, "semantic_masks")
        # gather all leaf dirs under semantic_masks
        mask_search_roots = []
        for dp, dn, _ in os.walk(sem_root):
            # use every directory under semantic_masks as a search root
            if os.path.isdir(dp):
                # dp relative to semantic_masks
                rel = os.path.relpath(dp, root)
                if rel.startswith("semantic_masks"):
                    mask_search_roots.append(rel)

    pairs: List[Tuple[str, str]] = []
    for ip in img_paths:
        rel = os.path.relpath(ip, os.path.join(root, "images"))
        mask_path = None
        for mr in mask_search_roots:
            candidate = os.path.join(root, mr, rel)
            if os.path.exists(candidate):
                mask_path = candidate
                break
            base, _ = os.path.splitext(candidate)
            if os.path.exists(base + ".png"):
                mask_path = base + ".png"
                break
        if mask_path is not None:
            pairs.append((ip, mask_path))
    return pairs


def train_val_split(index: List[Tuple[str, str]], val_ratio: float = 0.2, seed: int = 42):
    random.Random(seed).shuffle(index)
    n = len(index)
    n_val = max(1, int(n * val_ratio))
    val = index[:n_val]
    train = index[n_val:]
    return train, val


def save_split(index: List[Tuple[str, str]], out_json: str):
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(index, f, indent=2)


def load_split(path: str) -> List[Tuple[str, str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return [(a, b) for a, b in data]


class SISVSEFromIndex(Dataset):
    def __init__(self, index: List[Tuple[str, str]], transform: Optional[callable] = None):
        self.index = index
        self.transform = transform
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        ip, mp = self.index[i]
        img = Image.open(ip).convert('RGB')
        mask = Image.open(mp)
        if mask.mode not in ("I", "L"):
            mask = mask.convert("L")
        sample = {"image": img, "mask": mask, "image_path": ip, "mask_path": mp}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =========================
# Training loop with checkpointing and validation
# =========================

def count_classes(index: List[Tuple[str, str]]) -> int:
    # Heuristic: scan up to 50 masks to find max label
    mx = 0
    for _, mp in index[:50]:
        if os.path.exists(mp):
            arr = np.array(Image.open(mp))
            mx = max(mx, int(arr.max()))
    return mx + 1  # include background


def make_loaders_from_index(train_index, val_index, batch_size=4, size: Tuple[int,int]=(512,512)):
    transform = Compose([ResizePair((512, 512)), ToTensor()])
    train_ds = SISVSEFromIndex(train_index, transform)
    val_ds = SISVSEFromIndex(val_index, transform)
    collate = _collate_dict
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    return train_loader, val_loader


def validate(model, loader, num_classes, device, model_type="segformer"):
    return evaluate_model(model, loader, num_classes=num_classes, device=device, model_type=model_type)


def train_with_val(
    root: str,
    image_subroots: List[str],
    split_json_dir: str = "splits",
    split_name: str = "real_custom",
    val_ratio: float = 0.2,
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 5e-5,
    backend: str = "segformer",  # segformer | deeplab | mask2former
    model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = None,
    resume: bool = True,
    seed: int = 42,
    retest_best: bool = True
):

    # Derive default checkpoint path per backend + model variant
    if ckpt_path is None:
        suffix = ""
        if backend == "segformer":
            suffix = "_" + model_name.split("/")[-1]  # e.g., _segformer-b2-finetuned-ade-512-512
        ckpt_path = os.path.join("checkpoints", f"{backend}{suffix}_last.pt")

    # Build or load split
    os.makedirs(split_json_dir, exist_ok=True)
    split_path = os.path.join(split_json_dir, f"{split_name}.json")

    if os.path.exists(split_path):
        all_index = load_split(split_path)
    else:
        index = build_dataset_index(root, image_subroots)
        if len(index) == 0:
            raise RuntimeError("No (image, mask) pairs found. Check your paths.")
        # store the *full* index (we will make a stable split below)
        save_split(index, split_path)
        all_index = index

    random.Random(seed).shuffle(all_index)
    train_index, val_index = train_val_split(all_index, val_ratio=val_ratio, seed=seed)

    num_classes = count_classes(train_index)
    print(f"Detected num_classes={num_classes}")

    train_loader, val_loader = make_loaders_from_index(train_index, val_index, batch_size=batch_size, size=(512, 512))

    # Model per backend
    if backend == "segformer":
        model = SegFormerAdapter(model_name=model_name, num_labels=num_classes)
    elif backend == "deeplab":
        model = DeepLabAdapter(num_classes=num_classes)
    elif backend == "mask2former":
        model = Mask2FormerAdapter(num_labels=num_classes)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    best_ckpt = ckpt_path.replace('last', 'best') if 'last' in ckpt_path else ckpt_path + '.best'
    best_miou = -1.0
    history: List[dict] = []
    results_dir = os.path.join("experiments", "results", split_name, backend)
    _ensure_dir(results_dir)


    # Resume if requested and available (tolerant to shape changes & missing ckpt)
    filtered = None
    state = None
    if resume and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception as e:
            bad = ckpt_path + ".corrupt"
            print(f"[WARN] Checkpoint {ckpt_path} unreadable ({e}). Renaming to {bad} and starting fresh.")
            try:
                os.replace(ckpt_path, bad)
            except Exception:
                pass
            state = None  # ensure it's None

    if state is not None:
        # load only matching shapes/keys (lets you change architectures safely)
        raw_state = state.get("model", state)
        curr = model.state_dict()
        filtered = {k: v for k, v in raw_state.items() if k in curr and curr[k].shape == v.shape}

        if filtered:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(
                f"Resumed from {ckpt_path} "
                f"(loaded {len(filtered)} tensors; missing={len(missing)}, unexpected={len(unexpected)})"
            )
            # Optimizer/scaler only if model weights actually loaded
            if "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                except Exception:
                    print("[WARN] Optimizer state incompatible; resetting optimizer.")
            if "scaler" in state:
                try:
                    scaler.load_state_dict(state["scaler"])
                except Exception:
                    print("[WARN] GradScaler state incompatible; resetting scaler.")
        else:
            print(f"[WARN] {ckpt_path}: no compatible tensors; starting fresh.")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(train_loader)
        if tqdm is not None:
            pbar = tqdm(train_loader, total=total, desc=f"Epoch {epoch}/{epochs}", leave=True, smoothing=0.1)
            for i, batch in enumerate(pbar, 1):
                imgs = batch["image"].to(device)
                gts = batch["mask"].long().to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    if backend in ("segformer", "mask2former"):
                        pil_list = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in imgs]
                        logits = model(pil_list)
                    else:
                        logits = model(imgs)
                    logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits, gts, ignore_index=255)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item()
                pbar.set_postfix(loss=float(loss.item()))
        else:
            for i, batch in enumerate(train_loader, 1):
                imgs = batch["image"].to(device)
                gts = batch["mask"].long().to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    if backend in ("segformer", "mask2former"):
                        pil_list = [Image.fromarray((x.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) for x in imgs]
                        logits = model(pil_list)
                    else:
                        logits = model(imgs)
                    logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits, gts, ignore_index=255)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item()
                if i % 10 == 1 or i == total:
                    pct = int(100 * i / total)
                    print(f"Epoch {epoch}/{epochs} [{i}/{total} - {pct}%] loss={loss.item():.4f}")
        train_loss = running / max(1, len(train_loader))

        # Validation
        print(f"Starting validation over {len(val_loader)} batches...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            eval_res = evaluate_model(model, val_loader, num_classes=num_classes, device=device, model_type=("segformer" if backend=="segformer" else ("deeplab" if backend=="deeplab" else "mask2former")), use_tqdm=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Validation done in {time.time()-t0:.2f}s")
        print(f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | val_mIoU={eval_res.mean_iou:.4f} | val_Dice={eval_res.mean_dice:.4f}")
        history.append({"epoch": int(epoch), "train_loss": float(train_loss), "val_mIoU": float(eval_res.mean_iou), "val_Dice": float(eval_res.mean_dice)})
        # Write a running CSV for reproducibility
        csv_path = os.path.join(results_dir, "training_metrics.csv")
        if epoch == 1 and os.path.exists(csv_path):
            os.remove(csv_path)
        with open(csv_path, 'a') as f:
            if epoch == 1:
                f.write("epoch,train_loss,val_mIoU,val_Dice\n")
            f.write(f"{epoch},{train_loss:.6f},{eval_res.mean_iou:.6f},{eval_res.mean_dice:.6f}\n")


        # Save last (atomic)
        _tmp = ckpt_path + ".tmp"
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict()},
            _tmp,
        )
        os.replace(_tmp, ckpt_path)

        # Save best (atomic)
        if eval_res.mean_iou > best_miou:
            best_miou = eval_res.mean_iou
            _tmp_best = best_ckpt + ".tmp"
            torch.save({"model": model.state_dict()}, _tmp_best)
            os.replace(_tmp_best, best_ckpt)

    # ----- after training: publishable artifacts -----
    try:
        save_training_plots(history, results_dir)

        if retest_best and os.path.exists(best_ckpt):   # <— guard
            print(f"Loading best checkpoint for reporting: {best_ckpt}", flush=True)
            state_best = torch.load(best_ckpt, map_location=device, weights_only=True)
            print("Best checkpoint loaded.", flush=True)
            model.load_state_dict(state_best.get("model", state_best), strict=False)
            print(f"Re-evaluating best checkpoint on val set ({len(val_loader)} batches)...", flush=True)
            with torch.no_grad():
                final_res = evaluate_model(
                    model, val_loader, num_classes=num_classes, device=device,
                    model_type=("segformer" if backend=="segformer" else ("deeplab" if backend=="deeplab" else "mask2former")),
                    use_tqdm=True
                )
            save_classwise_bars(final_res.per_class_iou, final_res.per_class_dice, results_dir)
            save_qualitative_samples(
                model, val_loader, device,
                os.path.join(results_dir, "qualitative_grid.png"),
                num_samples=6,
                model_type=("segformer" if backend=="segformer" else ("deeplab" if backend=="deeplab" else "mask2former"))
            )
            md = write_markdown_report(results_dir, f"{split_name} · {backend}", num_classes, final_res)
            print(f"Saved figures and report to {results_dir} (report.md)")
        else:
            print("[info] Skipping best-ckpt re-test/report (smoke test).")

    except Exception as e:
        print(f"[WARN] Could not generate plots/report: {e}")

    print(f"Training complete. [{backend}] Best mIoU={best_miou:.4f}. Best ckpt: {best_ckpt}. Last ckpt: {ckpt_path}")
    return {"best": best_ckpt, "last": ckpt_path, "results_dir": results_dir, "backend": backend}

# =========================
# Multi-model comparison helpers
# =========================

def load_model_for_eval(backend: str, num_classes: int, device: torch.device, ckpt_path: str,
                        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> nn.Module:
    def _normalize_and_load(inner_module: nn.Module, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        raw = state.get("model", state)

        # strip "model." prefix if present
        if any(k.startswith("model.") for k in raw.keys()):
            raw = {k[len("model."):]: v for k, v in raw.items()}

        # load only matching keys/shapes
        cur = inner_module.state_dict()
        filt = {k: v for k, v in raw.items() if k in cur and cur[k].shape == v.shape}
        missing, unexpected = inner_module.load_state_dict(filt, strict=False)
        print(f"[eval-load] {os.path.basename(ckpt_path)} → loaded {len(filt)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    if backend == "segformer":
        m = SegFormerAdapter(model_name=model_name, num_labels=num_classes).to(device)
        if ckpt_path and os.path.exists(ckpt_path):
            _normalize_and_load(m.model, ckpt_path)
        m.eval()
        return m

    elif backend == "deeplab":
        m = DeepLabAdapter(num_classes=num_classes).to(device)
        if ckpt_path and os.path.exists(ckpt_path):
            _normalize_and_load(m.model, ckpt_path)
        m.eval()
        return m

    elif backend == "mask2former":
        m = Mask2FormerAdapter(num_labels=num_classes).to(device)
        if ckpt_path and os.path.exists(ckpt_path):
            _normalize_and_load(m.model, ckpt_path)
        m.eval()
        return m

    elif backend in ("sam_oracle", "sam_promptfree"):
        m = SAMAdapter().to(device)
        m.eval()
        return m

    else:
        raise ValueError(f"Unknown backend {backend}")



def overlay_metric_curves(split_name: str, backends: List[str], root: str = "experiments/results") -> str:
    outdir = os.path.join(root, split_name, "comparison")
    _ensure_dir(outdir)
    # mIoU overlay
    plt.figure(figsize=(7,5))
    for b in backends:
        csv_path = os.path.join(root, split_name, b, "training_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        epochs, miou = [], []
        with open(csv_path, 'r') as f:
            f.readline()
            for line in f:
                e, loss, mi, di = line.strip().split(',')
                epochs.append(int(e)); miou.append(float(mi))
        if epochs:
            plt.plot(epochs, miou, marker='o', label=f"mIoU · {b}")
    plt.xlabel('Epoch'); plt.ylabel('mIoU'); plt.title(f'Validation mIoU — {split_name}')
    plt.legend(); plt.grid(True, linestyle=':', linewidth=0.5)
    out_miou = os.path.join(outdir, 'miou_overlay.png')
    plt.tight_layout(); plt.savefig(out_miou, dpi=200); plt.close()

    # Dice overlay
    plt.figure(figsize=(7,5))
    for b in backends:
        csv_path = os.path.join(root, split_name, b, "training_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        epochs, dice = [], []
        with open(csv_path, 'r') as f:
            f.readline()
            for line in f:
                e, loss, mi, di = line.strip().split(',')
                epochs.append(int(e)); dice.append(float(di))
        if epochs:
            plt.plot(epochs, dice, marker='s', label=f"Dice · {b}")
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title(f'Validation Dice — {split_name}')
    plt.legend(); plt.grid(True, linestyle=':', linewidth=0.5)
    out_dice = os.path.join(outdir, 'dice_overlay.png')
    plt.tight_layout(); plt.savefig(out_dice, dpi=200); plt.close()
    return outdir


def qualitative_comparison_grid(split_name: str, backends: List[str], root: str, val_loader: DataLoader, num_classes: int, device: torch.device, samples: int = 4, model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> str:
    # Load models (SAM variants don't have checkpoints)
    models = []
    for b in backends:
        if b in ("sam_oracle", "sam_promptfree"):
            models.append(load_model_for_eval(b, num_classes, device, ckpt_path="", model_name=model_name))
        else:
            best = os.path.join("checkpoints", f"{b}_best.pt")
            if not os.path.exists(best):
                best = os.path.join("checkpoints", f"{b}_last.pt")
            models.append(load_model_for_eval(b, num_classes, device, best, model_name=model_name))

    for m in models:
        m.eval()

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            for i in range(imgs.size(0)):
                if len(rows) >= samples:
                    break
                entry = {"img": imgs[i].cpu(), "gt": gts[i].cpu(), "preds": []}
                for mi, b in enumerate(backends):
                    if b in ("segformer", "mask2former"):
                        pil_list = [Image.fromarray((imgs[i].clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))]
                        logits = models[mi](pil_list)
                        logits = F.interpolate(logits, size=gts[i].shape[-2:], mode="bilinear", align_corners=False)
                        pred = torch.argmax(logits, dim=1)[0].cpu()
                    elif b == "deeplab":
                        logits = models[mi](imgs[i][None, ...])
                        logits = F.interpolate(logits, size=gts[i].shape[-2:], mode="bilinear", align_corners=False)
                        pred = torch.argmax(logits, dim=1)[0].cpu()
                    elif b in ("sam_oracle", "sam_promptfree"):
                        pil = Image.fromarray((imgs[i].clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                        oracle = gts[i].cpu().numpy() if b == "sam_oracle" else None
                        mask_logits = models[mi](pil, oracle_mask=oracle, prompt_free=(b=="sam_promptfree"))
                        pred = (mask_logits[0] > 0).to(torch.int64).cpu()  # binary
                    else:
                        raise ValueError(f"Unknown backend {b}")
                    entry["preds"].append(pred)
                rows.append(entry)
            if len(rows) >= samples:
                break

    outdir = os.path.join(root, split_name, "comparison")
    _ensure_dir(outdir)
    cols = 2 + len(backends)
    plt.figure(figsize=(4*cols, 4*len(rows)))
    for r, entry in enumerate(rows):
        plt.subplot(len(rows), cols, r*cols+1)
        plt.imshow(np.clip(entry["img"].permute(1,2,0).numpy(),0,1)); plt.title("Input"); plt.axis('off')
        plt.subplot(len(rows), cols, r*cols+2)
        plt.imshow(entry["gt"].numpy(), cmap='tab20'); plt.title("GT"); plt.axis('off')
        for j, b in enumerate(backends):
            plt.subplot(len(rows), cols, r*cols+3+j)
            plt.imshow(entry["preds"][j].numpy(), cmap='tab20'); plt.title(b); plt.axis('off')
    out_path = os.path.join(outdir, 'qualitative_comparison.png')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return out_path

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def _predict_logits_single(model, backend, img_t, H, W):
    if backend in ("segformer", "mask2former"):
        pil = Image.fromarray((img_t.clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
        logits = model([pil])
    elif backend == "deeplab":
        logits = model(img_t[None].to(next(model.parameters()).device))
    else:
        raise ValueError(f"Unsupported backend for robustness/consistency: {backend}")
    return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

def _miou_from_logits(logits, gt_np, num_classes):
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)
    vals, mean_iou = per_class_iou(pred, gt_np.astype(np.int32), num_classes)
    return float(mean_iou)

def _apply_aug(img_t, kind):
    # img_t: (3,H,W) in [0,1]
    if kind == "flip_h":
        return img_t.flip(-1)
    if kind == "flip_v":
        return img_t.flip(-2)
    if kind == "gauss_noise":
        n = torch.randn_like(img_t) * 0.08
        return (img_t + n).clamp(0,1)
    if kind == "brightness+":
        return (img_t * 1.2).clamp(0,1)
    if kind == "brightness-":
        return (img_t * 0.8).clamp(0,1)
    if kind == "crop_pad":
        # center crop 90% then pad back
        _, H, W = img_t.shape
        ch, cw = int(H*0.9), int(W*0.9)
        y0 = (H - ch)//2; x0 = (W - cw)//2
        crop = img_t[:, y0:y0+ch, x0:x0+cw]
        return F.interpolate(crop[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return img_t

def robustness_heatmap(split_name, backends, root, val_loader, num_classes, device, samples=64):
    """Returns path to heatmap png. Measures mIoU drop under simple perturbations."""
    augs = ["flip_h","flip_v","gauss_noise","brightness+","brightness-","crop_pad"]
    outdir = os.path.join(root, split_name, "comparison")
    _safe_mkdir(outdir)

    # collect images/GTs (subset for speed)
    imgs_bank, gts_bank = [], []
    for batch in val_loader:
        imgs_bank.append(batch["image"])
        gts_bank.append(batch["mask"])
        if sum(x.size(0) for x in imgs_bank) >= samples: break
    imgs = torch.cat(imgs_bank, 0)[:samples].to(device)
    gts  = torch.cat(gts_bank, 0)[:samples].cpu().numpy()

    # load eval models
    models = []
    for b in backends:
        if b in ("sam_oracle","sam_promptfree"):
            continue  # skip SAM here (binary + different API)
        best = os.path.join("checkpoints", f"{b}_best.pt")
        if not os.path.exists(best):
            best = os.path.join("checkpoints", f"{b}_last.pt")
        models.append((b, load_model_for_eval(b, num_classes, device, best)))

    # compute baseline and augmented mIoU averages per backend
    import pandas as pd
    mat = []
    for b, m in models:
        m.eval()
        # baseline
        base_scores = []
        with torch.no_grad():
            for i in range(len(imgs)):
                H, W = imgs[i].shape[-2:]
                logits = _predict_logits_single(m, b, imgs[i], H, W)
                base_scores.append(_miou_from_logits(logits, gts[i], num_classes))
        base = np.mean(base_scores)

        # aug rows (negative = drop)
        row = []
        with torch.no_grad():
            for a in augs:
                sc = []
                for i in range(len(imgs)):
                    aug = _apply_aug(imgs[i], a)
                    H, W = aug.shape[-2:]
                    logits = _predict_logits_single(m, b, aug, H, W)
                    sc.append(_miou_from_logits(logits, gts[i], num_classes))
                row.append(np.mean(sc) - base)
        mat.append(row)

    if not mat:
        return ""

    mat = np.array(mat)  # backends x augs
    # plot heatmap
    plt.figure(figsize=(1.6+1.2*len(augs), 0.8+0.6*len(models)))
    im = plt.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-0.25, vmax=0.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Δ mIoU (aug - base)")
    plt.yticks(range(len(models)), [b for b,_ in models])
    plt.xticks(range(len(augs)), augs, rotation=30, ha='right')
    plt.title(f"Robustness (Δ mIoU) — {split_name}")
    outp = os.path.join(outdir, "robustness_heatmap.png")
    plt.tight_layout(); plt.savefig(outp, dpi=200); plt.close()
    return outp

def temporal_consistency_plot(split_name, backends, root, val_loader, num_classes, device, samples=16):
    """
    Proxy temporal consistency: prediction stability under small spatial jitter on same frame.
    Score = mean IoU(pred, pred_jitter).
    """
    outdir = os.path.join(root, split_name, "comparison")
    _safe_mkdir(outdir)

    # collect a small pool
    pool = []
    for batch in val_loader:
        for i in range(batch["image"].size(0)):
            pool.append(batch["image"][i])
            if len(pool) >= samples: break
        if len(pool) >= samples: break
    imgs = torch.stack(pool, 0).to(device)

    def _jitter(x):
        # small translate + scale
        _, H, W = x.shape
        crop = x[:, 8:H-8, 8:W-8]
        return F.interpolate(crop[None], size=(H, W), mode="bilinear", align_corners=False)[0].clamp(0,1)

    rows = []
    labels = []
    with torch.no_grad():
        for b in backends:
            if b in ("sam_oracle","sam_promptfree"):
                continue
            best = os.path.join("checkpoints", f"{b}_best.pt")
            if not os.path.exists(best):
                best = os.path.join("checkpoints", f"{b}_last.pt")
            model = load_model_for_eval(b, num_classes, device, best)
            model.eval()
            scs = []
            for i in range(len(imgs)):
                x = imgs[i]; xj = _jitter(x)
                H, W = x.shape[-2:]
                logits = _predict_logits_single(model, b, x, H, W)
                pj = _predict_logits_single(model, b, xj, H, W)
                p = torch.argmax(logits, dim=1)[0].cpu().numpy()
                q = torch.argmax(pj,     dim=1)[0].cpu().numpy()
                _, s = per_class_iou(p, q, num_classes)  # IoU between predictions
                scs.append(s)
            rows.append(np.mean(scs))
            labels.append(b)

    if not rows:
        return ""

    # bar chart
    plt.figure(figsize=(6,4))
    plt.bar(range(len(labels)), rows)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Mean IoU(pred vs jittered pred)")
    plt.title(f"Temporal Consistency (proxy) — {split_name}")
    outp = os.path.join(outdir, "temporal_consistency.png")
    plt.tight_layout(); plt.savefig(outp, dpi=200); plt.close()
    return outp

def performance_comparison_bar(split_name, backends, root="experiments/results"):
    """Reads each backend's last-row metrics CSV and plots a bar chart for mIoU/Dice."""
    outdir = os.path.join(root, split_name, "comparison")
    _safe_mkdir(outdir)
    xs, mi, di = [], [], []
    for b in backends:
        csv_path = os.path.join(root, split_name, b, "training_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        last = None
        with open(csv_path, "r") as f:
            f.readline()
            for line in f:
                last = line.strip().split(",")
        if last and len(last) == 4:
            xs.append(b); mi.append(float(last[2])); di.append(float(last[3]))
    if not xs:
        return "", ""
    # mIoU
    plt.figure(figsize=(6,4))
    plt.bar(range(len(xs)), mi)
    plt.xticks(range(len(xs)), xs)
    plt.ylabel("mIoU")
    plt.title(f"Performance (mIoU) — {split_name}")
    out_m = os.path.join(outdir, "performance_miou.png")
    plt.tight_layout(); plt.savefig(out_m, dpi=200); plt.close()
    # Dice
    plt.figure(figsize=(6,4))
    plt.bar(range(len(xs)), di)
    plt.xticks(range(len(xs)), xs)
    plt.ylabel("Dice")
    plt.title(f"Performance (Dice) — {split_name}")
    out_d = os.path.join(outdir, "performance_dice.png")
    plt.tight_layout(); plt.savefig(out_d, dpi=200); plt.close()
    return out_m, out_d

def ensure_per_model_qualitative(split_name: str, backends: List[str], root: str, val_loader: DataLoader, num_classes: int, device: torch.device, samples: int = 6, model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
    """
    Make sure each backend has its own qualitative_grid.png under:
      experiments/results/<split>/<backend>/qualitative_grid.png

    If missing, we load the model and generate it from the current val_loader.
    """
    for b in backends:
        out_dir = os.path.join(root, split_name, b)
        out_path = os.path.join(out_dir, "qualitative_grid.png")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_path):
            continue

        try:
            if b in ("sam_oracle", "sam_promptfree"):
                # eval-only SAM
                model = load_model_for_eval(b, num_classes, device, ckpt_path="", model_name=model_name)
                sam_mode = "oracle" if b == "sam_oracle" else "prompt_free"
                save_qualitative_samples(model, val_loader, device, out_path, num_samples=samples, model_type="sam", sam_mode=sam_mode)
            else:
                # trained backends
                best = os.path.join("checkpoints", f"{b}_best.pt")
                if not os.path.exists(best):
                    best = os.path.join("checkpoints", f"{b}_last.pt")
                model = load_model_for_eval(b, num_classes, device, best, model_name=model_name)
                model_type = "segformer" if b=="segformer" else ("deeplab" if b=="deeplab" else "mask2former")
                save_qualitative_samples(model, val_loader, device, out_path, num_samples=samples, model_type=model_type)
        except Exception as e:
            print(f"[WARN] Could not create qualitative grid for {b}: {e}")


def write_comparison_report(split_name: str, backends: List[str], root: str = "experiments/results") -> str:
    comp_dir = os.path.join(root, split_name, "comparison")
    _ensure_dir(comp_dir)
    md = os.path.join(comp_dir, 'comparison_report.md')

    lines = [f"# Model Comparison — {split_name}\n"]
    lines.append("## Summary Metrics (best epoch where available)\n")
    lines.append("Model | mIoU | Dice | Notes | Curves\n")
    lines.append("---|---:|---:|---|---\n")
    for b in backends:
        csv_path = os.path.join(root, split_name, b, "training_metrics.csv")
        mi, di = None, None
        note = ""
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                f.readline()
                for line in f:
                    e, loss, miou, dice = line.strip().split(',')
                    mi = float(miou); di = float(dice)
        else:
            if b in ("sam_oracle", "sam_promptfree"):
                note = "SAM eval-only (binary)"
        lines.append(f"{b} | {mi if mi is not None else 'NA'} | {di if di is not None else 'NA'} | {note} | [mIoU](miou_overlay.png), [Dice](dice_overlay.png)\n")

    def _exists(name):
        return os.path.exists(os.path.join(comp_dir, name))

    lines.append("\n## Overlaid Curves\n")
    if _exists("miou_overlay.png"):  lines.append("![mIoU Overlay](miou_overlay.png)\n")
    if _exists("dice_overlay.png"):  lines.append("![Dice Overlay](dice_overlay.png)\n")

    lines.append("\n## Performance Bars\n")
    if _exists("performance_miou.png"): lines.append("![Performance mIoU](performance_miou.png)\n")
    if _exists("performance_dice.png"): lines.append("![Performance Dice](performance_dice.png)\n")

    lines.append("\n## Robustness Heatmap (Δ mIoU vs baseline)\n")
    if _exists("robustness_heatmap.png"): lines.append("![Robustness](robustness_heatmap.png)\n")

    lines.append("\n## Temporal Consistency (Prediction Stability)\n")
    if _exists("temporal_consistency.png"): lines.append("![Temporal Consistency](temporal_consistency.png)\n")

    # Combined qualitative comparison (all models in one grid)
    lines.append("\n## Qualitative Comparison (Side-by-side)\n")
    if _exists("qualitative_comparison.png"): lines.append("![Qualitative](qualitative_comparison.png)\n")

    # NEW: Per-model qualitative grids
    lines.append("\n## Per-model Qualitative Grids\n")
    for b in backends:
        rel = os.path.join("..", b, "qualitative_grid.png")  # relative from comparison/ -> backend/
        if os.path.exists(os.path.join(root, split_name, b, "qualitative_grid.png")):
            lines.append(f"### {b}\n")
            lines.append(f"![{b} qualitative]({rel})\n")

    with open(md, 'w') as f:
        f.write('\n'.join(lines))
    return md


# =========================
# Main: train all models in sequence & compare
# =========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # >>> EDIT THIS: dataset root that contains images/ and semantic_masks/
    root = "./sisvse_dataset/miccai2022_sisvse_dataset"

    image_subroots = [
        os.path.join("images", "real"),
        os.path.join("images", "domain_random_syn"),
        os.path.join("images", "manual_syn"),
        os.path.join("images", "sean_spade_translation", "sean", "domain_random_syn"),
        os.path.join("images", "sean_spade_translation", "sean", "manual_syn"),
        os.path.join("images", "sean_spade_translation", "spade", "domain_random_syn"),
        os.path.join("images", "sean_spade_translation", "spade", "manual_syn"),
    ]

    split_name = "real_custom"
    backends = ["segformer", "deeplab", "mask2former"]
    # SAM eval-only backends (not trained): include in comparison grid and table
    eval_only_backends = []
    try:
        _ = SAMAdapter()  # quick import/availability check
        eval_only_backends = ["sam_oracle", "sam_promptfree"]
    except Exception:
        pass

    # Train/evaluate sequentially
    for b in backends:
        print(f"\n=== Training backend: {b} ===")
        ckpt_path = os.path.join("checkpoints", f"{b}_last.pt")

        model_name = None
        if b == "segformer":
            model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        elif b == "mask2former":
            model_name = "facebook/mask2former-swin-small-ade-semantic"

        _ = train_with_val(
            root=root,
            image_subroots=image_subroots,
            split_json_dir="splits",
            split_name=split_name,
            val_ratio=0.2,
            batch_size=4,
            epochs=20,
            lr=5e-5,
            backend=b,
            model_name=(model_name or "unused"),
            ckpt_path=ckpt_path,
            resume=True,
            retest_best=True,
        )

    # Build val loader on the same split for comparison plots
    all_index = load_split(os.path.join("splits", f"{split_name}.json"))
    train_index, val_index = train_val_split(all_index, val_ratio=0.2, seed=42)
    _, val_loader = make_loaders_from_index(train_index, val_index, batch_size=2, size=(512, 512))
    num_classes = count_classes(train_index)

    # Backends to compare (trained) + eval-only (e.g., SAM variants)
    comp_backends = backends + eval_only_backends

    # Overlaid curves and qualitative comparison grid (all models side-by-side)
    print("Generating overlays and qualitative grid...", flush=True)
    comp_dir = overlay_metric_curves(split_name, backends, root="experiments/results")
    _ = qualitative_comparison_grid(
        split_name,
        comp_backends,
        root="experiments/results",
        val_loader=val_loader,
        num_classes=num_classes,
        device=device
    )

    # Ensure each backend has its own qualitative grid (generate if missing)
    ensure_per_model_qualitative(
        split_name, comp_backends, root="experiments/results",
        val_loader=val_loader, num_classes=num_classes, device=device, samples=6
    )

    # Extra comparisons: performance bars, robustness heatmap, temporal consistency
    print("Generating extra comparisons...", flush=True)
    perf_m_path, perf_d_path = performance_comparison_bar(
        split_name, backends, root="experiments/results"
    )
    rb_path = robustness_heatmap(
        split_name, backends, root="experiments/results",
        val_loader=val_loader, num_classes=num_classes, device=device, samples=64
    )
    tc_path = temporal_consistency_plot(
        split_name, backends, root="experiments/results",
        val_loader=val_loader, num_classes=num_classes, device=device, samples=16
    )
    print("Extra comparisons done.", flush=True)

    # Write the comparison report (includes links to all generated figures)
    comp_md = write_comparison_report(split_name, comp_backends, root="experiments/results")
    print(f"Comparison report written to {comp_md}")


# Additionally, save a quick SAM per-model qualitative grid & one-line metrics CSV for table completeness
    if eval_only_backends:
        for sam_b in eval_only_backends:
            try:
                results_dir = os.path.join("experiments", "results", split_name, sam_b)
                _ensure_dir(results_dir)

                sam_model = load_model_for_eval(sam_b, num_classes, device, ckpt_path="")
                with torch.no_grad():
                    sam_mode = "oracle" if sam_b == "sam_oracle" else "prompt_free"
                    eval_res = evaluate_model(
                        sam_model,
                        val_loader,
                        num_classes=num_classes,
                        device=device,
                        model_type="sam",
                        sam_mode=sam_mode,
                        use_tqdm=True,
                    )

                # write minimal CSV so it shows up in comparison table (single row)
                csv_path = os.path.join(results_dir, "training_metrics.csv")
                with open(csv_path, "w") as f:
                    f.write("epoch,train_loss,val_mIoU,val_Dice\n")
                    f.write(f"1,0,{eval_res.mean_iou:.6f},{eval_res.mean_dice:.6f}\n")

                # qualitative grid for SAM alone
                save_qualitative_samples(
                    sam_model,
                    val_loader,
                    device,
                    os.path.join(results_dir, "qualitative_grid.png"),
                    num_samples=6,
                    model_type="sam",
                    sam_mode=sam_mode,
                )

                # small report
                write_markdown_report(results_dir, f"{split_name} · {sam_b}", num_classes, eval_res)

            except Exception as e:
                print(f"[WARN] SAM eval failed for {sam_b}: {e}")

