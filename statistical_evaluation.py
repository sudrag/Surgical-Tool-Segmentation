#!/usr/bin/env python3
"""
Standalone evaluation + statistical testing + plotting for segmentation checkpoints.

Key updates (v2)
----------------
- Robust mask matching across arbitrarily nested folders:
  * Builds an index over the provided --masks root and matches by normalized filename stem
    (handles suffixes like _mask, -mask, _label, -label, _seg, -seg, _gt, -gt, _ann, -ann).
  * If multiple masks share a normalized stem, choose the one with the most similar parent-path tokens.
- New flags:
  * --dry-run : don't run models; only build image→mask mapping, save a report, and exit.
  * --limit   : evaluate only the first N matched images (useful for quick checks).
- Fixed SegFormer checkpoint loader (removed invalid kwarg).

Outputs
-------
All CSVs and plots are saved under --output-dir (default: statistical_evaluation_outputs):
- *_per_image_metrics.csv
- per_image_metrics_wide.csv
- statistical_tests.csv
- boxplot_miou.png, boxplot_dice.png
- scatter_*_miou.png, scatter_*_dice.png
- mapping_report.csv (in --dry-run or always saved for transparency)

Usage (single pass over all subfolders)
--------------------------------------
python evaluate_significance.py \
  --images path/to/miccai2022_sisvse_dataset/images \
  --masks  path/to/miccai2022_sisvse_dataset/semantic_masks \
  --num-classes 3 \
  --ignore-bg \
  --deeplab     checkpoints/deeplab_best.pt \
  --segformer   checkpoints/segformer_best.pt
"""

import argparse
import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import contextlib

from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from tqdm import tqdm
import gc

# Bring in the framework adapters/checkpoint loader so evaluation matches training exactly.
from surgical_segmentation_framework import load_model_for_eval


# -------------------------------------------------------------------------------------------------
# Utility: image & mask loading
# -------------------------------------------------------------------------------------------------
def _load_resized_image(path: Path, size: Tuple[int, int]) -> Tuple[Image.Image, torch.Tensor]:
    """Returns (PIL_rgb_resized, tensor[1,3,H,W])"""
    img = Image.open(path).convert("RGB")
    if size is not None:
        # size is (H, W); PIL expects (W, H)
        img = img.resize((size[1], size[0]), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return img, tensor


def _load_resized_mask(path: Path, size: Tuple[int, int]) -> np.ndarray:
    mask = Image.open(path)
    if mask.mode not in ("I", "L"):
        mask = mask.convert("L")
    if size is not None:
        mask = mask.resize((size[1], size[0]), resample=Image.NEAREST)
    return np.asarray(mask, dtype=np.int64)


SISVSE_TOOL_CLASSES = set(range(1, 24))
SISVSE_ORGAN_CLASSES = set(range(24, 32))


def collapse_to_tool_organ(arr: np.ndarray) -> np.ndarray:
    collapsed = np.zeros_like(arr, dtype=np.int64)
    tool_mask = np.isin(arr, list(SISVSE_TOOL_CLASSES))
    organ_mask = np.isin(arr, list(SISVSE_ORGAN_CLASSES))
    collapsed[tool_mask] = 1
    collapsed[organ_mask] = 2
    return collapsed

# -------------------------------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------------------------------
def compute_iou_and_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool) -> Tuple[float, float]:
    classes = list(range(num_classes))
    if ignore_bg and 0 in classes:
        classes.remove(0)

    ious, dices = [], []
    for cls in classes:
        p = (pred == cls)
        g = (gt == cls)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        denom = (p.sum() + g.sum())

        # ✅ Safer fallback: if both masks are empty, skip the class
        if union == 0 and denom == 0:
            continue

        iou = inter / (union + 1e-8) if union > 0 else 0.0
        dice = (2 * inter) / (denom + 1e-8) if denom > 0 else 0.0
        ious.append(iou)
        dices.append(dice)

    miou = float(np.mean(ious)) if ious else 0.0
    mdice = float(np.mean(dices)) if dices else 0.0
    return miou, mdice



# -------------------------------------------------------------------------------------------------
# Robust mask matching across arbitrary folders
# -------------------------------------------------------------------------------------------------
_SUFFIXES = ["_mask", "-mask", "_label", "-label", "_seg", "-seg", "_gt", "-gt", "_ann", "-ann"]

def normalize_stem(stem: str) -> str:
    s = stem.lower()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def path_tokens(p: Path, depth: int = 5) -> set:
    # take last `depth` components, lowercase, alnum tokens
    parts = [q.lower() for q in p.parts[-depth:]]
    toks = set()
    for q in parts:
        for t in q.replace("-", "_").split("_"):
            if t:
                toks.add(t)
    return toks

def build_mask_index(masks_root: Path) -> Dict[str, List[Path]]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    index: Dict[str, List[Path]] = {}
    for m in masks_root.rglob("*"):
        if m.is_file() and m.suffix.lower() in exts:
            key = normalize_stem(m.stem)
            index.setdefault(key, []).append(m)
    return index

def match_mask_for_image(img_path: Path, mask_index: Dict[str, List[Path]]) -> Optional[Path]:
    key = normalize_stem(img_path.stem)
    cands = mask_index.get(key, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Disambiguate by comparing parent path tokens
    itoks = path_tokens(img_path.parent)
    best, best_score = None, -1.0
    for c in cands:
        ctoks = path_tokens(c.parent)
        # Jaccard similarity
        inter = len(itoks & ctoks)
        union = len(itoks | ctoks) if (itoks | ctoks) else 1
        score = inter / union
        if score > best_score:
            best, best_score = c, score
    return best

# -------------------------------------------------------------------------------------------------
# Evaluation Loop (Final Version: Background Ignored)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Final Evaluation Loop — robust, background-ignored, resolution-safe
# -------------------------------------------------------------------------------------------------
def infer_and_score(
    backend: str,
    model: nn.Module,
    device: torch.device,
    pairs: List[Tuple[Path, Path]],
    input_size: Tuple[int, int],
    num_classes: Optional[int],
    ignore_bg: bool = True,
    start: int = 0,
    limit: Optional[int] = None,
    flush_every: int = 200,
    out_csv: Optional[Path] = None,
    use_amp: bool = False,
    collapse_tool_organ: bool = False,
) -> pd.DataFrame:
    import torch.nn.functional as F

    model.eval()

    end = len(pairs) if limit is None else min(len(pairs), start + limit)
    work = pairs[start:end]
    records = []
    all_records: List[dict] = []
    processed = 0

    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else contextlib.nullcontext

    def compute_per_class_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True):
        classes = list(range(num_classes))
        if ignore_bg and 0 in classes:
            classes.remove(0)
        iou_map, dice_map = {}, {}
        for c in classes:
            p = pred == c
            g = gt == c
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            total = p.sum() + g.sum()
            if union == 0 and total == 0:
                continue
            iou_map[c] = float(inter / (union + 1e-8) if union > 0 else 0.0)
            dice_map[c] = float((2 * inter) / (total + 1e-8) if total > 0 else 0.0)
        return iou_map, dice_map

    pbar = tqdm(work, desc=f"Evaluating {backend}", unit="img")
    for img_path, mask_path in pbar:
        try:
            img_pil, img_tensor = _load_resized_image(img_path, input_size)
            gt = _load_resized_mask(mask_path, input_size)

            with torch.inference_mode():
                if backend == "deeplab":
                    x = img_tensor.to(device, non_blocking=True)
                    with amp_ctx():
                        logits = model(x)
                elif backend in ("segformer", "mask2former"):
                    logits = model([img_pil])
                else:
                    raise ValueError(f"Unknown backend {backend}")

            if isinstance(logits, dict):
                logits = logits.get("out", logits.get("logits"))

            if logits is None:
                raise RuntimeError("Model forward did not return logits tensor.")

            if not torch.is_tensor(logits):
                logits = torch.as_tensor(logits)

            logits = logits.to(device)
            if logits.shape[-2:] != gt.shape:
                logits = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)

            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            metric_pred = collapse_to_tool_organ(pred) if collapse_tool_organ else pred
            metric_gt = collapse_to_tool_organ(gt) if collapse_tool_organ else gt

            effective_classes = 3 if collapse_tool_organ else (num_classes or logits.shape[1])

            miou, mdice = compute_iou_and_dice(metric_pred, metric_gt, num_classes=effective_classes, ignore_bg=ignore_bg)
            iou_dict, dice_dict = compute_per_class_metrics(metric_pred, metric_gt, num_classes=effective_classes, ignore_bg=ignore_bg)

            per_class_cols = {f"IoU_c{c}": v for c, v in iou_dict.items()}
            per_class_cols.update({f"Dice_c{c}": v for c, v in dice_dict.items()})

            records.append(
                {
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "mIoU": miou,
                    "Dice": mdice,
                    **per_class_cols,
                }
            )
            processed += 1

            if processed % flush_every == 0 and out_csv is not None and records:
                df_partial = pd.DataFrame(records)
                if out_csv.exists():
                    df_partial.to_csv(out_csv, mode="a", header=False, index=False)
                else:
                    df_partial.to_csv(out_csv, index=False)
                all_records.extend(records)
                records.clear()
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        except Exception as e:
            print(f"[WARN] Failed on {img_path.name}: {e}")
            records.append(
                {
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "mIoU": float("nan"),
                    "Dice": float("nan"),
                }
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


    # -------------------- Final flush --------------------
    if records:
        if out_csv is not None:
            df_tail = pd.DataFrame(records)
            if out_csv.exists():
                df_tail.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                df_tail.to_csv(out_csv, index=False)
        all_records.extend(records)
        records.clear()

    df = pd.DataFrame(all_records if all_records else records)

    return df


def _log_result_preview(label: str, df: pd.DataFrame, preview: int = 3) -> None:
    if df is None or df.empty:
        print(f"[INFO] {label}: no records to report yet.")
        return

    miou_vals = df["mIoU"].to_numpy(dtype=float, copy=True) if "mIoU" in df.columns else np.array([], dtype=float)
    dice_vals = df["Dice"].to_numpy(dtype=float, copy=True) if "Dice" in df.columns else np.array([], dtype=float)

    mean_miou = float(np.nanmean(miou_vals)) if miou_vals.size else float("nan")
    mean_dice = float(np.nanmean(dice_vals)) if dice_vals.size else float("nan")

    print(
        f"[RESULT] {label}: samples={len(df)}, mean mIoU={mean_miou:.4f}, mean Dice={mean_dice:.4f}, "
        f"valid mIoU={np.isfinite(miou_vals).sum()}, valid Dice={np.isfinite(dice_vals).sum()}"
    )





# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------
def save_boxplot(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    groups = sorted(df["Model"].unique())
    data = [df[df["Model"] == g][metric].values for g in groups]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=groups, showmeans=True)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_scatter(xvals: np.ndarray, yvals: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(xvals, yvals, s=12, alpha=0.7)
    lo = min(np.min(xvals), np.min(yvals))
    hi = max(np.max(xvals), np.max(yvals))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Stats
# -------------------------------------------------------------------------------------------------
def run_stats(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> Dict[str, float]:
    from scipy.stats import ttest_rel, wilcoxon
    mask = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[mask], b[mask]
    if len(a2) < 3:
        return {"Comparison": f"{label_a} vs {label_b}", "t-test p": float("nan"), "Wilcoxon p": float("nan")}
    t_stat, t_p = ttest_rel(a2, b2)
    try:
        w_stat, w_p = wilcoxon(a2, b2)
    except Exception:
        w_p = float("nan")
    return {"Comparison": f"{label_a} vs {label_b}", "t-test p": float(t_p), "Wilcoxon p": float(w_p)}

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone statistical evaluation for segmentation checkpoints.")
    parser.add_argument("--images", type=str, required=True, help="Path to dataset images root (will recurse)")
    parser.add_argument("--masks", type=str, required=True, help="Path to dataset masks root (will recurse)")
    parser.add_argument("--input-width", type=int, default=512, help="Input width for inference")
    parser.add_argument("--input-height", type=int, default=512, help="Input height for inference")
    parser.add_argument("--num-classes", type=int, default=32, help="Number of semantic classes for the model head (incl. background=0)")
    parser.add_argument("--ignore-bg", action="store_true", help="Ignore background (class 0) when averaging metrics")
    # === NEW FLAGS FOR CHUNKED / SAFE EVAL ===
    parser.add_argument("--start", type=int, default=0, help="Start index into matched pairs")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N pairs from start")
    parser.add_argument("--flush-every", type=int, default=200, help="Write partial CSV every N samples")
    parser.add_argument("--amp", action="store_true", help="Use autocast (float16/bf16) for inference")



    parser.add_argument("--deeplab", type=str, default=None, help="Path to DeepLab checkpoint")
    parser.add_argument("--mask2former", type=str, default=None, help="Path to Mask2Former checkpoint")
    parser.add_argument("--segformer", type=str, default=None, help="Path to SegFormer checkpoint")
    parser.add_argument("--segformer-model-name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512", help="Base HF model name used when constructing SegFormer adapters")
    parser.add_argument("--mask2former-model-name", type=str, default="facebook/mask2former-swin-small-ade-semantic", help="Base HF model name used when constructing Mask2Former adapters")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--output-dir", type=str, default="statistical_evaluation_outputs", help="Output folder")
    parser.add_argument("--dry-run", action="store_true", help="Only build image→mask mapping report and exit")
    parser.add_argument("--collapse-tool-organ", action="store_true", help="Collapse SISVSE labels into background/tools/organs before scoring")
    # parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N matched images (for quick checks)")
    args = parser.parse_args()

    device = torch.device(args.device)
    input_size = (args.input_height, args.input_width)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images_root = Path(args.images)
    if not images_root.exists():
        raise FileNotFoundError(f"Images path not found: {images_root}")
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = sorted([p for p in images_root.rglob("*") if p.suffix.lower() in img_exts])

    if not image_paths:
        raise RuntimeError(f"No images found under {images_root}")

    # Build mask index
    masks_root = Path(args.masks)
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks path not found: {masks_root}")
    mask_index = build_mask_index(masks_root)

    # Pair images to masks
    pairs: List[Tuple[Path, Path]] = []
    unmatched = []
    for img in image_paths:
        m = match_mask_for_image(img, mask_index)
        if m is None:
            unmatched.append(img)
        else:
            pairs.append((img, m))

    # Save mapping report
    map_rows = [{"image": str(i), "mask": (str(m) if m else "")} for i, m in pairs]
    for u in unmatched:
        map_rows.append({"image": str(u), "mask": ""})
    pd.DataFrame(map_rows).to_csv(out_dir / "mapping_report.csv", index=False)

    if not pairs:
        print("[ERROR] No image→mask pairs were matched. Check --masks root and naming.")
        sys.exit(1)

    print(f"[INFO] Found {len(pairs)} matched pairs; {len(unmatched)} images unmatched.")

    if args.dry_run:
        print(f"[DRY-RUN] Mapping saved to {out_dir / 'mapping_report.csv'}. Exiting before model eval.")
        sys.exit(0)

    results_per_model: Dict[str, pd.DataFrame] = {}

    # ------------------- DeepLab -------------------
    if args.deeplab:
        try:
            deeplab_model = load_model_for_eval(
                backend="deeplab",
                num_classes=args.num_classes,
                device=device,
                ckpt_path=args.deeplab,
            )

            out_csv_path = out_dir / "deeplab_per_image_metrics.csv"
            df_dl = infer_and_score(
                backend="deeplab",
                model=deeplab_model,
                device=device,
                pairs=pairs,
                input_size=input_size,
                num_classes=args.num_classes,
                ignore_bg=args.ignore_bg,
                start=args.start,
                limit=args.limit,
                flush_every=args.flush_every,
                out_csv=out_csv_path,
                use_amp=args.amp,
                collapse_tool_organ=args.collapse_tool_organ,
            )
            results_per_model["DeepLab"] = df_dl
            print(f"[OK] DeepLab evaluated on {len(df_dl)} images (start={args.start}).")
            _log_result_preview("DeepLab", df_dl)
        except Exception as e:
            print(f"[WARN] Skipping DeepLab: {e}")

    # ------------------- SegFormer -------------------
    if args.segformer:
        try:
            segformer_model = load_model_for_eval(
                backend="segformer",
                num_classes=args.num_classes,
                device=device,
                ckpt_path=args.segformer,
                model_name=args.segformer_model_name,
            )

            out_csv_path = out_dir / "segformer_per_image_metrics.csv"
            df_sf = infer_and_score(
                backend="segformer",
                model=segformer_model,
                device=device,
                pairs=pairs,
                input_size=input_size,
                num_classes=args.num_classes,
                ignore_bg=args.ignore_bg,
                start=args.start,
                limit=args.limit,
                flush_every=args.flush_every,
                out_csv=out_csv_path,
                use_amp=args.amp,
                collapse_tool_organ=args.collapse_tool_organ,
            )
            results_per_model["SegFormer"] = df_sf
            print(f"[OK] SegFormer evaluated on {len(df_sf)} images (start={args.start}).")
            _log_result_preview("SegFormer", df_sf)
        except Exception as e:
            print(f"[WARN] Skipping SegFormer: {e}")

    # ------------------- Mask2Former -------------------
    if args.mask2former:
        try:
            mask2former_model = load_model_for_eval(
                backend="mask2former",
                num_classes=args.num_classes,
                device=device,
                ckpt_path=args.mask2former,
                model_name=args.mask2former_model_name,
            )

            out_csv_path = out_dir / "mask2former_per_image_metrics.csv"
            df_m2f = infer_and_score(
                backend="mask2former",
                model=mask2former_model,
                device=device,
                pairs=pairs,
                input_size=input_size,
                num_classes=args.num_classes,
                ignore_bg=args.ignore_bg,
                start=args.start,
                limit=args.limit,
                flush_every=args.flush_every,
                out_csv=out_csv_path,
                use_amp=args.amp,
                collapse_tool_organ=args.collapse_tool_organ,
            )
            results_per_model["Mask2Former"] = df_m2f
            print(f"[OK] Mask2Former evaluated on {len(df_m2f)} images (start={args.start}).")
            _log_result_preview("Mask2Former", df_m2f)
        except Exception as e:
            print(f"[WARN] Skipping Mask2Former: {e}")


    # Combine for stats
    if len(results_per_model) < 2:
        print("[ERROR] Fewer than 2 models produced results; cannot run pairwise statistics.")
        sys.exit(0)

    # Align on common image set across all available models
    common = None
    for name, df in results_per_model.items():
        df = df.copy()
        df["Model"] = name
        results_per_model[name] = df
        common = df[["image"]] if common is None else common.merge(df[["image"]], on="image", how="inner")
    if common is None or common.empty:
        print("[ERROR] No common images across models. Exiting.")
        sys.exit(0)

    # Build a wide table for stats
    wide = common.copy()
    for name, df in results_per_model.items():
        cols = ["image"]
        if "mIoU" in df.columns:
            cols.append("mIoU")
        if "Dice" in df.columns:
            cols.append("Dice")
        renamed = df[cols].rename(
            columns={
                "mIoU": f"mIoU_{name}",
                "Dice": f"Dice_{name}",
            }
        )
        wide = wide.merge(renamed, on="image", how="left")

    # Save a combined CSV for debugging
    wide.to_csv(out_dir / "per_image_metrics_wide.csv", index=False)

    # Pairwise stats
    models = list(results_per_model.keys())
    stats_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_miou = wide[f"mIoU_{models[i]}"].to_numpy(dtype=float)
            b_miou = wide[f"mIoU_{models[j]}"].to_numpy(dtype=float)
            a_dice = wide[f"Dice_{models[i]}"].to_numpy(dtype=float)
            b_dice = wide[f"Dice_{models[j]}"].to_numpy(dtype=float)

            s_miou = run_stats(a_miou, b_miou, models[i], models[j])
            s_miou["Metric"] = "mIoU"
            s_dice = run_stats(a_dice, b_dice, models[i], models[j])
            s_dice["Metric"] = "Dice"
            stats_rows.extend([s_miou, s_dice])

    stats_df = pd.DataFrame(stats_rows, columns=["Metric", "Comparison", "t-test p", "Wilcoxon p"])
    stats_df.to_csv(out_dir / "statistical_tests.csv", index=False)
    print(stats_df)

    # Plots (boxplots)
    long_df = pd.concat([df for df in results_per_model.values()], ignore_index=True)
    save_boxplot(long_df, "mIoU", out_dir / "boxplot_miou.png", "Per-Image mIoU Distribution")
    save_boxplot(long_df, "Dice", out_dir / "boxplot_dice.png", "Per-Image Dice Distribution")

    # Pairwise scatter plots
    def maybe_scatter(ref: str, oth: str, metric: str, fname: str):
        col_x = f"{metric}_{ref}"
        col_y = f"{metric}_{oth}"
        if col_x in wide.columns and col_y in wide.columns:
            x = wide[col_x].to_numpy(dtype=float)
            y = wide[col_y].to_numpy(dtype=float)
            save_scatter(x, y, f"{ref} {metric}", f"{oth} {metric}", f"{ref} vs {oth} ({metric})", out_dir / fname)

    if "SegFormer" in models and "DeepLab" in models:
        maybe_scatter("SegFormer", "DeepLab", "mIoU", "scatter_segformer_vs_deeplab_miou.png")
        maybe_scatter("SegFormer", "DeepLab", "Dice",  "scatter_segformer_vs_deeplab_dice.png")
    if "SegFormer" in models and "Mask2Former" in models:
        maybe_scatter("SegFormer", "Mask2Former", "mIoU", "scatter_segformer_vs_mask2former_miou.png")
        maybe_scatter("SegFormer", "Mask2Former", "Dice",  "scatter_segformer_vs_mask2former_dice.png")
    if "Mask2Former" in models and "DeepLab" in models:
        maybe_scatter("Mask2Former", "DeepLab", "mIoU", "scatter_mask2former_vs_deeplab_miou.png")
        maybe_scatter("Mask2Former", "DeepLab", "Dice",  "scatter_mask2former_vs_deeplab_dice.png")

    print(f"[DONE] All outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
