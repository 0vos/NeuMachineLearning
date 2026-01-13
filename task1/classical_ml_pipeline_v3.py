#!/usr/bin/env python3
"""
Advanced Classical ML pipeline for plant seedling classification (v3).
Includes:
- Background Segmentation (HSV Thresholding)
- Bag of Visual Words (SIFT + KMeans)
- Color Moments & Histograms (Masked)
- Haralick Texture Features (Masked)
- Ensemble Classification (RF + SVM + XGB)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import skeletonize
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Default relative paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "neu-plant-seedling-classification-2025" / "dataset-for-task1" / "dataset-for-task1"
DEFAULT_TRAIN_DIR = DEFAULT_DATASET_ROOT / "train"
def run_grouped_cv(
    base_pipeline: Pipeline,
    thin_refiner_proto: Pipeline,
    thin_detector_proto: Pipeline,
    X: np.ndarray,
    y_aug: np.ndarray,
    group_ids: np.ndarray,
    is_original_mask: np.ndarray,
    original_labels: np.ndarray,
    original_image_ids: np.ndarray,
    class_names: Sequence[str],
    n_splits: int,
    thin_threshold: float,
) -> np.ndarray:
    """Run StratifiedKFold on original images while training with augmented samples."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: List[float] = []
    aggregate_true: List[str] = []
    aggregate_pred: List[str] = []
    aggregate_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    misclassified_records: List[str] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(original_labels)), original_labels), start=1):
        train_mask = np.isin(group_ids, train_idx)
        val_rows = [np.where((group_ids == idx) & is_original_mask)[0][0] for idx in val_idx]

        X_train_fold = X[train_mask]
        y_train_fold = y_aug[train_mask]
        X_val_fold = X[val_rows]
        y_val_fold = original_labels[val_idx]

        model = clone(base_pipeline)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)

        # train thin refiner on augmented training samples of thin classes
        refiner_model = None
        thin_mask_train = np.isin(y_train_fold, list(THIN_CLASSES))
        if thin_mask_train.sum() >= 2 and len(np.unique(y_train_fold[thin_mask_train])) == len(THIN_CLASSES):
            refiner_model = clone(thin_refiner_proto)
            refiner_model.fit(X_train_fold[thin_mask_train], y_train_fold[thin_mask_train])

        thin_detector_model = None
        thin_indicator_train = thin_mask_train.astype(int)
        if thin_indicator_train.min() != thin_indicator_train.max():
            thin_detector_model = clone(thin_detector_proto)
            thin_detector_model.fit(X_train_fold, thin_indicator_train)

        if refiner_model is not None:
            thin_pred_mask = np.isin(preds, list(THIN_CLASSES))
            if thin_pred_mask.any():
                refined = refiner_model.predict(X_val_fold[thin_pred_mask])
                preds[thin_pred_mask] = refined

        if refiner_model is not None and thin_detector_model is not None:
            thin_probs = thin_detector_model.predict_proba(X_val_fold)[:, 1]
            escalate_mask = (~np.isin(preds, list(THIN_CLASSES))) & (thin_probs >= thin_threshold)
            if escalate_mask.any():
                refined = refiner_model.predict(X_val_fold[escalate_mask])
                preds[escalate_mask] = refined

        score = f1_score(y_val_fold, preds, average="macro")
        scores.append(score)
        aggregate_true.extend(y_val_fold)
        aggregate_pred.extend(preds)

        print(f"Fold {fold} Macro F1: {score:.4f}")
        print(classification_report(
            y_val_fold,
            preds,
            labels=class_names,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ))

        cm = confusion_matrix(y_val_fold, preds, labels=class_names)
        aggregate_cm += cm

        for local_idx, true_label, pred_label in zip(val_idx, y_val_fold, preds):
            if true_label != pred_label and true_label in {"Black-grass", "Loose Silky-bent"}:
                misclassified_records.append(
                    f"Fold {fold}: {original_image_ids[local_idx]} true={true_label} pred={pred_label}"
                )

    print("Aggregated Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(aggregate_cm, index=class_names, columns=class_names))
    print("Aggregated Classification Report:")
    print(classification_report(
        aggregate_true,
        aggregate_pred,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    scores_array = np.array(scores)
    print(f"Mean Macro F1: {scores_array.mean():.4f} ± {scores_array.std():.4f}")

    if misclassified_records:
        print("Misclassified thin-leaf samples (Black-grass / Loose Silky-bent):")
        for line in misclassified_records:
            print("  " + line)

    return scores_array

DEFAULT_TEST_DIR = DEFAULT_DATASET_ROOT / "test"
DEFAULT_SUBMISSION_PATH = BASE_DIR / "submission-for-task1.csv"
THIN_CLASSES = {"Black-grass", "Loose Silky-bent"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical ML classifier (BoVW + Color + Texture) and generate predictions."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Path to training images")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Path to test images")
    parser.add_argument(
        "--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH, help="Where to write submission CSV"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize images to this size for processing",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=100,
        help="Number of visual words (K-Means clusters) for SIFT BoVW",
    )
    parser.add_argument(
        "--orb-vocab-size",
        type=int,
        default=64,
        help="Number of visual words for ORB descriptors (0 to disable)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Augment training data",
    )
    parser.add_argument(
        "--thin-threshold",
        type=float,
        default=0.6,
        help="Probability threshold to trigger thin-class override",
    )
    return parser.parse_args(argv)


def list_image_paths(root: Path) -> List[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
    return paths


def load_training_metadata(train_dir: Path) -> Tuple[List[Path], List[str]]:
    image_paths: List[Path] = []
    labels: List[str] = []
    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {train_dir}")
    for class_dir in class_dirs:
        current_paths = list_image_paths(class_dir)
        image_paths.extend(current_paths)
        labels.extend([class_dir.name] * len(current_paths))
    return image_paths, labels


def load_test_metadata(test_dir: Path) -> Tuple[List[Path], List[str]]:
    paths = list_image_paths(test_dir)
    ids = [p.name for p in paths]
    return paths, ids


def create_mask_for_plant(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask where the plant is white and background is black.
    Combines HSV thresholding with excess-green detection to better retain thin leaves.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 35, 25])
    upper_green = np.array([95, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)

    # Excess green (ExG) helps capture thin, desaturated leaves
    rgb = image.astype(np.float32)
    exg = 2 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_exg = cv2.threshold(exg_norm, 140, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_hsv, mask_exg)

    # Clean small holes while preserving thin structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    return mask


def preprocess_image(path: Path, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read image, resize, and create mask."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask = create_mask_for_plant(image)
    return image, mask


def augment_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    is_thin_class: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate augmented versions with optional extra transforms for thin-leaf classes."""
    augmented = [(image, mask)]
    
    # Flips
    for code in [0, 1, -1]:
        augmented.append((cv2.flip(image, code), cv2.flip(mask, code)))
        
    # Rotations
    for code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
        augmented.append((cv2.rotate(image, code), cv2.rotate(mask, code)))

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Small-angle rotations help differentiate thin leaves without drastic orientation change
    for angle in (-15, 15):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        rotated_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        augmented.append((rotated_img, rotated_mask))

    if is_thin_class:
        # Extra orientations capture slender stems that often lean mildly in the dataset
        for angle in (-35, -25, 25, 35):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            rotated_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            augmented.append((rotated_img, rotated_mask))

        # Mild shears mimic wind-driven bending unique to thin leaves
        for shear in (-0.12, 0.12):
            shear_matrix = np.array([[1, shear, -shear * center[1]], [0, 1, 0]], dtype=np.float32)
            sheared_img = cv2.warpAffine(
                image,
                shear_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            sheared_mask = cv2.warpAffine(
                mask,
                shear_matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            augmented.append((sheared_img, sheared_mask))

        # Slight scaling encourages robustness to distance variations
        for scale in (0.9, 1.1):
            matrix = cv2.getRotationMatrix2D(center, 0, scale)
            scaled_img = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            scaled_mask = cv2.warpAffine(
                mask,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            augmented.append((scaled_img, scaled_mask))
    
    return augmented


# --- Feature Extractors ---

def get_color_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Enhanced color features from multiple color spaces.
    Extracts: HSV moments, LAB moments, saturation distribution, color ratios.
    """
    if cv2.countNonZero(mask) == 0:
        return np.zeros(70, dtype=np.float32)

    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # HSV color space - critical for plant color analysis
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[mask > 0]
    
    # LAB color space - perceptually uniform, good for color differences
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    lab_pixels = lab[mask > 0]
    
    if len(hsv_pixels) == 0:
        return np.zeros(70, dtype=np.float32)

    features = []
    
    # HSV statistics (15 features)
    for i in range(3):
        channel = hsv_pixels[:, i].astype(np.float32)
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel),
            np.min(channel),
            np.max(channel)
        ])
    
    # LAB statistics (15 features) - captures green intensity better
    for i in range(3):
        channel = lab_pixels[:, i].astype(np.float32)
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel),
            np.min(channel),
            np.max(channel)
        ])
    
    # Saturation distribution analysis (3 features) - distinguishes vibrant vs pale leaves
    s_channel = hsv_pixels[:, 1].astype(np.float32)
    low_sat = np.sum(s_channel < 50) / len(s_channel)
    mid_sat = np.sum((s_channel >= 50) & (s_channel < 150)) / len(s_channel)
    high_sat = np.sum(s_channel >= 150) / len(s_channel)
    features.extend([low_sat, mid_sat, high_sat])
    
    # Hue dominant ranges (3 features) - captures dominant color regions
    h_channel = hsv_pixels[:, 0].astype(np.float32)
    hue_ranges = [
        np.sum((h_channel >= 25) & (h_channel < 45)) / len(h_channel),  # yellow-green
        np.sum((h_channel >= 45) & (h_channel < 75)) / len(h_channel),  # green
        np.sum((h_channel >= 75) & (h_channel < 95)) / len(h_channel),  # blue-green
    ]
    features.extend(hue_ranges)
    
    # Color channel ratios (3 features) - relative color strength
    rgb_pixels = image[mask > 0].astype(np.float32)
    r_mean, g_mean, b_mean = rgb_pixels.mean(axis=0)
    total = r_mean + g_mean + b_mean + 1e-6
    features.extend([r_mean/total, g_mean/total, b_mean/total])
    
    # Fine-grained histograms (24 features)
    hist_h = cv2.calcHist([hsv], [0], mask, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()
    
    hist = np.concatenate([hist_h, hist_s, hist_v])
    hist = hist / (np.sum(hist) + 1e-6)
    features.extend(hist.tolist())
    
    # Excess green index (1 feature) - discriminates greenness
    exg = 2 * g_mean - r_mean - b_mean
    features.append(exg / 255.0)
    
    # Color variance spatial analysis (6 features)
    for channel_idx in range(3):
        channel_img = hsv[:, :, channel_idx]
        channel_masked = channel_img[mask > 0]
        if len(channel_masked) > 0:
            features.append(np.percentile(channel_masked, 90) - np.percentile(channel_masked, 10))
    
    return np.array(features[:70], dtype=np.float32)


def get_texture_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Enhanced multi-scale texture features.
    Combines: Haralick GLCM, multi-scale Canny edges, gradient statistics, Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(mask)
    
    if coords is None:
        return np.zeros(45, dtype=np.float32)
    
    features = []
    
    # 1. Haralick GLCM features (12 features)
    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[y:y+h, x:x+w]
    roi_quantized = (roi // 8).astype(np.uint8)
    
    try:
        glcm = graycomatrix(roi_quantized, distances=[1, 2], angles=[0, np.pi/2], levels=32, symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            val = graycoprops(glcm, prop).ravel()
            features.extend([val.mean(), val.std()])
    except ValueError:
        features.extend([0.0] * 12)
    
    # 2. Multi-scale edge density (6 features) - captures leaf edge complexity at different scales
    for threshold1, threshold2 in [(50, 150), (100, 200), (150, 250)]:
        edges = cv2.Canny(gray, threshold1, threshold2)
        edge_density = np.sum((edges > 0) & (mask > 0)) / (cv2.countNonZero(mask) + 1e-6)
        edge_strength = np.mean(edges[mask > 0]) if np.any(mask) else 0.0
        features.extend([edge_density, edge_strength / 255.0])
    
    # 3. Gradient magnitude statistics (6 features)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_pixels = gradient_mag[mask > 0]
    if len(grad_pixels) > 0:
        features.extend([
            np.mean(grad_pixels) / 255.0,
            np.std(grad_pixels) / 255.0,
            np.median(grad_pixels) / 255.0,
            np.percentile(grad_pixels, 25) / 255.0,
            np.percentile(grad_pixels, 75) / 255.0,
            np.max(grad_pixels) / 255.0
        ])
    else:
        features.extend([0.0] * 6)
    
    # 4. Laplacian variance at multiple scales (3 features) - focus/sharpness measure
    for ksize in [3, 5, 7]:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        lap_var = np.var(laplacian[mask > 0]) if np.any(mask) else 0.0
        features.append(lap_var / 10000.0)
    
    # 5. Local intensity variation (3 features)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    diff_pixels = diff[mask > 0]
    if len(diff_pixels) > 0:
        features.extend([
            np.mean(diff_pixels) / 255.0,
            np.std(diff_pixels) / 255.0,
            np.max(diff_pixels) / 255.0
        ])
    else:
        features.extend([0.0] * 3)
    
    # 6. Entropy (1 feature) - texture complexity
    roi_pixels = roi.flatten()
    if len(roi_pixels) > 0:
        hist, _ = np.histogram(roi_pixels, bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        features.append(entropy / 5.0)
    else:
        features.append(0.0)
    
    # 7. Directional gradients (8 features) - edge orientation strength
    for angle in [0, 45, 90, 135]:
        theta = np.radians(angle)
        kx = np.cos(theta)
        ky = np.sin(theta)
        grad_dir = sobelx * kx + sobely * ky
        dir_pixels = grad_dir[mask > 0]
        if len(dir_pixels) > 0:
            features.extend([
                np.mean(np.abs(dir_pixels)) / 255.0,
                np.std(dir_pixels) / 255.0
            ])
        else:
            features.extend([0.0, 0.0])
    
    # 8. High frequency content (6 features) - fine texture details
    gray_float = gray.astype(np.float32)
    # Apply high-pass filters
    kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_freq = cv2.filter2D(gray_float, -1, kernel_hp)
    hf_pixels = high_freq[mask > 0]
    if len(hf_pixels) > 0:
        features.extend([
            np.mean(np.abs(hf_pixels)) / 255.0,
            np.std(hf_pixels) / 255.0,
            np.percentile(np.abs(hf_pixels), 90) / 255.0
        ])
    else:
        features.extend([0.0] * 3)
    
    # Different kernel
    kernel_hp2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    high_freq2 = cv2.filter2D(gray_float, -1, kernel_hp2)
    hf2_pixels = high_freq2[mask > 0]
    if len(hf2_pixels) > 0:
        features.extend([
            np.mean(np.abs(hf2_pixels)) / 255.0,
            np.std(hf2_pixels) / 255.0,
            np.max(np.abs(hf2_pixels)) / 255.0
        ])
    else:
        features.extend([0.0] * 3)
    
    return np.array(features[:45], dtype=np.float32)


def get_shape_features(mask: np.ndarray) -> np.ndarray:
    """Enhanced shape features with convex hull, multi-scale morphology, and ellipse fitting."""
    total_pixels = mask.size
    area = float(cv2.countNonZero(mask))
    area_ratio = area / total_pixels if total_pixels else 0.0

    coords = cv2.findNonZero(mask)
    if coords is None or area == 0:
        return np.zeros(45, dtype=np.float32)

    features = []
    
    # 1. Basic geometric features
    x, y, w, h = cv2.boundingRect(coords)
    bbox_area = float(w * h) if w and h else 1.0
    extent = area / bbox_area if bbox_area else 0.0
    aspect_ratio = (w / h) if h else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) / max(mask.shape)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    main_contour = None
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    
    features.extend([
        area_ratio, extent, aspect_ratio, equiv_diameter, circularity,
        perimeter / (mask.shape[0] + mask.shape[1]),
        w / max(mask.shape), h / max(mask.shape)
    ])
    
    # 2. Convex hull analysis
    if contours and main_contour is not None:
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = hull_perimeter / (perimeter + 1e-6)
        
        hull_indices = cv2.convexHull(main_contour, returnPoints=False)
        if len(hull_indices) > 3 and len(main_contour) > 3:
            try:
                defects = cv2.convexityDefects(main_contour, hull_indices)
                if defects is not None:
                    defect_depths = defects[:, 0, 3] / 256.0
                    avg_defect = np.mean(defect_depths) / max(mask.shape)
                    max_defect = np.max(defect_depths) / max(mask.shape)
                    num_defects = len(defects) / 100.0
                else:
                    avg_defect = max_defect = num_defects = 0.0
            except:
                avg_defect = max_defect = num_defects = 0.0
        else:
            avg_defect = max_defect = num_defects = 0.0
            
        features.extend([solidity, convexity, avg_defect, max_defect, num_defects, 
                        hull_area / total_pixels, (hull_area - area) / (area + 1e-6)])
    else:
        features.extend([0.0] * 7)
    
    # 3. Hu moments
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    features.extend(hu.tolist())
    
    # 4. Skeleton analysis
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    skeleton_length = float(skeleton.sum())
    length_ratio = skeleton_length / (area + 1e-6)
    norm_length = skeleton_length / (mask.shape[0] + mask.shape[1])

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = convolve(skeleton, kernel, mode="constant", cval=0)
    endpoints = float(np.sum((skeleton == 1) & (neighbor_count == 2)))
    junctions = float(np.sum((skeleton == 1) & (neighbor_count >= 4)))
    branches = float(np.sum((skeleton == 1) & (neighbor_count == 3)))
    endpoint_density = endpoints / (skeleton_length + 1e-6)
    junction_density = junctions / (skeleton_length + 1e-6)
    
    features.extend([length_ratio, norm_length, endpoint_density, junction_density, 
                     endpoints / 100.0, junctions / 50.0, branches / 50.0])
    
    # 5. Multi-scale morphological features
    for kernel_size in [3, 7, 15]:
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
        open_ratio = cv2.countNonZero(opened) / (area + 1e-6)
        close_diff = (cv2.countNonZero(closed) - area) / total_pixels
        features.extend([open_ratio, close_diff])
    
    # 6. Eccentricity and ellipse fitting
    if moments['mu20'] + moments['mu02'] > 0:
        eccentricity = np.sqrt(1 - (moments['mu20'] - moments['mu02'])**2 / (moments['mu20'] + moments['mu02'])**2)
    else:
        eccentricity = 0.0
    
    if main_contour is not None and len(main_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(main_contour)
            ellipse_ratio = min(ellipse[1]) / (max(ellipse[1]) + 1e-6)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4.0
            ellipse_fill = area / (ellipse_area + 1e-6)
        except:
            ellipse_ratio = ellipse_fill = 0.0
    else:
        ellipse_ratio = ellipse_fill = 0.0
    
    features.extend([eccentricity, ellipse_ratio, ellipse_fill])
    
    # 7. Compactness variations
    iso_quotient = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    roughness = perimeter / (2 * np.sqrt(np.pi * area) + 1e-6)
    rectangularity = area / (bbox_area + 1e-6)
    shape_factor = (perimeter ** 2) / (area + 1e-6)
    
    features.extend([iso_quotient, roughness, rectangularity, shape_factor / 100.0])
    
    # 8. Distance transform statistics
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_values = dist[mask > 0]
    if len(dist_values) > 0:
        features.extend([
            np.mean(dist_values) / max(mask.shape),
            np.std(dist_values) / max(mask.shape),
            np.max(dist_values) / max(mask.shape)
        ])
    else:
        features.extend([0.0] * 3)
    
    return np.array(features[:45], dtype=np.float32)





def get_width_features(mask: np.ndarray) -> np.ndarray:
    """Enhanced width features via distance transform with percentiles."""
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(7, dtype=np.float32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = skeletonize(binary).astype(bool)
    samples = dist[skeleton]
    if samples.size == 0:
        samples = dist[binary > 0]
    widths = samples * 2.0  # approximate diameter
    norm = max(mask.shape)
    
    if len(widths) > 0:
        return np.array([
            widths.mean() / norm,
            widths.std() / norm,
            widths.max() / norm,
            widths.min() / norm,
            np.median(widths) / norm,
            np.percentile(widths, 25) / norm,
            np.percentile(widths, 75) / norm,
        ], dtype=np.float32)
    else:
        return np.zeros(7, dtype=np.float32)


def get_radial_features(mask: np.ndarray, num_bins: int = 16) -> np.ndarray:
    """Radial signature of contour distances from centroid across angles."""
    coords = cv2.findNonZero(mask)
    if coords is None or cv2.countNonZero(mask) == 0:
        return np.zeros(num_bins + 4, dtype=np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(num_bins + 4, dtype=np.float32)

    main_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(main_contour)
    if moments["m00"] == 0:
        return np.zeros(num_bins + 4, dtype=np.float32)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    points = main_contour[:, 0, :].astype(np.float32)
    vectors = points - np.array([[cx, cy]], dtype=np.float32)
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    distances = np.linalg.norm(vectors, axis=1)

    radial_hist = np.zeros(num_bins, dtype=np.float32)
    bin_indices = np.floor(angles / (2 * np.pi) * num_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    for idx, dist_val in zip(bin_indices, distances):
        if dist_val > radial_hist[idx]:
            radial_hist[idx] = dist_val

    norm = float(max(mask.shape)) or 1.0
    radial_hist /= norm
    stats = np.array([
        radial_hist.mean(),
        radial_hist.std(),
        radial_hist.max(),
        radial_hist.min(),
    ], dtype=np.float32)
    return np.concatenate([radial_hist, stats])


def get_hole_features(mask: np.ndarray) -> np.ndarray:
    """Capture internal hole statistics to describe complex leaf shapes."""
    area = float(cv2.countNonZero(mask))
    if area == 0:
        return np.zeros(5, dtype=np.float32)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return np.zeros(5, dtype=np.float32)

    hole_areas: List[float] = []
    circularities: List[float] = []
    for idx, node in enumerate(hierarchy[0]):
        parent = node[3]
        if parent != -1:  # child contour -> hole
            hole_area = cv2.contourArea(contours[idx])
            if hole_area <= 0:
                continue
            hole_areas.append(hole_area)
            perim = cv2.arcLength(contours[idx], True)
            circ = (4 * np.pi * hole_area) / (perim ** 2 + 1e-6)
            circularities.append(circ)

    if not hole_areas:
        return np.zeros(5, dtype=np.float32)

    hole_areas_np = np.array(hole_areas, dtype=np.float32)
    total_hole_area = hole_areas_np.sum()
    largest_hole = hole_areas_np.max()

    return np.array([
        len(hole_areas) / 10.0,
        total_hole_area / (area + 1e-6),
        largest_hole / (area + 1e-6),
        float(np.mean(circularities)) if circularities else 0.0,
        float(np.std(hole_areas_np) / (area + 1e-6)),
    ], dtype=np.float32)


def get_channel_gradient_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Color-channel gradient statistics that capture venation intensity."""
    if cv2.countNonZero(mask) == 0:
        return np.zeros(11, dtype=np.float32)

    mask_bool = mask.astype(bool)
    channel_means: List[float] = []
    feats: List[float] = []
    for ch in range(3):
        channel = image[:, :, ch].astype(np.float32)
        sobelx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        values = magnitude[mask_bool]
        if values.size == 0:
            feats.extend([0.0, 0.0, 0.0])
            channel_means.append(0.0)
        else:
            feats.extend([
                float(np.mean(values) / 255.0),
                float(np.std(values) / 255.0),
                float(np.max(values) / 255.0),
            ])
            channel_means.append(float(np.mean(values)))

    # Relative gradient strength between channels (green vs others)
    g = channel_means[1] if len(channel_means) > 1 else 0.0
    r = channel_means[0] if channel_means else 0.0
    b = channel_means[2] if len(channel_means) > 2 else 0.0
    green_red_ratio = (g - r) / (g + r + 1e-6)
    green_blue_ratio = (g - b) / (g + b + 1e-6)
    feats.extend([green_red_ratio, green_blue_ratio])
    return np.array(feats, dtype=np.float32)


def get_frequency_features(image: np.ndarray, mask: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Frequency-domain energy distribution for masked grayscale texture."""
    if cv2.countNonZero(mask) == 0:
        return np.zeros(5, dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    masked = resized.astype(np.float32) * (mask_resized > 0)

    spectrum = np.fft.fftshift(np.fft.fft2(masked))
    magnitude = np.abs(spectrum)
    energy = magnitude ** 2
    total_energy = energy.sum() + 1e-6

    rows, cols = energy.shape
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    y_grid, x_grid = np.ogrid[:rows, :cols]
    radius = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    radius_norm = radius / (radius.max() + 1e-6)

    low = energy[radius_norm < 0.3].sum() / total_energy
    mid = energy[(radius_norm >= 0.3) & (radius_norm < 0.6)].sum() / total_energy
    high = energy[radius_norm >= 0.6].sum() / total_energy

    spectral_centroid = float((radius_norm * energy).sum() / total_energy)
    flattened = (energy / total_energy).ravel()
    spectral_entropy = float(-(flattened * np.log2(flattened + 1e-9)).sum())

    return np.array([low, mid, high, spectral_centroid, spectral_entropy], dtype=np.float32)


def get_multiscale_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract features at multiple scales to capture both fine details and overall structure.
    Resizes image to different scales and computes basic statistics.
    """
    if cv2.countNonZero(mask) == 0:
        return np.zeros(15, dtype=np.float32)
    
    features = []
    
    # Original scale statistics
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masked_pixels = hsv[mask > 0]
    
    if len(masked_pixels) == 0:
        return np.zeros(15, dtype=np.float32)
    
    # Multi-scale analysis
    scales = [0.5, 1.0, 2.0]
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            img_scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA)
            mask_scaled = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            img_scaled = image
            mask_scaled = mask
        
        # Compute edge density at this scale
        gray_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_scaled, 100, 200)
        edge_density = np.sum((edges > 0) & (mask_scaled > 0)) / (cv2.countNonZero(mask_scaled) + 1e-6)
        
        # Compute color variance at this scale
        hsv_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2HSV)
        h_variance = np.var(hsv_scaled[:, :, 0][mask_scaled > 0]) if np.any(mask_scaled) else 0.0
        s_mean = np.mean(hsv_scaled[:, :, 1][mask_scaled > 0]) if np.any(mask_scaled) else 0.0
        
        features.extend([edge_density, h_variance / 1000.0, s_mean / 255.0])
    
    # Pyramid features - capture hierarchical structure
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pyr_down = cv2.pyrDown(gray)
    pyr_up = cv2.pyrUp(pyr_down, dstsize=(gray.shape[1], gray.shape[0]))
    detail = cv2.absdiff(gray, pyr_up)
    detail_pixels = detail[mask > 0]
    
    if len(detail_pixels) > 0:
        features.extend([
            np.mean(detail_pixels) / 255.0,
            np.std(detail_pixels) / 255.0,
            np.max(detail_pixels) / 255.0
        ])
    else:
        features.extend([0.0] * 3)
    
    return np.array(features[:15], dtype=np.float32)


def get_orientation_features(image: np.ndarray, mask: np.ndarray, bins: int = 12) -> np.ndarray:
    """Gradient orientation histogram restricted to plant pixels."""
    if cv2.countNonZero(mask) == 0:
        return np.zeros(bins, dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    orientation = np.arctan2(sobely, sobelx)

    mask_bool = mask.astype(bool)
    mag_values = magnitude[mask_bool]
    if mag_values.size == 0:
        return np.zeros(bins, dtype=np.float32)

    orient_values = orientation[mask_bool]
    hist, _ = np.histogram(orient_values, bins=bins, range=(-np.pi, np.pi), weights=mag_values)
    hist = hist / (mag_values.sum() + 1e-6)
    return hist.astype(np.float32)


def get_lbp_features(image: np.ndarray, mask: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    """Masked LBP histogram capturing local micro-texture."""
    if cv2.countNonZero(mask) == 0:
        return np.zeros(n_points + 2, dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    masked_lbp = lbp[mask > 0]
    if masked_lbp.size == 0:
        return np.zeros(n_points + 2, dtype=np.float32)

    hist, _ = np.histogram(masked_lbp, bins=np.arange(0, n_points + 3), density=True)
    return hist.astype(np.float32)


def get_sift_descriptors(image: np.ndarray, mask: np.ndarray, sift: cv2.SIFT) -> Optional[np.ndarray]:
    """Detect and compute SIFT descriptors in the masked region."""
    # Keypoints only in the mask
    keypoints = sift.detect(image, mask)
    if not keypoints:
        return None
    _, descriptors = sift.compute(image, keypoints)
    return descriptors


def get_orb_descriptors(image: np.ndarray, mask: np.ndarray, orb: cv2.ORB) -> Optional[np.ndarray]:
    """Detect and compute ORB descriptors limited to the mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None:
        return None
    return descriptors.astype(np.float32)


# --- Main Pipeline ---

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    
    # 1. Load Metadata
    train_paths, train_labels = load_training_metadata(args.train_dir)
    test_paths, test_ids = load_test_metadata(args.test_dir)
    
    print(f"Loaded {len(train_paths)} training images.")
    print(f"Loaded {len(test_paths)} test images.")
    
    # 2. Preprocess & Augment Training Data
    # We need to store images in memory to extract SIFT and other features efficiently
    # For 500 images * 8 augmentations = 4000 images. 256x256x3 is small enough for RAM.
    
    print("Preprocessing and augmenting training data...")
    train_images_aug = []
    train_masks_aug = []
    train_labels_aug = []
    group_ids: List[int] = []
    is_original_flags: List[bool] = []
    
    for idx, (path, label) in enumerate(tqdm(zip(train_paths, train_labels), total=len(train_paths))):
        img, mask = preprocess_image(path, args.image_size)
        if args.augment:
            aug_pairs = augment_image_and_mask(img, mask, label in THIN_CLASSES)
            for aug_idx, (aug_img, aug_mask) in enumerate(aug_pairs):
                train_images_aug.append(aug_img)
                train_masks_aug.append(aug_mask)
                train_labels_aug.append(label)
                group_ids.append(idx)
                is_original_flags.append(aug_idx == 0)
        else:
            train_images_aug.append(img)
            train_masks_aug.append(mask)
            train_labels_aug.append(label)
            group_ids.append(idx)
            is_original_flags.append(True)
            
    print(f"Total training samples after augmentation: {len(train_images_aug)}")
    
    # 3. Preprocess Test Data
    print("Preprocessing test data...")
    test_images = []
    test_masks = []
    for path in tqdm(test_paths):
        img, mask = preprocess_image(path, args.image_size)
        test_images.append(img)
        test_masks.append(mask)
        
    # 4. Bag of Visual Words (SIFT)
    print("Building Bag of Visual Words vocabulary...")
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=500) if args.orb_vocab_size > 0 else None
    all_descriptors = []
    orb_all_descriptors: List[np.ndarray] = [] if orb is not None else []
    
    # Collect descriptors from a subset of training data to build vocab (to save time)
    # Using all augmented data might be too much, let's use every 5th sample or just original ones?
    # Let's use all for better quality if it fits in memory.
    
    descriptors_list_train = [] # Store per image to avoid re-computing
    descriptors_list_train_orb: Optional[List[Optional[np.ndarray]]] = [] if orb is not None else None
    
    for img, mask in tqdm(zip(train_images_aug, train_masks_aug), desc="Keypoint Extract Train", total=len(train_images_aug)):
        des = get_sift_descriptors(img, mask, sift)
        descriptors_list_train.append(des)
        if des is not None:
            all_descriptors.append(des)

        if orb is not None and descriptors_list_train_orb is not None:
            des_orb = get_orb_descriptors(img, mask, orb)
            descriptors_list_train_orb.append(des_orb)
            if des_orb is not None:
                orb_all_descriptors.append(des_orb)
            
    if not all_descriptors:
        print("Error: No SIFT descriptors found in training set!")
        return

    # Stack a random subset of descriptors for KMeans to speed up
    all_descriptors_stacked = np.vstack(all_descriptors)
    print(f"Total descriptors found: {all_descriptors_stacked.shape[0]}")
    
    # Limit descriptors for KMeans training to e.g. 100k to keep it fast
    if all_descriptors_stacked.shape[0] > 200000:
        indices = np.random.choice(all_descriptors_stacked.shape[0], 200000, replace=False)
        training_descriptors = all_descriptors_stacked[indices]
    else:
        training_descriptors = all_descriptors_stacked
        
    print(f"Clustering {training_descriptors.shape[0]} descriptors into {args.vocab_size} visual words...")
    kmeans = MiniBatchKMeans(n_clusters=args.vocab_size, random_state=42, batch_size=1000, n_init='auto')
    kmeans.fit(training_descriptors)

    orb_kmeans = None
    if orb is not None and descriptors_list_train_orb is not None:
        if orb_all_descriptors:
            orb_stack = np.vstack(orb_all_descriptors)
            if orb_stack.shape[0] > 200000:
                indices = np.random.choice(orb_stack.shape[0], 200000, replace=False)
                orb_training_descriptors = orb_stack[indices]
            else:
                orb_training_descriptors = orb_stack
            print(
                f"Clustering {orb_training_descriptors.shape[0]} ORB descriptors into {args.orb_vocab_size} visual words..."
            )
            orb_kmeans = MiniBatchKMeans(
                n_clusters=args.orb_vocab_size,
                random_state=42,
                batch_size=2000,
                n_init='auto',
            )
            orb_kmeans.fit(orb_training_descriptors)
        else:
            print("Warning: No ORB descriptors detected; disabling ORB vocab.")
            orb = None
            descriptors_list_train_orb = None

    test_orb_descriptors: Optional[List[Optional[np.ndarray]]] = None
    if orb_kmeans is not None and orb is not None:
        print("Extracting ORB descriptors for test set...")
        test_orb_descriptors = []
        for img, mask in tqdm(zip(test_images, test_masks), total=len(test_images), desc="ORB Extract Test"):
            test_orb_descriptors.append(get_orb_descriptors(img, mask, orb))
    
    # 5. Construct Feature Matrices
    def build_features(images, masks, descriptors_list=None, orb_descriptors_list=None):
        features = []
        for i, (img, mask) in enumerate(tqdm(zip(images, masks), total=len(images), desc="Building Features")):
            # Color
            feat_color = get_color_features(img, mask)
            
            # Texture (45 features)
            feat_texture = get_texture_features(img, mask)

            # Shape (45 features)
            feat_shape = get_shape_features(mask)

            # Width (7 features)
            feat_width = get_width_features(mask)

            # Radial signature (20 features) & hole stats (5 features)
            feat_radial = get_radial_features(mask)
            feat_holes = get_hole_features(mask)

            # Color-channel gradients (11 features)
            feat_channel_grad = get_channel_gradient_features(img, mask)

            # Orientation (12 features)
            feat_orientation = get_orientation_features(img, mask)

            # LBP (26 features)
            feat_lbp = get_lbp_features(img, mask)
            
            # Multi-scale (15 features)
            feat_multiscale = get_multiscale_features(img, mask)

            # Frequency-domain texture (5 features)
            feat_frequency = get_frequency_features(img, mask)
            
            # BoVW
            if descriptors_list is not None:
                des = descriptors_list[i]
            else:
                des = get_sift_descriptors(img, mask, sift)

            if des is not None:
                words = kmeans.predict(des)
                feat_bovw, _ = np.histogram(words, bins=np.arange(args.vocab_size + 1), density=True)
            else:
                feat_bovw = np.zeros(args.vocab_size, dtype=np.float32)

            feature_parts = [
                feat_color,        # 70 features
                feat_texture,      # 45 features
                feat_shape,        # 45 features
                feat_width,        # 7 features
                feat_radial,       # 20 features
                feat_holes,        # 5 features
                feat_channel_grad, # 11 features
                feat_orientation,  # 12 features
                feat_lbp,          # 26 features
                feat_multiscale,   # 15 features
                feat_frequency,    # 5 features
                feat_bovw,         # vocab_size features (default 100)
            ]

            if orb_kmeans is not None and orb is not None:
                if orb_descriptors_list is not None:
                    orb_des = orb_descriptors_list[i]
                else:
                    orb_des = get_orb_descriptors(img, mask, orb)

                if orb_des is not None:
                    orb_words = orb_kmeans.predict(orb_des)
                    feat_orb, _ = np.histogram(
                        orb_words,
                        bins=np.arange(args.orb_vocab_size + 1),
                        density=True,
                    )
                else:
                    feat_orb = np.zeros(args.orb_vocab_size, dtype=np.float32)

                feature_parts.append(feat_orb.astype(np.float32))

            # Concatenate
            features.append(np.concatenate(feature_parts))
            
        return np.vstack(features)

    print("Constructing training feature matrix...")
    X_train = build_features(
        train_images_aug,
        train_masks_aug,
        descriptors_list_train,
        descriptors_list_train_orb if orb_kmeans is not None else None,
    )
    y_train = np.array(train_labels_aug)
    group_ids = np.array(group_ids)
    is_original_mask = np.array(is_original_flags, dtype=bool)
    original_labels = np.array(train_labels)
    original_image_ids = np.array([p.name for p in train_paths])
    class_names = np.array(sorted(set(train_labels)))
    
    print("Constructing test feature matrix...")
    X_test = build_features(
        test_images,
        test_masks,
        None,
        test_orb_descriptors,
    )
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # 6. Model Training & CV
    # Define Ensemble
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ('svc', SVC(kernel='rbf', C=100.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
    ]
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42,
            eval_metric='mlogloss'
        )))
        
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ensemble)
    ])

    thin_refiner = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=20.0, gamma='scale', probability=False, class_weight='balanced', random_state=42)),
    ])

    thin_detector = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        class_weight='balanced',
        l2_regularization=1e-3,
        random_state=42,
    )
    
    print(f"Performing {args.n_splits}-fold Cross-Validation (grouped by original image)...")
    cv_scores = run_grouped_cv(
        pipeline,
        thin_refiner,
        thin_detector,
        X_train,
        y_train,
        group_ids,
        is_original_mask,
        original_labels,
        original_image_ids,
        class_names,
        args.n_splits,
        args.thin_threshold,
    )

    print("Training final model on full augmented data...")
    pipeline.fit(X_train, y_train)

    thin_refiner_full = None
    thin_mask_full = np.isin(y_train, list(THIN_CLASSES))
    if thin_mask_full.sum() >= 2 and len(np.unique(y_train[thin_mask_full])) == len(THIN_CLASSES):
        thin_refiner_full = clone(thin_refiner)
        thin_refiner_full.fit(X_train[thin_mask_full], y_train[thin_mask_full])
        print("Trained thin-class refiner on full data.")
    else:
        print("Warning: insufficient samples to train thin-class refiner; skipping.")

    thin_detector_full = None
    thin_indicator_full = thin_mask_full.astype(int)
    if thin_indicator_full.min() != thin_indicator_full.max():
        thin_detector_full = clone(thin_detector)
        thin_detector_full.fit(X_train, thin_indicator_full)
        print("Trained thin detector on full data.")
    else:
        print("Warning: insufficient diversity to train thin detector; skipping.")
    
    print("Predicting test set...")
    preds = pipeline.predict(X_test)
    if thin_refiner_full is not None:
        thin_pred_mask = np.isin(preds, list(THIN_CLASSES))
        if thin_pred_mask.any():
            refined = thin_refiner_full.predict(X_test[thin_pred_mask])
            preds[thin_pred_mask] = refined

    if thin_detector_full is not None and thin_refiner_full is not None:
        thin_probs = thin_detector_full.predict_proba(X_test)[:, 1]
        escalate_mask = (~np.isin(preds, list(THIN_CLASSES))) & (thin_probs >= args.thin_threshold)
        if escalate_mask.any():
            refined = thin_refiner_full.predict(X_test[escalate_mask])
            preds[escalate_mask] = refined
    
    submission = pd.DataFrame({"ID": test_ids, "Category": preds})
    submission.to_csv(args.submission_path, index=False)
    print(f"Submission file saved to {args.submission_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
