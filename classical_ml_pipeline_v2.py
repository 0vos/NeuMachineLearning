#!/usr/bin/env python3
"""Classical (non-deep-learning) pipeline for plant seedling classification."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Default relative paths (can be overridden through CLI arguments)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "dataset-for-task1" / "dataset-for-task1"
DEFAULT_TRAIN_DIR = DEFAULT_DATASET_ROOT / "train"
DEFAULT_TEST_DIR = DEFAULT_DATASET_ROOT / "test"
DEFAULT_SUBMISSION_PATH = BASE_DIR / "submission-for-task1.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical ML classifier (HOG + color + LBP + Haralick features) and generate predictions."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Path to training images")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Path to test images")
    parser.add_argument(
        "--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH, help="Where to write submission CSV"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square size (pixels) each image is resized to before feature extraction",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits used to report cross-validation Mean F1 macro",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Augment training data with flips and rotations",
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
        if not current_paths:
            raise FileNotFoundError(f"No images found in {class_dir}")
        image_paths.extend(current_paths)
        labels.extend([class_dir.name] * len(current_paths))
    return image_paths, labels


def load_test_metadata(test_dir: Path) -> Tuple[List[Path], List[str]]:
    paths = list_image_paths(test_dir)
    if not paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    ids = [p.name for p in paths]
    return paths, ids


def preprocess_image(path: Path, target_size: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return image


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """Generate augmented versions of an image (flips and rotations)."""
    augmented = [image]
    # Flips
    augmented.append(cv2.flip(image, 0))  # Vertical
    augmented.append(cv2.flip(image, 1))  # Horizontal
    augmented.append(cv2.flip(image, -1)) # Both
    # Rotations
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented.append(cv2.rotate(image, cv2.ROTATE_180))
    return augmented


def color_histogram(rgb_image: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def hog_features(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype("float32") / 255.0
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features


def lbp_histogram(rgb_image: np.ndarray, radius: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    n_points = radius * 8
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    return hist


def haralick_features(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Quantize to fewer levels for GLCM (e.g., 32 levels) to reduce sparsity
    gray = (gray // 8).astype(np.uint8)
    
    # Compute GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        # Average over angles and distances
        val = graycoprops(glcm, prop).ravel()
        features.extend([val.mean(), val.std()])
        
    return np.array(features)


def extract_feature_vector_from_image(rgb_image: np.ndarray) -> np.ndarray:
    features = np.concatenate([
        color_histogram(rgb_image),
        hog_features(rgb_image),
        lbp_histogram(rgb_image),
        haralick_features(rgb_image)
    ])
    return features


def build_feature_matrix(paths: Sequence[Path], target_size: int, augment: bool = False, labels: Sequence[str] | None = None) -> Tuple[np.ndarray, Sequence[str] | None]:
    feature_list: List[np.ndarray] = []
    augmented_labels: List[str] = []
    
    for i, path in enumerate(tqdm(paths, desc="Extracting features")):
        try:
            original_image = preprocess_image(path, target_size)
            images_to_process = [original_image]
            
            if augment:
                images_to_process = augment_image(original_image)
            
            for img in images_to_process:
                feature_list.append(extract_feature_vector_from_image(img))
                if labels is not None:
                    augmented_labels.append(labels[i])
                    
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
            
    if labels is not None:
        return np.vstack(feature_list), augmented_labels
    return np.vstack(feature_list), None


def build_model() -> Pipeline:
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('svc', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
    ]
    
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42,
            eval_metric='mlogloss'
        )))
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", ensemble),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    train_paths, train_labels = load_training_metadata(args.train_dir)
    test_paths, test_ids = load_test_metadata(args.test_dir)

    print(f"Loaded {len(train_paths)} training images across {len(set(train_labels))} classes")
    print(f"Loaded {len(test_paths)} test images")

    # Extract features for training data (with augmentation)
    print("Processing training data...")
    train_features, augmented_labels = build_feature_matrix(train_paths, args.image_size, augment=args.augment, labels=train_labels)
    print(f"Training feature matrix shape: {train_features.shape}")

    # Perform Cross-Validation
    print(f"Performing {args.n_splits}-fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    f1_scores = []
    
    # Map original indices to augmented indices
    # Each original image i produced n_aug images
    n_aug = 7 if args.augment else 1
    
    y_all_aug = np.array(augmented_labels)
    y_original = np.array(train_labels)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_paths, y_original)):
        # Construct training set (augmented)
        train_indices_aug = []
        for i in train_idx:
            train_indices_aug.extend(range(i * n_aug, (i + 1) * n_aug))
            
        X_train_fold = train_features[train_indices_aug]
        y_train_fold = y_all_aug[train_indices_aug]
        
        # Construct validation set (original images only, to avoid leakage)
        val_indices_orig = val_idx * n_aug
        X_val_fold = train_features[val_indices_orig]
        y_val_fold = y_original[val_idx]
        
        model_fold = build_model()
        model_fold.fit(X_train_fold, y_train_fold)
        preds_fold = model_fold.predict(X_val_fold)
        
        score = f1_score(y_val_fold, preds_fold, average='macro')
        f1_scores.append(score)
        print(f"Fold {fold+1} F1: {score:.4f}")
        
    print(f"Mean CV F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # Extract features for test data (no augmentation)
    print("Processing test data...")
    test_features, _ = build_feature_matrix(test_paths, args.image_size, augment=False)

    model = build_model()
    
    # For CV, we should be careful not to leak augmented data into validation.
    # However, since we augmented *before* splitting, standard CV would leak.
    # To do this correctly, we should split first, then augment.
    # But for simplicity in this script, let's just report CV on the augmented set (optimistic)
    # OR better: Use GroupKFold if we had groups, but we don't track original image ID easily here.
    # Let's just fit on everything and rely on the ensemble robustness.
    # If user wants strict CV, we'd need to refactor to split paths first.
    
    # Let's do a quick CV on the *original* data to get a realistic score, 
    # but that requires extracting features without augmentation first.
    # To save time, I will skip the "pure" CV step and just train on the augmented dataset.
    # The user cares about the final submission score.
    
    print("Training model on augmented data...")
    model.fit(train_features, augmented_labels)
    
    # Predict
    predictions = model.predict(test_features)

    submission = pd.DataFrame({"ID": test_ids, "Category": predictions})
    submission.to_csv(args.submission_path, index=False)
    print(f"Submission file saved to {args.submission_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
