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
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# Default relative paths (can be overridden through CLI arguments)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "dataset-for-task1" / "dataset-for-task1"
DEFAULT_TRAIN_DIR = DEFAULT_DATASET_ROOT / "train"
DEFAULT_TEST_DIR = DEFAULT_DATASET_ROOT / "test"
DEFAULT_SUBMISSION_PATH = BASE_DIR / "submission-for-task1.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical ML classifier (HOG + color + LBP features) and generate predictions."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Path to training images")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Path to test images")
    parser.add_argument(
        "--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH, help="Where to write submission CSV"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=192,
        help="Square size (pixels) each image is resized to before feature extraction",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits used to report cross-validation Mean F1 macro",
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


def color_histogram(rgb_image: np.ndarray, bins: Tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
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


def lbp_histogram(rgb_image: np.ndarray, radius: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    n_points = radius * 8
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    return hist


def extract_feature_vector(image_path: Path, target_size: int) -> np.ndarray:
    rgb_image = preprocess_image(image_path, target_size)
    features = np.concatenate([
        color_histogram(rgb_image),
        hog_features(rgb_image),
        lbp_histogram(rgb_image),
    ])
    return features


def build_feature_matrix(paths: Sequence[Path], target_size: int) -> np.ndarray:
    feature_list: List[np.ndarray] = []
    for path in tqdm(paths, desc="Extracting features"):
        feature_list.append(extract_feature_vector(path, target_size))
    return np.vstack(feature_list)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=42,
                ),
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    train_paths, labels = load_training_metadata(args.train_dir)
    test_paths, test_ids = load_test_metadata(args.test_dir)

    print(f"Loaded {len(train_paths)} training images across {len(set(labels))} classes")
    print(f"Loaded {len(test_paths)} test images")

    train_features = build_feature_matrix(train_paths, args.image_size)
    test_features = build_feature_matrix(test_paths, args.image_size)

    model = build_model()
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, train_features, labels, cv=cv, scoring="f1_macro")
    print(
        f"Cross-validation Mean F1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} ({args.n_splits}-fold)"
    )

    model.fit(train_features, labels)
    predictions = model.predict(test_features)

    submission = pd.DataFrame({"ID": test_ids, "Category": predictions})
    submission.to_csv(args.submission_path, index=False)
    print(f"Submission file saved to {args.submission_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
