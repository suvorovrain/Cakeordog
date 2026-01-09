"""
Data loading and preprocessing utilities for the Cake or Dog classifier.

This module handles loading, preprocessing, and parallel processing of image data
for training and inference. All images are converted to a standardized format.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, cast

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.io import imread
from skimage.transform import resize

CATEGORIES: List[str] = ["muffin", "chihuahua"]
"""List of category names. Index corresponds to label (0=muffin, 1=chihuahua)."""

IMAGE_SIZE: Tuple[int, int] = (32, 32)
"""Target image dimensions (height, width) for resizing."""


def _ensure_rgb(img: np.ndarray, path: str) -> np.ndarray:
    """
    Convert image to standard 3-channel RGB format.

    Handles multiple input formats:
    - Grayscale (2D) → RGB by duplicating channels
    - RGBA (4 channels) → RGB by dropping alpha channel
    - RGB (3 channels) → passed through unchanged

    Args:
        img: Input image array
        path: Source file path (for error messages)

    Returns:
        np.ndarray: Image in RGB format with shape (height, width, 3)

    Raises:
        ValueError: If image has unexpected number of dimensions or channels
    """
    if img.ndim == 2:
        return cast(np.ndarray, gray2rgb(img))
    if img.ndim == 3 and img.shape[2] == 4:
        return cast(np.ndarray, rgba2rgb(img))
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unexpected image shape {img.shape} for {path}")


def _load_one(args: Tuple[str, int], anti_aliasing=True) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess a single image file.

    Args:
        args: Tuple containing:
            - path: Image file path
            - label: Integer label (0 for muffin, 1 for chihuahua)

    Returns:
        Tuple[np.ndarray, int]: Flattened image vector and label

    Raises:
        ValueError: If image has unexpected number of dimensions or channels
    """
    path, label = args
    img = imread(path)
    img = _ensure_rgb(img, path)
    img = resize(img, IMAGE_SIZE, anti_aliasing=anti_aliasing, preserve_range=True)
    x = img.reshape(-1).astype(np.float32)
    return x, label


def load_split_parallel(root_dir: str, max_count: int = 1_000_000, anti_aliasing=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all images from a directory structure using parallel processing.

    Args:
        root_dir: Path to root directory containing category subdirectories
        max_count: Max count of processed photos.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - data: 2D array of shape (n_samples, n_features)
            - labels: 1D array of integer labels

    Raises:
        FileNotFoundError: If root_dir or category subdirectories don't exist
        ValueError: If no image files are found
    """
    tasks: List[Tuple[str, int]] = []
    category_count: dict[int] = {}
    for label, category in enumerate(CATEGORIES):
        cat_dir = os.path.join(root_dir, category)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")

        for fname in os.listdir(cat_dir):
            if label in category_count and category_count[label] >= max_count:
                break
            path = os.path.join(cat_dir, fname)
            if os.path.isfile(path):
                tasks.append((path, label))
                if label not in category_count:
                    category_count[label] = 0
                category_count[label] += 1

    if not tasks:
        raise ValueError(f"No files found under: {root_dir}")

    anti_aliasings = [anti_aliasing] * len(tasks)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_load_one, tasks, anti_aliasings, chunksize=32))

    data, labels = zip(*out, strict=False)
    return np.stack(data), np.asarray(labels, dtype=np.int64)


def load_single_image(path: str, anti_aliasing=True) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.

    Args:
        path: Path to image file

    Returns:
        np.ndarray: Flattened image vector ready for model input

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image has invalid format
    """
    img = imread(path)
    img = _ensure_rgb(img, path)
    img = resize(img, IMAGE_SIZE, anti_aliasing=anti_aliasing, preserve_range=True)
    x = img.reshape(-1).astype(np.float32)
    return cast(np.ndarray, x)
