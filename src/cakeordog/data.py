import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, cast

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.io import imread
from skimage.transform import resize

CATEGORIES: List[str] = ["muffin", "chihuahua"]
IMAGE_SIZE: Tuple[int, int] = (32, 32)


def _ensure_rgb(img: np.ndarray, path: str) -> np.ndarray:
    if img.ndim == 2:
        return cast(np.ndarray, gray2rgb(img))
    if img.ndim == 3 and img.shape[2] == 4:
        return cast(np.ndarray, rgba2rgb(img))
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unexpected image shape {img.shape} for {path}")


def _load_one(args: Tuple[str, int]) -> Tuple[np.ndarray, int]:
    path, label = args
    img = imread(path)
    img = _ensure_rgb(img, path)
    img = resize(img, IMAGE_SIZE, anti_aliasing=True, preserve_range=True)
    x = img.reshape(-1).astype(np.float32)
    return x, label


def load_split_parallel(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    tasks: List[Tuple[str, int]] = []
    for label, category in enumerate(CATEGORIES):
        cat_dir = os.path.join(root_dir, category)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")

        for fname in os.listdir(cat_dir):
            path = os.path.join(cat_dir, fname)
            if os.path.isfile(path):
                tasks.append((path, label))

    if not tasks:
        raise ValueError(f"No files found under: {root_dir}")

    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_load_one, tasks, chunksize=32))

    data, labels = zip(*out, strict=False)
    return np.stack(data), np.asarray(labels, dtype=np.int64)


def load_single_image(path: str) -> np.ndarray:
    img = imread(path)
    img = _ensure_rgb(img, path)
    img = resize(img, IMAGE_SIZE, anti_aliasing=True, preserve_range=True)
    x = img.reshape(-1).astype(np.float32)
    return cast(np.ndarray, x)
