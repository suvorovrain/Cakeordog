import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_five() -> int:
    """
    Returns 5.

    Returns:
        int: Always 5

    Examples:
        >>> get_five()
        5
    """
    return 5


def load_one(args):
    path, label = args

    img = imread(path)

    if img.ndim == 2:  # grayscale
        img = gray2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
        img = rgba2rgb(img)
    elif img.ndim == 3 and img.shape[2] != 3:
        raise ValueError(f"Unexpected channel count {img.shape[2]} for {path}")

    img = resize(img, (32, 32), anti_aliasing=True, preserve_range=True)

    x = img.reshape(-1).astype(np.float32)
    return x, label


def load_split_parallel(root_dir, categories, max_workers=None):
    tasks = []
    for label, category in enumerate(categories):
        cat_dir = os.path.join(root_dir, category)
        for fname in os.listdir(cat_dir):
            tasks.append((os.path.join(cat_dir, fname), label))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(load_one, tasks, chunksize=32))

    data, labels = zip(*out)
    return np.stack(data), np.asarray(labels, dtype=np.int64)


def main() -> None:
    """Entry point"""
    result = get_five()
    print(f"Result: {result}")
    input_train_dir = "/home/suvorovrain/Projects/Cakeordog/data/train"
    input_test_dir = "/home/suvorovrain/Projects/Cakeordog/data/test"
    categories = ["muffin", "chihuahua"]

    data_train, labels_train = load_split_parallel(input_train_dir, categories)
    data_test, labels_test = load_split_parallel(input_test_dir, categories)
    print("reading done")

    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])

    param_grid = {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", "auto", 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid.fit(data_train, labels_train)
    best_model = grid.best_estimator_

    test_accuracy = best_model.score(data_test, labels_test)
    print("Best params:", grid.best_params_)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
