"""
Command Line Interface (CLI) for the Cake or Dog classifier.

This module provides a command-line interface for training and using
a binary image classifier that distinguishes between muffins and chihuahuas.

Commands:
    1. train: Train a new model on labeled data
    2. predict: Make predictions on images using a trained model
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from cakeordog.data import CATEGORIES, load_single_image, load_split_parallel
from cakeordog.model import load_model, predict_label, save_model, train_svm_gridsearch


def cmd_train(args: argparse.Namespace) -> int:
    """
    Train an SVM classifier using grid search and save the best model.

    This function orchestrates the training pipeline:
    1. Loads training and test data in parallel
    2. Performs hyperparameter grid search to find optimal SVM parameters
    3. Saves the best performing model to disk
    4. Prints training results to stdout

    Args:
        args: Command-line arguments containing:
            - train_dir: Path to training data directory
            - test_dir: Path to test data directory
            - model_out: Output path for the trained model

    Returns:
        int: Exit code (0 for success)

    Raises:
        FileNotFoundError: If training or test directories don't exist
        ValueError: If data loading fails or grid search encounters issues
    """
    data_train, labels_train = load_split_parallel(args.train_dir)
    data_test, labels_test = load_split_parallel(args.test_dir)

    result = train_svm_gridsearch(
        data_train=data_train,
        labels_train=labels_train,
        data_test=data_test,
        labels_test=labels_test,
    )

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    save_model(result.best_model, args.model_out)

    print("Best params:", result.best_params)
    print("Test accuracy:", result.test_accuracy)
    print("Saved model:", args.model_out)
    return 0


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
"""
Set of supported image file extensions (case-insensitive).
Extensions should be checked in lowercase using `f.suffix.lower() in _IMAGE_EXTS`.
"""


def _expand_inputs(items: List[str]) -> List[str]:
    """
    Expand various input patterns into a list of image file paths.

    Supports multiple input types:
    - Directories: Expands to all image files within the directory
    - Files: Single image file paths
    - Glob patterns: Shell-style wildcard patterns (e.g., "*.jpg", "data/*.png")

    Args:
        items: List of input strings (paths, directories, or glob patterns)

    Returns:
        List[str]: Sorted, deduplicated list of image file paths

    Raises:
        FileNotFoundError: If an input item doesn't exist
        ValueError: If no images are found after expansion
    """
    out: List[str] = []

    for item in items:
        p = Path(item)

        if p.exists() and p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                    out.append(str(f))
            continue

        if p.exists() and p.is_file():
            out.append(str(p))
            continue

        matches = sorted(Path().glob(item))
        if matches:
            for m in matches:
                if m.is_file():
                    out.append(str(m))
            continue

        raise FileNotFoundError(f"Input not found: {item}")

    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    if not uniq:
        raise ValueError("No images found to predict")

    return uniq


def _true_label_from_path(path: str) -> Optional[int]:
    """
    Extract true label from an image file path based on directory structure.

    Assumes images are organized in category-named directories:
    - 'muffin/' → label 0
    - 'chihuahua/' → label 1

    Args:
        path: File path to analyze

    Returns:
        Optional[int]: Integer label (0 or 1) if parent directory matches a category,
                      None otherwise
    """
    parent = Path(path).parent.name
    if parent in CATEGORIES:
        return CATEGORIES.index(parent)
    return None


def cmd_predict(args: argparse.Namespace) -> int:
    """
    Perform batch prediction on images using a trained model.

    Loads a trained model, processes multiple images in parallel,
    and outputs predictions. If images are in category-named directories,
    calculates and displays accuracy metrics.

    Args:
        args: Command-line arguments containing:
            - model: Path to trained model file
            - images: List of image paths, directories, or glob patterns

    Returns:
        int: Exit code (0 for success)

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If no valid images are found

    Output:
        Prints tab-separated lines: "path<TAB>prediction"
        If ground truth labels are detectable, prints accuracy statistics
    """
    model = load_model(args.model)
    paths = _expand_inputs(args.images)

    def worker(image_path: str) -> Tuple[str, int, Optional[int]]:
        """
        Worker function for parallel prediction.

        Args:
            path: Image file path

        Returns:
            Tuple containing:
                - path: Original file path
                - pred: Predicted label (0 or 1)
                - true_label: Ground truth label if detectable, else None
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is invalid
        """
        x = load_single_image(image_path)
        pred_label = predict_label(model, x)
        extracted_true_label = _true_label_from_path(image_path)
        return image_path, pred_label, extracted_true_label

    workers = min(32, (os.cpu_count() or 1) * 4)

    correct = 0
    total = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, pred, true_label in ex.map(worker, paths, chunksize=32):
            print(f"{path}\t{CATEGORIES[pred]}")

            if true_label is not None:
                total += 1
                if pred == true_label:
                    correct += 1

    if total > 0:
        accuracy = correct / total
        print()
        print(f"Correct: {correct}")
        print(f"Total:   {total}")
        print(f"Accuracy: {accuracy:.4f}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for Cake or Dog classifier.

    Creates a parser with two subcommands:
    1. train: Train a new model
    2. predict: Make predictions using a trained model

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="cake_or_dog",
        description="Muffin vs Chihuahua classifier",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save it")
    p_train.add_argument("--train-dir", required=True)
    p_train.add_argument("--test-dir", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict images")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image paths, directories, or glob patterns",
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main() -> int:
    """
    Main entry point for the Cake or Dog CLI.

    Parses command-line arguments and dispatches to appropriate command function.

    Returns:
        int: Exit code from the executed command function
    """
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
