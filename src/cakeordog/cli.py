import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from cakeordog.data import load_split_parallel, load_single_image, CATEGORIES
from cakeordog.model import train_svm_gridsearch, save_model, load_model, predict_label


def cmd_train(args: argparse.Namespace) -> int:
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


def _expand_inputs(items: List[str]) -> List[str]:
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
    parent = Path(path).parent.name
    if parent in CATEGORIES:
        return CATEGORIES.index(parent)
    return None


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    paths = _expand_inputs(args.images)

    def worker(path: str) -> Tuple[str, int, Optional[int]]:
        x = load_single_image(path)
        pred = predict_label(model, x)
        true_label = _true_label_from_path(path)
        return path, pred, true_label

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
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
