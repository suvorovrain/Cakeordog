import argparse
import os

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


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    x = load_single_image(args.image)
    label = predict_label(model, x)
    print(CATEGORIES[label])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cake_or_dog", description="Muffin vs Chihuahua classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save it")
    p_train.add_argument("--train-dir", required=True, help="Path to data/train")
    p_train.add_argument("--test-dir", required=True, help="Path to data/test")
    p_train.add_argument("--model-out", required=True, help="Where to save the trained model .joblib")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict using a saved model")
    p_pred.add_argument("--model", required=True, help="Path to .joblib model")
    p_pred.add_argument("--image", required=True, help="Path to an image file")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
