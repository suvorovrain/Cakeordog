"""Tests for classificator"""

from pathlib import Path

from cakeordog.data import load_split_parallel
from cakeordog.model import load_model

NEEDED_ACCURACY = 0.8
TESTS_PATH = "data/test"
MODEL_PATH = "models/svc_muffin_chihuahua.joblib"


def test_accuracy_with_pretrained_model() -> None:
    """Test accuracy using a pretrained model"""

    test_dir = Path(TESTS_PATH)

    assert test_dir.exists(), "Test directory not found"

    model_path = Path(MODEL_PATH)

    assert model_path.exists(), "Model not found"

    data_test, labels_test = load_split_parallel(str(test_dir))

    model = load_model(str(model_path))

    test_accuracy = model.score(data_test, labels_test)

    print(f"Pretrained model accuracy: {test_accuracy:.2%}")

    assert (
        test_accuracy > NEEDED_ACCURACY
    ), f"Pretrained model accuracy {test_accuracy:.2%} is below needed level"
