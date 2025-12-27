from dataclasses import dataclass
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class TrainResult:
    best_model: Pipeline
    best_params: Dict[str, Any]
    test_accuracy: float


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf")),
        ]
    )


def train_svm_gridsearch(
    data_train: np.ndarray,
    labels_train: np.ndarray,
    data_test: np.ndarray,
    labels_test: np.ndarray,
) -> TrainResult:
    pipe = build_pipeline()

    param_grid = {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", "auto", 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(data_train, labels_train)
    best_model = grid.best_estimator_

    test_accuracy = float(best_model.score(data_test, labels_test))
    return TrainResult(
        best_model=best_model,
        best_params=dict(grid.best_params_),
        test_accuracy=test_accuracy,
    )


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_label(model: Pipeline, x: np.ndarray) -> int:
    pred = model.predict(x.reshape(1, -1))
    return int(pred[0])
