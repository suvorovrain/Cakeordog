"""
Machine learning model utilities for the Cake or Dog classifier.

This module provides training, evaluation, and inference functions using
Scikit-learn's SVM classifier with grid search for hyperparameter tuning.
"""

from dataclasses import dataclass
from typing import Any, Dict

import joblib  # type: ignore
import numpy as np
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore


@dataclass(frozen=True)
class TrainResult:
    """
    Container for training results.

    Attributes:
        best_model: The optimal pipeline found during grid search
        best_params: Dictionary of the best hyperparameter values
        test_accuracy: Accuracy score on the test set (0.0 to 1.0)
    """

    best_model: Pipeline
    best_params: Dict[str, Any]
    test_accuracy: float


def build_pipeline() -> Pipeline:
    """
    Create a standardized preprocessing and classification pipeline.

    Returns:
        Pipeline: Scikit-learn pipeline with two steps:
            1. StandardScaler: Standardizes features (zero mean, unit variance)
            2. SVC: Support Vector Classifier with RBF kernel
    """
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
    """
    Train an SVM classifier with grid search for hyperparameter optimization.

    Performs 5-fold cross-validation grid search over C and gamma parameters,
    then evaluates the best model on the test set.

    Args:
        data_train: Training data of shape (n_train_samples, n_features)
        labels_train: Training labels of shape (n_train_samples)
        data_test: Test data of shape (n_test_samples, n_features)
        labels_test: Test labels of shape (n_test_samples)

    Returns:
        TrainResult: Container with best model, parameters, and test accuracy

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid values

    Grid search parameters:
        C (regularization): [0.1, 1, 10, 100, 1000]
        Gamma (kernel width): ['scale', 'auto', 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    """
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
    """
    Save a trained model pipeline to disk using joblib.

    This function serializes a scikit-learn pipeline object to a file,
    preserving the complete model state including:
    - Trained classifier parameters
    - Fitted preprocessing scaler statistics
    - Pipeline metadata and configuration

    Args:
        model: Trained scikit-learn pipeline to save
        path: File path where model will be saved (typically with .pkl or .joblib extension)

    Returns:
        None

    Raises:
        ValueError: If the model object is not serializable
    """
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    """
    Load a saved model pipeline from disk.

    Args:
        path: File path to the saved model

    Returns:
        Pipeline: Loaded scikit-learn pipeline

    Raises:
        ValueError: If the file contains invalid/corrupted data
    """
    return joblib.load(path)


def predict_label(model: Pipeline, x: np.ndarray) -> int:
    """
    Predict class label for a single flattened image vector.

    Args:
        model: Trained pipeline
        x: Flattened image vector of shape (n_features)

    Returns:
        int: Predicted class label (0 for muffin, 1 for chihuahua)

    Raises:
        ValueError: If input shape doesn't match model expectations
    """
    pred = model.predict(x.reshape(1, -1))
    return int(pred[0])
