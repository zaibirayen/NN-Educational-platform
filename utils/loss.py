import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_grad(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_grad(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true / y_pred / y_true.shape[0]
