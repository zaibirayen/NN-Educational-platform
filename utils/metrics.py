import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix, labels


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    results = {}
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return results


def mse_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def rmse_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse_value(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
