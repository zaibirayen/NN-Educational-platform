import numpy as np

from utils.activation import sigmoid, softmax
from utils.loss import binary_cross_entropy, categorical_cross_entropy
from utils.metrics import accuracy_score


class PerceptronModel:
    def __init__(self, epochs: int = 200, lr: float = 0.01, l2: float = 0.0):
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.history = {"loss": [], "accuracy": []}
        self.W = None
        self.b = None
        self.classes_ = None

    def _initialize(self, n_features: int, n_classes: int):
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    def fit(self, X: np.ndarray, y: np.ndarray, task: str):
        if task == "regression":
            raise ValueError("Historical Perceptron is only supported for classification.")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = 1 if task == "binary" else self.classes_.size
        self._initialize(X.shape[1], n_classes)

        for epoch in range(self.epochs):
            scores = X.dot(self.W) + self.b
            if task == "binary":
                predictions = (scores.squeeze() >= 0).astype(int)
            else:
                predictions = np.argmax(scores, axis=1)

            mistakes = predictions != y_encoded
            if not np.any(mistakes):
                loss = 0.0
            else:
                if task == "binary":
                    probs = sigmoid(scores.squeeze())
                    loss = binary_cross_entropy(y_encoded, probs)
                else:
                    probs = softmax(scores)
                    one_hot = np.eye(n_classes)[y_encoded]
                    loss = categorical_cross_entropy(one_hot, probs)

            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy_score(y_encoded, predictions))

            for i in range(X.shape[0]):
                xi = X[i : i + 1]
                yi = y_encoded[i]
                si = scores[i]
                pred = predictions[i]
                if pred != yi:
                    if task == "binary":
                        gradient = (yi - pred) * xi.T
                        self.W[:, 0] += self.lr * gradient.squeeze()
                        self.b += self.lr * (yi - pred)
                    else:
                        self.W[:, yi] += self.lr * xi.squeeze()
                        self.W[:, pred] -= self.lr * xi.squeeze()
                        self.b[0, yi] += self.lr
                        self.b[0, pred] -= self.lr

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X.dot(self.W) + self.b
        if self.W.shape[1] == 1:
            return (scores.squeeze() >= 0).astype(int)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = X.dot(self.W) + self.b
        if self.W.shape[1] == 1:
            return sigmoid(scores.squeeze())[:, None]
        return softmax(scores)
