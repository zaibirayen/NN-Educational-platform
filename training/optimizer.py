import numpy as np


class GradientDescent:
    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2

    def update(self, W: np.ndarray, dW: np.ndarray) -> np.ndarray:
        regularization = self.l2 * W
        return W - self.learning_rate * (dW + regularization)


class MiniBatchGradientDescent(GradientDescent):
    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0, batch_size: int = 32):
        super().__init__(learning_rate, l2)
        self.batch_size = batch_size

    def get_batches(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield X[batch_idx], y[batch_idx]
