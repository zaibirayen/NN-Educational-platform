import numpy as np

from training.optimizer import GradientDescent, MiniBatchGradientDescent
from utils.activation import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax
from utils.loss import mse, binary_cross_entropy, categorical_cross_entropy
from utils.metrics import accuracy_score


class MLPModel:
    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        activation: str = "relu",
        output_activation: str | None = None,
        learning_rate: float = 0.01,
        epochs: int = 200,
        l2: float = 0.0,
        batch_size: int = 0,
        early_stopping: bool = True,
        patience: int = 10,
        dropout: float = 0.0,
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout = dropout
        self.weights = []
        self.biases = []
        self.history = {"loss": [], "accuracy": []}

    def _activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu(Z)
        if self.activation == "tanh":
            return tanh(Z)
        return sigmoid(Z)

    def _activation_derivative(self, A: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu_derivative(A)
        if self.activation == "tanh":
            return tanh_derivative(A)
        return sigmoid_derivative(A)

    def _output_activation(self, Z: np.ndarray) -> np.ndarray:
        if self.output_activation == "softmax":
            return softmax(Z)
        if self.output_activation == "sigmoid":
            return sigmoid(Z)
        return Z

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray, task: str) -> float:
        if task == "regression":
            return mse(y_true, y_pred)
        if task == "binary":
            return binary_cross_entropy(y_true, y_pred.squeeze())
        return categorical_cross_entropy(y_true, y_pred)

    def _loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray, task: str) -> np.ndarray:
        if task == "regression":
            return 2 * (y_pred - y_true) / y_true.size
        if task == "binary":
            return (y_pred.squeeze() - y_true)[:, None] / y_true.size
        return (y_pred - y_true) / y_true.shape[0]

    def _initialize_weights(self, input_dim: int, output_dim: int):
        layer_dims = [input_dim, *self.layer_sizes, output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_dims) - 1):
            limit = np.sqrt(6 / (layer_dims[i] + layer_dims[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_dims[i], layer_dims[i + 1])))
            self.biases.append(np.zeros((1, layer_dims[i + 1])))

    def _forward(self, X: np.ndarray, training: bool = True):
        activations = [X]
        preactivations = []
        dropout_masks = []

        A = X
        for i in range(len(self.weights) - 1):
            Z = A.dot(self.weights[i]) + self.biases[i]
            preactivations.append(Z)
            A = self._activation(Z)
            if training and self.dropout > 0:
                mask = (np.random.rand(*A.shape) > self.dropout).astype(float) / (1.0 - self.dropout)
                A *= mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(np.ones_like(A))
            activations.append(A)

        Z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        preactivations.append(Z)
        A = self._output_activation(Z)
        activations.append(A)
        return activations, preactivations, dropout_masks

    def _backward(self, activations, preactivations, dropout_masks, y_true, task):
        grads_W = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        y_pred = activations[-1]
        dA = self._loss_gradient(y_true, y_pred, task)

        if self.output_activation == "softmax" or task == "binary":
            dZ = dA
        else:
            dZ = dA

        dA_prev = dZ
        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]
            grads_W[i] = A_prev.T.dot(dA_prev) / A_prev.shape[0] + self.l2 * self.weights[i]
            grads_b[i] = np.mean(dA_prev, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dA_prev.dot(self.weights[i].T) * self._activation_derivative(activations[i])
                dA_prev *= dropout_masks[i - 1]

        return grads_W, grads_b

    def fit(self, X: np.ndarray, y: np.ndarray, task: str):
        if task == "regression" and y.ndim == 1:
            y = y.reshape(-1, 1)

        output_dim = 1 if task == "binary" else (y.shape[1] if task == "multi" else 1)
        self._initialize_weights(X.shape[1], output_dim)
        optimizer = (
            MiniBatchGradientDescent(self.learning_rate, self.l2, self.batch_size)
            if self.batch_size and self.batch_size < X.shape[0]
            else GradientDescent(self.learning_rate, self.l2)
        )

        best_loss = float("inf")
        patience_count = 0

        for epoch in range(self.epochs):
            if self.batch_size and self.batch_size < X.shape[0]:
                for X_batch, y_batch in optimizer.get_batches(X, y):
                    activations, preactivations, dropout_masks = self._forward(X_batch, training=True)
                    grads_W, grads_b = self._backward(activations, preactivations, dropout_masks, y_batch, task)
                    self.weights = [optimizer.update(W, dW) for W, dW in zip(self.weights, grads_W)]
                    self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, grads_b)]
            else:
                activations, preactivations, dropout_masks = self._forward(X, training=True)
                grads_W, grads_b = self._backward(activations, preactivations, dropout_masks, y, task)
                self.weights = [optimizer.update(W, dW) for W, dW in zip(self.weights, grads_W)]
                self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, grads_b)]

            y_pred = self.predict_proba(X)
            loss = self._loss(y, y_pred, task)
            if task in {"binary", "multi"}:
                true_labels = np.argmax(y, axis=1) if task == "multi" else y.squeeze()
                accuracy = accuracy_score(true_labels, self.predict(X))
            else:
                accuracy = 0.0
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            if self.early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _, _ = self._forward(X, training=False)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.predict_proba(X)
        if self.output_activation == "softmax":
            return np.argmax(y_pred, axis=1)
        if self.output_activation == "sigmoid":
            return (y_pred.squeeze() >= 0.5).astype(int)
        return y_pred.squeeze()
