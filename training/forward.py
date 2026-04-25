import numpy as np


def linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X.dot(W) + b


def activation_forward(Z: np.ndarray, activation: str) -> np.ndarray:
    from utils.activation import sigmoid, relu, tanh, softmax
    if activation == "sigmoid":
        return sigmoid(Z)
    if activation == "relu":
        return relu(Z)
    if activation == "tanh":
        return tanh(Z)
    if activation == "softmax":
        return softmax(Z)
    return Z
