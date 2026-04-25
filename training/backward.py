import numpy as np


def linear_backward(dZ: np.ndarray, A_prev: np.ndarray):
    dW = A_prev.T.dot(dZ) / A_prev.shape[0]
    db = np.mean(dZ, axis=0, keepdims=True)
    dA_prev = dZ.dot(dW.T)
    return dA_prev, dW, db


def activation_backward(dA: np.ndarray, Z: np.ndarray, activation: str) -> np.ndarray:
    from utils.activation import sigmoid_derivative, relu_derivative, tanh_derivative
    if activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
        return dA * sigmoid_derivative(A)
    if activation == "relu":
        return dA * relu_derivative(Z)
    if activation == "tanh":
        A = np.tanh(Z)
        return dA * tanh_derivative(A)
    return dA
