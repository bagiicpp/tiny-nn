import numpy as np


# z = Wx + b
def dense_forward(W, x, b):
    if W.shape[1] != x.shape[0]:
        raise ValueError(
            f"W input size {W.shape[1]}, got {x.shape[0]}, which doesn't match"
        )

    if W.shape[0] != b.shape[0]:
        raise ValueError(
            f"W and b must both have the same number of neurons {W.shape[0]} for the operation to make mathematical sense"
        )

    if b.shape[1] != 1:
        raise ValueError(f"Bias must have exactly one column, not {b.shape[1]}")

    z = (W @ x) + b

    return z


def dense_backward_weights(dZ, x):
    if dZ.ndim != 2:
        raise ValueError("Array must be 2 dimensional")

    if x.ndim != 2:
        raise ValueError("Array must be 2 dimensional")

    if dZ.shape[1] != x.shape[1]:
        raise ValueError("Arrays must have the same number of columns")

    if dZ.shape[1] == 0 or dZ.shape[0] == 0:
        raise ValueError("dZ is not empty")


    if x.shape[1] == 0 or x.shape[0] == 0:
        raise ValueError("x is not empty")

    dW = dZ @ x.T
    return dW


def dense_backward_bias(dZ):
    if dZ.ndim != 2:
        raise ValueError("Array must be 2 dimensional")

    db = np.sum(dZ, axis=1, keepdims=True)
    return db


def dense_backward_inputs(W, dZ):
    if W.ndim != 2 or dZ.ndim != 2:
        raise ValueError("Both arrays must be two-dimensional.")

    # W = (3, 4) dZ = (3, 1)
    if W.shape[0] != dZ.shape[0]:
        raise ValueError("Shape mismatch.")

    if W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("W array must be positve.")

    if dZ.shape[0] <= 0 or dZ.shape[1] <= 0:
        raise ValueError("dZ array must be positive.")

    dX = W.T @ dZ
    return dX


def dense_backward(W, x, dZ):

    if W.shape[1] != x.shape[0]:
        raise ValueError("Input feature dimensions must match")

    dW = dense_backward_weights(dZ, x)
    db = dense_backward_bias(dZ)
    dX = dense_backward_inputs(W, dZ)

    return dW, db, dX
