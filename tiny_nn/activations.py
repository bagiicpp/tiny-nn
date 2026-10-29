import numpy as np

def relu_backward(Z, dA):
    if dA.ndim != 2 or Z.ndim != 2:
        raise ValueError("Both arrays must be 2-dimensional")

    if dA.shape != Z.shape:
        raise ValueError("Shapes of arrays must match")

    dZ = dA * (Z > 0)

    return dZ

def relu_forward(Z):
    if Z.ndim != 2:
        raise ValueError("Array must be 2-dimensional")

    return np.maximum(Z, 0)
