import numpy as np
from tiny_nn.activations import relu_backward

Z = np.array([
    [-2, 3, 0],
    [5, -1, 4]
])

dA = np.array([
    [4, 2, 7],
    [3, 8, 6]
])

dZ = relu_backward(Z, dA)

print(dZ)
print(dZ.shape)
