import numpy as np
from tiny_nn.activations import relu_backward, relu_forward
from tiny_nn.layers import dense_backward, dense_forward

W = np.array([
    [1, -2],
    [3,  1]
])

X = np.array([
    [1, 2],
    [2, 1]
])

b = np.array([
    [0],
    [1]
])

dA = np.array([
    [2, 3],
    [4, 5]
])

# Forward propagation
Z = dense_forward(W, X, b)
A = relu_forward(Z)

# Backward propagation
dZ = relu_backward(Z, dA)
dW, db, dX = dense_backward(W, X, dZ)

print("A:", A)
print("dW:", dW)
print("db:", db)
print("dX:", dX)

