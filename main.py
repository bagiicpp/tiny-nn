import numpy as np

from tiny_nn.layers import dense_backward, dense_forward


def main():
    # Weight matrix: (3 neurons, 4 input features)
    W = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 4]])

    # Input: (4 features, 1 example)
    x = np.array([[1], [6], [10], [5]])

    # Biases: (3 neurons, 1)
    b = np.array([[0.2], [0.6], [-0.1]])

    # Incoming gradients: (3 neurons, 1 example)
    dZ = np.array([[2], [-1], [0.5]])

    # Forward pass
    z = dense_forward(W, x, b)

    print("Forward output:")
    print(z)
    print("Shape:", z.shape)

    # Backward pass
    dW, db, dX = dense_backward(W, x, dZ)

    print("\nWeight gradients:")
    print(dW)
    print("Shape:", dW.shape)

    print("\nBias gradients:")
    print(db)
    print("Shape:", db.shape)

    print("\nInput gradients:")
    print(dX)
    print("Shape:", dX.shape)


if __name__ == "__main__":
    main()
