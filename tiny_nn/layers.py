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


def main():

    # W = (3, 4)
    W = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 4]])

    # x = (4, 5)
    x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 9, 8, 7, 6], [5, 4, 3, 2, 1]])

    # b = (3, 1)
    b = np.array([[0.2], [0.6], [-0.1]])

    dense_forward(W, x, b)


if __name__ == "__main__":
    main()
