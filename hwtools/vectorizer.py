import numpy as np


def vertical_proj(image: np.ndarray):
    return np.sum(image == 1, axis=1)


def histo(proj: np.ndarray):
    x = list(range(1, np.amax(proj) + 1))
    y = []
    for i in x:
        y.append(np.where(proj == i)[0].size)
    return np.array([x, y])


A = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]).reshape((4, 3))
print(
    A,
    vertical_proj(A),
    np.where(vertical_proj(A)==2)
)
print(histo(vertical_proj(A)))

# def align(histo1,histo2):
