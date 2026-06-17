import numpy as np


def generate_coords(n_row, n_col):
    x = np.linspace(-1.0, 1.0, n_row)
    y = np.linspace(-1.0, 1.0, n_col)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    return coords.astype(np.float32)