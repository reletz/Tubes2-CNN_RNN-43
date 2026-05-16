from __future__ import annotations

import numpy as np


def get_xp(x):
    if type(x).__module__ == 'cupy':
        import cupy as cp
        return cp
    return np


class Flatten:
    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        xp = get_xp(x)
        if x.ndim == 3:
            return xp.reshape(x, (1, -1))
        if x.ndim == 4:
            N = x.shape[0]
            return xp.reshape(x, (N, -1))
        raise ValueError("Flatten.forward expects input with 3 or 4 dimensions")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("Cannot call backward() before forward()")
        xp = get_xp(grad)
        return xp.reshape(grad, self.input_shape)

