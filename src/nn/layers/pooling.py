from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _pool2d(x: np.ndarray, pool_size: Tuple[int, int], strides: Tuple[int, int], mode: str):
    N, H, W, C = x.shape
    kH, kW = pool_size
    sH, sW = strides

    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1

    windows = sliding_window_view(x, window_shape=(kH, kW), axis=(1, 2))
    # windows shape: (N, H-kH+1, W-kW+1, kH, kW, C)
    windows = windows[:, ::sH, ::sW, ...]  # (N, out_h, out_w, kH, kW, C)

    if mode == "max":
        return np.max(windows, axis=(3, 4))
    elif mode == "avg":
        return np.mean(windows, axis=(3, 4))
    else:
        raise ValueError("mode must be 'max' or 'avg'")


class MaxPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(x, self.pool_size, tuple(self.strides), mode="max")


class AvgPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(x, self.pool_size, tuple(self.strides), mode="avg")


class GlobalAvgPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=(1, 2))


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=(1, 2))
