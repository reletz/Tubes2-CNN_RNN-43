from __future__ import annotations

from typing import Tuple

import numpy as np

def get_xp(x):
    if type(x).__module__ == 'cupy':
        import cupy as cp
        return cp
    return np

def _sliding_window_view(x, window_shape, axis, xp):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(x, window_shape, axis)
    else:
        from cupy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(x, window_shape, axis)


def _pool2d(x: np.ndarray, pool_size: Tuple[int, int], strides: Tuple[int, int], mode: str):
    xp = get_xp(x)
    N, H, W, C = x.shape
    kH, kW = pool_size
    sH, sW = strides

    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1

    windows = _sliding_window_view(x, window_shape=(kH, kW), axis=(1, 2), xp=xp)
    windows = windows[:, ::sH, ::sW, ...] 

    if mode == "max":
        return xp.max(windows, axis=(-2, -1))
    elif mode == "avg":
        return xp.mean(windows, axis=(-2, -1))
    else:
        raise ValueError("mode must be 'max' or 'avg'")


class MaxPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size
        self._input_cache: np.ndarray | None = None
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        out = _pool2d(x, self.pool_size, tuple(self.strides), mode="max")
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("MaxPool2D.backward() called before forward().")

        x = self._input_cache
        xp = get_xp(grad_output)
        N, H, W, C = self._input_cache.shape
        kH, kW = self.pool_size
        sH, sW = self.strides

        grad_input = xp.zeros_like(self._input_cache, dtype=xp.float32)

        for i in range(grad_output.shape[1]):
            for j in range(grad_output.shape[2]):
                h_start, w_start = i * sH, j * sW
                h_end, w_end = h_start + kH, w_start + kW
                
                x_slice = self._input_cache[:, h_start:h_end, w_start:w_end, :]
                mask = (x_slice == xp.max(x_slice, axis=(1, 2), keepdims=True))
                
                grad_slice = grad_output[:, i:i+1, j:j+1, :]
                grad_input[:, h_start:h_end, w_start:w_end, :] += mask * grad_slice

        return grad_input


class AvgPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size
        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return _pool2d(x, self.pool_size, tuple(self.strides), mode="avg")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise ValueError("AvgPool2D.backward() called before forward().")

        xp = get_xp(grad_output)
        x = self._input_cache
        N, H, W, C = x.shape
        kH, kW = self.pool_size
        sH, sW = tuple(self.strides)
        out_h = (H - kH) // sH + 1
        out_w = (W - kW) // sW + 1

        grad_input = xp.zeros_like(x, dtype=xp.float32)
        scale = 1.0 / (kH * kW)

        for out_i in range(out_h):
            h_start = out_i * sH
            h_slice = slice(h_start, h_start + kH)
            for out_j in range(out_w):
                w_start = out_j * sW
                w_slice = slice(w_start, w_start + kW)
                grad_input[:, h_slice, w_slice, :] += grad_output[:, out_i, out_j, :][:, None, None, :] * scale

        return grad_input


class GlobalAvgPooling2D:
    def __init__(self) -> None:
        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = get_xp(x)
        self._input_cache = x
        return xp.mean(x, axis=(1, 2))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise ValueError("GlobalAvgPooling2D.backward() called before forward().")

        xp = get_xp(grad_output)
        x = self._input_cache
        N, H, W, C = x.shape

        return xp.broadcast_to(grad_output[:, None, None, :] / (H * W), (N, H, W, C)).copy()


class GlobalMaxPooling2D:
    def __init__(self) -> None:
        self._input_cache: np.ndarray | None = None
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = get_xp(x)
        self._input_cache = x
        out = xp.max(x, axis=(1, 2))
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("GlobalMaxPooling2D.backward() called before forward().")

        xp = get_xp(grad_output)
        x = self._input_cache
        out = self._output_cache
        N, H, W, C = x.shape

        grad_input = xp.zeros_like(x, dtype=xp.float32)
        max_mask = x == out[:, None, None, :]
        counts = max_mask.sum(axis=(1, 2), keepdims=True)
        counts = xp.where(counts == 0, 1.0, counts)
        grad_input += max_mask * (grad_output[:, None, None, :] / counts)
        return grad_input
