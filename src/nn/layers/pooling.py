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

    # sliding_window_view returns shape (N, out_h, out_w, C, kH, kW)
    # reduce over the last two axes (kH, kW) to preserve channel axis
    if mode == "max":
        return np.max(windows, axis=(-2, -1))
    elif mode == "avg":
        return np.mean(windows, axis=(-2, -1))
    else:
        raise ValueError("mode must be 'max' or 'avg'")


class MaxPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size
        self._input_cache: np.ndarray | None = None
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = np.asarray(x)
        out = _pool2d(x, self.pool_size, tuple(self.strides), mode="max")
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("MaxPool2D.backward() called before forward().")

        x = self._input_cache
        out = self._output_cache
        grad_output = np.asarray(grad_output, dtype=np.float32)

        if grad_output.shape != out.shape:
            raise ValueError(f"Expected grad_output shape {out.shape}, got {grad_output.shape}")

        N, H, W, C = x.shape
        kH, kW = self.pool_size
        sH, sW = tuple(self.strides)
        out_h, out_w = out.shape[1], out.shape[2]

        grad_input = np.zeros_like(x, dtype=np.float32)

        for out_i in range(out_h):
            h_start = out_i * sH
            h_slice = slice(h_start, h_start + kH)
            for out_j in range(out_w):
                w_start = out_j * sW
                w_slice = slice(w_start, w_start + kW)

                region = x[:, h_slice, w_slice, :]
                pooled = out[:, out_i, out_j, :][:, None, None, :]
                mask = region == pooled
                mask_count = mask.sum(axis=(1, 2), keepdims=True)
                mask_count = np.where(mask_count == 0, 1.0, mask_count)

                grad_region = mask * (grad_output[:, out_i, out_j, :][:, None, None, :] / mask_count)
                grad_input[:, h_slice, w_slice, :] += grad_region

        return grad_input


class AvgPool2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] | None = None):
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.strides = strides if strides is not None else self.pool_size
        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = np.asarray(x)
        return _pool2d(x, self.pool_size, tuple(self.strides), mode="avg")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise ValueError("AvgPool2D.backward() called before forward().")

        x = self._input_cache
        grad_output = np.asarray(grad_output, dtype=np.float32)
        N, H, W, C = x.shape
        kH, kW = self.pool_size
        sH, sW = tuple(self.strides)
        out_h = (H - kH) // sH + 1
        out_w = (W - kW) // sW + 1

        if grad_output.shape != (N, out_h, out_w, C):
            raise ValueError(f"Expected grad_output shape {(N, out_h, out_w, C)}, got {grad_output.shape}")

        grad_input = np.zeros_like(x, dtype=np.float32)
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
        self._input_cache = np.asarray(x)
        return np.mean(x, axis=(1, 2))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise ValueError("GlobalAvgPooling2D.backward() called before forward().")

        x = self._input_cache
        grad_output = np.asarray(grad_output, dtype=np.float32)
        N, H, W, C = x.shape
        if grad_output.shape != (N, C):
            raise ValueError(f"Expected grad_output shape {(N, C)}, got {grad_output.shape}")

        return np.broadcast_to(grad_output[:, None, None, :] / (H * W), (N, H, W, C)).copy()


class GlobalMaxPooling2D:
    def __init__(self) -> None:
        self._input_cache: np.ndarray | None = None
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = np.asarray(x)
        out = np.max(x, axis=(1, 2))
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("GlobalMaxPooling2D.backward() called before forward().")

        x = self._input_cache
        out = self._output_cache
        grad_output = np.asarray(grad_output, dtype=np.float32)
        N, H, W, C = x.shape
        if grad_output.shape != (N, C):
            raise ValueError(f"Expected grad_output shape {(N, C)}, got {grad_output.shape}")

        grad_input = np.zeros_like(x, dtype=np.float32)
        max_mask = x == out[:, None, None, :]
        counts = max_mask.sum(axis=(1, 2), keepdims=True)
        counts = np.where(counts == 0, 1.0, counts)
        grad_input += max_mask * (grad_output[:, None, None, :] / counts)
        return grad_input
