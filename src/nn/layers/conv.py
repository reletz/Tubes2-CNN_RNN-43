from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _compute_padding(in_size: int, kernel: int, stride: int) -> Tuple[int, int]:
    out_size = int(np.ceil(in_size / stride))
    pad_needed = max(0, (out_size - 1) * stride + kernel - in_size)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    return pad_before, pad_after


class Conv2D:
    """Minimal Conv2D (cross-correlation) supporting set_weights().
    Forward accepts input in NHWC format: (N, H, W, C).
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        padding: str = "valid",
        use_bias: bool = True,
    ) -> None:
        self.filters = int(filters)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.strides = (int(strides[0]), int(strides[1]))
        self.padding = padding
        self.use_bias = use_bias

        # weights will be set via set_weights(kernel, bias) where
        # kernel shape is (kH, kW, C_in, filters)
        self.kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def set_weights(self, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.kernel = np.asarray(kernel)
        if bias is not None:
            self.bias = np.asarray(bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D weights are not set. Call set_weights() first.")

        N, H, W, C = x.shape
        kH, kW, Ck, out_filters = self.kernel.shape
        if Ck != C:
            raise ValueError(f"Input channels ({C}) != kernel channels ({Ck})")

        stride_h, stride_w = self.strides

        if self.padding == "same":
            pad_h = _compute_padding(H, kH, stride_h)
            pad_w = _compute_padding(W, kW, stride_w)
            x_padded = np.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
        elif self.padding == "valid":
            x_padded = x
        else:
            raise ValueError("padding must be 'valid' or 'same'")

        H_p, W_p = x_padded.shape[1], x_padded.shape[2]
        out_h = (H_p - kH) // stride_h + 1
        out_w = (W_p - kW) // stride_w + 1

        # extract patches: shape (N, out_h, out_w, kH, kW, C)
        windows = sliding_window_view(x_padded, window_shape=(kH, kW, C), axis=(1, 2, 3))
        # sliding_window_view with three axes creates shape (N, H_p-kH+1, W_p-kW+1, kH, kW, C)
        windows = windows[:, ::stride_h, ::stride_w, ...]

        k_flat = kH * kW * C
        patches = windows.reshape(N, out_h, out_w, k_flat)
        kernel_flat = self.kernel.reshape(k_flat, out_filters)

        # perform batched matmul: (N,out_h,out_w,k_flat) @ (k_flat, out_filters) -> (N,out_h,out_w,out_filters)
        out = np.tensordot(patches, kernel_flat, axes=([3], [0]))

        if self.use_bias:
            if self.bias is None:
                raise ValueError("bias is expected but not set")
            out += self.bias.reshape((1, 1, 1, -1))

        return out


class LocallyConnected2D:
    """Locally connected layer (no weight sharing).

    Expected weight formats supported by set_weights():
      - kernel shape (out_h, out_w, kH, kW, C_in, filters)
      - or flattened (out_h*out_w, kH*kW*C_in, filters)
    Bias may be (out_h, out_w, filters) or (out_h*out_w, filters).
    Forward input is NHWC.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        filters: int,
        strides: Tuple[int, int] = (1, 1),
        padding: str = "valid",
        use_bias: bool = True,
    ) -> None:
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.filters = int(filters)
        self.strides = (int(strides[0]), int(strides[1]))
        self.padding = padding
        self.use_bias = use_bias

        self.kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def set_weights(self, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.kernel = np.asarray(kernel)
        if bias is not None:
            self.bias = np.asarray(bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("LocallyConnected2D weights not set. Call set_weights() first.")

        N, H, W, C = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.strides

        if self.padding == "same":
            pad_h = _compute_padding(H, kH, stride_h)
            pad_w = _compute_padding(W, kW, stride_w)
            x_padded = np.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
        elif self.padding == "valid":
            x_padded = x
        else:
            raise ValueError("padding must be 'valid' or 'same'")

        H_p, W_p = x_padded.shape[1], x_padded.shape[2]
        out_h = (H_p - kH) // stride_h + 1
        out_w = (W_p - kW) // stride_w + 1

        windows = sliding_window_view(x_padded, window_shape=(kH, kW, C), axis=(1, 2, 3))
        windows = windows[:, ::stride_h, ::stride_w, ...]
        k_flat = kH * kW * C
        patches = windows.reshape(N, out_h, out_w, k_flat)

        # normalize kernel shape to (out_h, out_w, k_flat, filters)
        kernel = self.kernel
        if kernel.ndim == 6 and kernel.shape[:2] == (out_h, out_w):
            kernel_flat = kernel.reshape(out_h, out_w, k_flat, self.filters)
        elif kernel.ndim == 3 and kernel.shape[0] == out_h * out_w:
            kernel_flat = kernel.reshape(out_h, out_w, k_flat, self.filters)
        else:
            kernel_flat = np.asarray(kernel)
            try:
                kernel_flat = kernel_flat.reshape(out_h, out_w, k_flat, self.filters)
            except Exception as exc:
                raise ValueError(f"Unexpected kernel shape: {kernel.shape}") from exc

        # einsum over patch dim: (N, out_h, out_w, k) and (out_h, out_w, k, f) -> (N, out_h, out_w, f)
        out = np.einsum("n o p k, o p k f -> n o p f", patches, kernel_flat)

        if self.use_bias:
            if self.bias is None:
                raise ValueError("bias is expected but not set")

            b = self.bias
            if b.ndim == 2 and b.shape[0] == out_h * out_w:
                b = b.reshape(out_h, out_w, self.filters)
            elif b.ndim == 3 and b.shape[:2] == (out_h, out_w):
                b = b
            else:
                try:
                    b = b.reshape(out_h, out_w, self.filters)
                except Exception:
                    raise ValueError(f"Unexpected bias shape: {self.bias.shape}")

            out += b.reshape((1, out_h, out_w, self.filters))

        return out