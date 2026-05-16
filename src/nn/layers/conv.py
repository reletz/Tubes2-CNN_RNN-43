from __future__ import annotations

from typing import Iterable, Tuple

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


def _compute_padding(in_size: int, kernel: int, stride: int, xp=np) -> Tuple[int, int]:
    out_size = int(xp.ceil(in_size / stride))
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
        self.kernel_gradients: np.ndarray | None = None
        self.bias_gradients: np.ndarray | None = None
        self._input_cache: np.ndarray | None = None
        self._padded_input_cache: np.ndarray | None = None
        self._pad_h: Tuple[int, int] | None = None
        self._pad_w: Tuple[int, int] | None = None
        self._output_shape: Tuple[int, int, int, int] | None = None

    def set_weights(self, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.kernel = np.asarray(kernel)
        if bias is not None:
            self.bias = np.asarray(bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D weights are not set. Call set_weights() first.")

        xp = get_xp(x)
        self._input_cache = x

        N, H, W, C = x.shape
        kH, kW, Ck, out_filters = self.kernel.shape
        if Ck != C:
            raise ValueError(f"Input channels ({C}) != kernel channels ({Ck})")

        stride_h, stride_w = self.strides

        if self.padding == "same":
            pad_h = _compute_padding(H, kH, stride_h, xp)
            pad_w = _compute_padding(W, kW, stride_w, xp)
            x_padded = xp.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
        elif self.padding == "valid":
            pad_h = (0, 0)
            pad_w = (0, 0)
            x_padded = x
        else:
            raise ValueError("padding must be 'valid' or 'same'")

        self._padded_input_cache = x_padded
        self._pad_h = pad_h
        self._pad_w = pad_w

        H_p, W_p = x_padded.shape[1], x_padded.shape[2]
        out_h = (H_p - kH) // stride_h + 1
        out_w = (W_p - kW) // stride_w + 1

        windows = _sliding_window_view(x_padded, window_shape=(kH, kW, C), axis=(1, 2, 3), xp=xp)
        windows = windows[:, ::stride_h, ::stride_w, ...]

        k_flat = kH * kW * C
        patches = windows.reshape(N, out_h, out_w, k_flat)
        kernel_flat = self.kernel.reshape(k_flat, out_filters)

        out = xp.tensordot(patches, kernel_flat, axes=([3], [0]))

        if self.use_bias:
            if self.bias is None:
                raise ValueError("bias is expected but not set")
            out += self.bias.reshape((1, 1, 1, -1))

        self._output_shape = out.shape

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D weights are not set. Call set_weights() first.")
        if self._padded_input_cache is None or self._output_shape is None:
            raise ValueError("Conv2D.backward() called before forward().")

        xp = get_xp(grad_output)
        
        if grad_output.shape != self._output_shape:
            raise ValueError(f"Expected grad_output shape {self._output_shape}, got {grad_output.shape}")

        x_padded = self._padded_input_cache
        N, H_p, W_p, C = x_padded.shape
        kH, kW, Ck, out_filters = self.kernel.shape
        stride_h, stride_w = self.strides
        out_h, out_w = self._output_shape[1], self._output_shape[2]
        k_flat = kH * kW * C

        kernel_flat = self.kernel.reshape(k_flat, out_filters)
        grad_kernel_flat = xp.zeros_like(kernel_flat, dtype=xp.float32)
        grad_input_padded = xp.zeros_like(x_padded, dtype=xp.float32)

        for out_i in range(out_h):
            h_start = out_i * stride_h
            h_slice = slice(h_start, h_start + kH)
            for out_j in range(out_w):
                w_start = out_j * stride_w
                w_slice = slice(w_start, w_start + kW)

                patch = x_padded[:, h_slice, w_slice, :].reshape(N, k_flat)
                grad_slice = grad_output[:, out_i, out_j, :]

                grad_kernel_flat += patch.T @ grad_slice
                grad_patch = grad_slice @ kernel_flat.T
                grad_input_padded[:, h_slice, w_slice, :] += grad_patch.reshape(N, kH, kW, C)

        self.kernel_gradients = grad_kernel_flat.reshape(self.kernel.shape)
        if self.use_bias:
            self.bias_gradients = grad_output.sum(axis=(0, 1, 2))
        else:
            self.bias_gradients = None

        if self.padding == "same":
            pad_top, pad_bottom = self._pad_h or (0, 0)
            pad_left, pad_right = self._pad_w or (0, 0)
            return grad_input_padded[:, pad_top : H_p - pad_bottom, pad_left : W_p - pad_right, :]

        return grad_input_padded


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

        xp = get_xp(x)
        N, H, W, C = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.strides

        if self.padding == "same":
            pad_h = _compute_padding(H, kH, stride_h, xp)
            pad_w = _compute_padding(W, kW, stride_w, xp)
            x_padded = xp.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
        elif self.padding == "valid":
            x_padded = x
        else:
            raise ValueError("padding must be 'valid' or 'same'")

        H_p, W_p = x_padded.shape[1], x_padded.shape[2]
        out_h = (H_p - kH) // stride_h + 1
        out_w = (W_p - kW) // stride_w + 1

        windows = _sliding_window_view(x_padded, window_shape=(kH, kW, C), axis=(1, 2, 3), xp=xp)
        windows = windows[:, ::stride_h, ::stride_w, ...]
        k_flat = kH * kW * C
        patches = windows.reshape(N, out_h, out_w, k_flat)

        kernel = self.kernel
        if kernel.size == out_h * out_w * k_flat * self.filters:
            kernel_flat = kernel.reshape(out_h, out_w, k_flat, self.filters)
        elif kernel.size == k_flat * self.filters:
            kernel_shared_flat = kernel.reshape(k_flat, self.filters)
            kernel_flat = xp.broadcast_to(kernel_shared_flat, (out_h, out_w, k_flat, self.filters))
        else:
            kernel_flat = xp.asarray(kernel)
            kernel_flat = kernel_flat.reshape(out_h, out_w, k_flat, self.filters)

        # Convert einsum to batched matmul
        # patches: (N, out_h, out_w, k_flat) -> transpose to (out_h, out_w, N, k_flat)
        # kernel_flat: (out_h, out_w, k_flat, filters)
        patches_reshaped = patches.transpose(1, 2, 0, 3) 
        out = xp.matmul(patches_reshaped, kernel_flat) 
        # out: (out_h, out_w, N, filters) -> transpose back to (N, out_h, out_w, filters)
        out = out.transpose(2, 0, 1, 3)

        if self.use_bias:
            if self.bias is None:
                raise ValueError("bias is expected but not set")

            b = self.bias
            if b.size == out_h * out_w * self.filters:
                b = b.reshape(out_h, out_w, self.filters)
            elif b.size == self.filters:
                b = xp.broadcast_to(b.reshape(self.filters), (out_h, out_w, self.filters))
            else:
                b = b.reshape(out_h, out_w, self.filters)

            out += b.reshape((1, out_h, out_w, self.filters))

        return out