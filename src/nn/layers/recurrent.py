from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SimpleRNNCell:
    """Simple RNN cell with tanh activation.

    Keras weight format:
      - kernel: (input_dim, units)
      - recurrent_kernel: (units, units)
      - bias: (units,)
    """

    def __init__(self, units: int) -> None:
        self.units = int(units)
        self.kernel: np.ndarray | None = None
        self.recurrent_kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def set_weights(
        self,
        kernel: np.ndarray,
        recurrent_kernel: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = np.asarray(recurrent_kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    def get_initial_state(self, batch_size: int) -> np.ndarray:
        return np.zeros((int(batch_size), self.units), dtype=np.float32)

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("SimpleRNNCell weights are not set. Call set_weights() first.")

        x = np.asarray(x, dtype=np.float32)
        h_prev = np.asarray(h_prev, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected x shape (batch, input_dim), got {x.shape}")
        if h_prev.ndim != 2:
            raise ValueError(f"Expected h_prev shape (batch, units), got {h_prev.shape}")
        if x.shape[0] != h_prev.shape[0]:
            raise ValueError("Batch size mismatch between x and h_prev")
        if h_prev.shape[1] != self.units:
            raise ValueError(f"Expected h_prev second dim {self.units}, got {h_prev.shape[1]}")

        z = x @ self.kernel + h_prev @ self.recurrent_kernel + self.bias
        return np.tanh(z)


class LSTMCell:
    """LSTM cell with Keras-compatible weights.

    Keras weight format:
      - kernel: (input_dim, 4 * units)
      - recurrent_kernel: (units, 4 * units)
      - bias: (4 * units,)

    Gate order follows Keras: [input, forget, cell, output].
    """

    def __init__(self, units: int) -> None:
        self.units = int(units)
        self.kernel: np.ndarray | None = None
        self.recurrent_kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.kernel_gradients: np.ndarray | None = None
        self.recurrent_kernel_gradients: np.ndarray | None = None
        self.bias_gradients: np.ndarray | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def set_weights(
        self,
        kernel: np.ndarray,
        recurrent_kernel: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = np.asarray(recurrent_kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    def get_initial_state(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        h0 = np.zeros((batch_size, self.units), dtype=np.float32)
        c0 = np.zeros((batch_size, self.units), dtype=np.float32)
        return h0, c0

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("LSTMCell weights are not set. Call set_weights() first.")

        x = np.asarray(x, dtype=np.float32)
        h_prev = np.asarray(h_prev, dtype=np.float32)
        c_prev = np.asarray(c_prev, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected x shape (batch, input_dim), got {x.shape}")
        if h_prev.ndim != 2 or c_prev.ndim != 2:
            raise ValueError("Expected h_prev and c_prev to be 2D arrays")
        if x.shape[0] != h_prev.shape[0] or x.shape[0] != c_prev.shape[0]:
            raise ValueError("Batch size mismatch among x, h_prev, and c_prev")
        if h_prev.shape[1] != self.units or c_prev.shape[1] != self.units:
            raise ValueError(f"Expected hidden/cell state second dim {self.units}")

        z = x @ self.kernel + h_prev @ self.recurrent_kernel + self.bias
        i, f, g, o = np.split(z, 4, axis=1)

        i = _sigmoid(i)
        f = _sigmoid(f)
        g = np.tanh(g)
        o = _sigmoid(o)

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        self._cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "i": i,
            "f": f,
            "g": g,
            "o": o,
            "c": c,
        }
        return h, c

    def backward(
        self,
        grad_h: np.ndarray,
        grad_c: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("LSTMCell weights are not set. Call set_weights() first.")
        if self._cache is None:
            raise ValueError("LSTMCell.backward() called before forward().")

        grad_h = np.asarray(grad_h, dtype=np.float32)
        if grad_c is None:
            grad_c = np.zeros_like(grad_h, dtype=np.float32)
        else:
            grad_c = np.asarray(grad_c, dtype=np.float32)

        x = self._cache["x"]
        h_prev = self._cache["h_prev"]
        c_prev = self._cache["c_prev"]
        i = self._cache["i"]
        f = self._cache["f"]
        g = self._cache["g"]
        o = self._cache["o"]
        c = self._cache["c"]

        tanh_c = np.tanh(c)
        do = grad_h * tanh_c
        dc = grad_h * o * (1.0 - tanh_c**2) + grad_c
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f

        di_pre = di * i * (1.0 - i)
        df_pre = df * f * (1.0 - f)
        dg_pre = dg * (1.0 - g**2)
        do_pre = do * o * (1.0 - o)

        dz = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)

        self.kernel_gradients = x.T @ dz
        self.recurrent_kernel_gradients = h_prev.T @ dz
        self.bias_gradients = dz.sum(axis=0)

        dx = dz @ self.kernel.T
        dh_prev = dz @ self.recurrent_kernel.T
        return dx, dh_prev, dc_prev
