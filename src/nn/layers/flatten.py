from __future__ import annotations

import numpy as np


class Flatten:
	"""Flatten layer.

	Converts NHWC -> (N, H*W*C). Keeps input shape to support backward().
	"""

	def __init__(self) -> None:
		self.input_shape: tuple[int, ...] | None = None

	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim == 3:
			# single example H,W,C -> (1, H*W*C)
			self.input_shape = x.shape
			return x.reshape(1, -1)
		if x.ndim == 4:
			self.input_shape = x.shape
			N = x.shape[0]
			return x.reshape(N, -1)

		raise ValueError("Flatten.forward expects input with 3 or 4 dimensions (H,W,C) or (N,H,W,C)")

	def backward(self, grad: np.ndarray) -> np.ndarray:
		if self.input_shape is None:
			raise ValueError("Cannot call backward() before forward()")

		if grad.ndim != 2:
			raise ValueError("Gradient to Flatten.backward must be 2D (N, features)")

		N = grad.shape[0]
		if len(self.input_shape) == 3:
			H, W, C = self.input_shape
			if N != 1:
				raise ValueError("Batch size mismatch between grad and stored input shape")
			return grad.reshape((H, W, C))

		# input_shape is (N,H,W,C)
		return grad.reshape(self.input_shape)

