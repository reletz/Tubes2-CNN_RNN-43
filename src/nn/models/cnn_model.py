from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Normal package imports – used when the module is imported as part of the
# project package (e.g., during normal execution).
# ---------------------------------------------------------------------------
try:
	from ..activations.Activation import Linear, Softmax
	from ..initializers.Initializer import Xavier, Zero
	from ..layers.conv import Conv2D, LocallyConnected2D
	from ..layers.dense import Layer as DenseLayer
	from ..layers.flatten import Flatten
	from ..layers.pooling import AvgPool2D, GlobalAvgPooling2D, GlobalMaxPooling2D, MaxPool2D
except ImportError:  # pragma: no cover – fallback for direct file loading in tests
	# When the test suite loads this file via a raw path (not as a package),
	# relative imports fail. We fall back to loading the sibling modules by
	# absolute file location using importlib.
	import importlib.util
	from pathlib import Path

	def _load_module(module_name: str, relative_path: str):
		module_path = Path(__file__).resolve().parents[1] / relative_path
		spec = importlib.util.spec_from_file_location(module_name, str(module_path))
		mod = importlib.util.module_from_spec(spec)
		assert spec is not None and spec.loader is not None
		spec.loader.exec_module(mod)
		return mod

	_activation_mod = _load_module("activation_fallback", "activations/Activation.py")
	_initializer_mod = _load_module("initializer_fallback", "initializers/Initializer.py")
	_conv_mod = _load_module("conv_fallback", "layers/conv.py")
	_dense_mod = _load_module("dense_fallback", "layers/dense.py")
	_flatten_mod = _load_module("flatten_fallback", "layers/flatten.py")
	_pool_mod = _load_module("pool_fallback", "layers/pooling.py")

	Linear = _activation_mod.Linear
	Softmax = _activation_mod.Softmax
	Xavier = _initializer_mod.Xavier
	Zero = _initializer_mod.Zero
	Conv2D = _conv_mod.Conv2D
	LocallyConnected2D = _conv_mod.LocallyConnected2D
	DenseLayer = _dense_mod.Layer
	Flatten = _flatten_mod.Flatten
	AvgPool2D = _pool_mod.AvgPool2D
	GlobalAvgPooling2D = _pool_mod.GlobalAvgPooling2D
	GlobalMaxPooling2D = _pool_mod.GlobalMaxPooling2D
	MaxPool2D = _pool_mod.MaxPool2D


ConvKind = Literal["conv", "locally"]
PoolKind = Literal["max", "avg"]
HeadKind = Literal["flatten", "global_avg", "global_max"]


class ConvBlockSpec:
	"""Immutable configuration for one convolutional block.

	We avoid ``@dataclass`` because loading this module via a raw file path
	(as the test suite does) results in ``__module__`` being ``None`` which
	breaks the dataclass machinery. A simple class with read‑only attributes
	provides the same API without the import‑time error.
	"""

	def __init__(
		self,
		filters: int,
		kernel_size: tuple[int, int],
		padding: str = "valid",
		strides: tuple[int, int] = (1, 1),
		pool_size: tuple[int, int] = (2, 2),
		pool_strides: tuple[int, int] | None = None,
		use_pool: bool = True,
	) -> None:
		self.filters = int(filters)
		self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
		self.padding = padding
		self.strides = (int(strides[0]), int(strides[1]))
		self.pool_size = (int(pool_size[0]), int(pool_size[1]))
		self.pool_strides = None if pool_strides is None else (int(pool_strides[0]), int(pool_strides[1]))
		self.use_pool = bool(use_pool)


def _conv_out_dim(size: int, kernel: int, stride: int, padding: str) -> int:
	if padding == "same":
		return int(np.ceil(size / stride))
	if padding == "valid":
		return (size - kernel) // stride + 1
	raise ValueError("padding must be 'valid' or 'same'")


def _pool_out_dim(size: int, pool: int, stride: int) -> int:
	return (size - pool) // stride + 1


class CNNClassifier:
	"""Sequential CNN classifier builder.

	The model is composed from scratch layers and supports two convolution
	variants: shared-parameter Conv2D and non-shared LocallyConnected2D.

	It is intentionally lightweight and focused on forward inference + weight
	injection for comparing against Keras models.
	"""

	def __init__(
		self,
		input_shape: tuple[int, int, int],
		num_classes: int,
		conv_blocks: Sequence[ConvBlockSpec] | None = None,
		conv_kind: ConvKind = "conv",
		pool_kind: PoolKind = "max",
		head_kind: HeadKind = "flatten",
	) -> None:
		if len(input_shape) != 3:
			raise ValueError("input_shape must be (H, W, C)")
		if num_classes <= 0:
			raise ValueError("num_classes must be positive")

		self.input_shape = tuple(int(v) for v in input_shape)
		self.num_classes = int(num_classes)
		self.conv_kind = conv_kind
		self.pool_kind = pool_kind
		self.head_kind = head_kind

		if conv_blocks is None:
			conv_blocks = (
				ConvBlockSpec(filters=32, kernel_size=(3, 3), padding="same"),
				ConvBlockSpec(filters=64, kernel_size=(3, 3), padding="same"),
			)
		self.conv_blocks = list(conv_blocks)
		if not self.conv_blocks:
			raise ValueError("conv_blocks must contain at least one block")

		self.conv_layers: list[Any] = []
		self.pool_layers: list[Any] = []
		self.head_layer: Any
		self.output_layer: Any

		self._build_layers()

	def _build_layers(self) -> None:
		h, w, c = self.input_shape
		self.conv_layers = []
		self.pool_layers = []

		for block in self.conv_blocks:
			if self.conv_kind == "conv":
				conv_layer = Conv2D(
					filters=block.filters,
					kernel_size=block.kernel_size,
					strides=block.strides,
					padding=block.padding,
					use_bias=True,
				)
			elif self.conv_kind == "locally":
				conv_layer = LocallyConnected2D(
					filters=block.filters,
					kernel_size=block.kernel_size,
					strides=block.strides,
					padding=block.padding,
					use_bias=True,
				)
			else:
				raise ValueError("conv_kind must be 'conv' or 'locally'")

			self.conv_layers.append(conv_layer)

			h = _conv_out_dim(h, block.kernel_size[0], block.strides[0], block.padding)
			w = _conv_out_dim(w, block.kernel_size[1], block.strides[1], block.padding)
			c = block.filters

			if block.use_pool:
				pool_stride = block.pool_strides or block.pool_size
				if self.pool_kind == "max":
					pool_layer = MaxPool2D(pool_size=block.pool_size, strides=pool_stride)
				elif self.pool_kind == "avg":
					pool_layer = AvgPool2D(pool_size=block.pool_size, strides=pool_stride)
				else:
					raise ValueError("pool_kind must be 'max' or 'avg'")

				self.pool_layers.append(pool_layer)
				h = _pool_out_dim(h, block.pool_size[0], pool_stride[0])
				w = _pool_out_dim(w, block.pool_size[1], pool_stride[1])
			else:
				self.pool_layers.append(None)  # type: ignore[arg-type]

		if self.head_kind == "flatten":
			self.head_layer = Flatten()
			head_dim = h * w * c
		elif self.head_kind == "global_avg":
			self.head_layer = GlobalAvgPooling2D()
			head_dim = c
		elif self.head_kind == "global_max":
			self.head_layer = GlobalMaxPooling2D()
			head_dim = c
		else:
			raise ValueError("head_kind must be 'flatten', 'global_avg', or 'global_max'")

		self.output_layer = DenseLayer(head_dim, self.num_classes, Softmax(), Xavier())

	def set_conv_weights(self, block_index: int, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
		"""Set weights for one convolutional block."""
		self.conv_layers[block_index].set_weights(kernel, bias)

	def set_output_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
		"""Set weights for the final classifier layer."""
		self.output_layer.set_weights(weights, biases.reshape(1, -1))

	def forward(self, x: np.ndarray) -> np.ndarray:
		"""Forward pass.

		Args:
			x: Input image batch in NHWC format `(N, H, W, C)` or a single
			   image `(H, W, C)`.

		Returns:
			Class probabilities with shape `(N, num_classes)`.
		"""
		x = np.asarray(x, dtype=np.float32)
		if x.ndim == 3:
			x = x[None, ...]
		if x.ndim != 4:
			raise ValueError(f"Expected input shape (N,H,W,C) or (H,W,C), got {x.shape}")

		out = x
		for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
			out = conv_layer.forward(out)
			if pool_layer is not None:
				out = pool_layer.forward(out)

		out = self.head_layer.forward(out)
		out = self.output_layer.forward(out)
		return out

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Alias for forward()."""
		return self.forward(x)

