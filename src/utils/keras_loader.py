"""Helpers to load Keras weight files and map them to scratch layers.

This module provides utilities to read HDF5 weight files created by
model.save_weights(...) and convenience functions to assign weight arrays
to our layer objects which expect Keras-style array shapes.

Usage:
    from src.utils.keras_loader import load_h5_weights, find_layer_weights
    weights = load_h5_weights('model_weights.h5')
    # inspect keys and pick arrays for a specific layer
    arrs = find_layer_weights(weights, 'conv2d')
    assign_conv2d(conv_layer, arrs)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import h5py
except Exception:
    h5py = None


def load_h5_weights(h5_path: str) -> Dict[str, np.ndarray]:
    """Load all datasets from an HDF5 weights file into a flat dict.

    The returned mapping keys are the HDF5 dataset full paths (without leading
    slash). This is intentionally generic so callers can inspect keys and
    choose the arrays that correspond to layers they want to load.
    """
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 weight files")

    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name.lstrip("/")] = np.asarray(obj)

        f.visititems(visit)

    return out


def find_layer_weights(weights: Dict[str, np.ndarray], layer_name: str) -> List[np.ndarray]:
    """Return a list of arrays whose HDF5 path contains layer_name.

    The ordering is arbitrary; callers should verify shapes. This helper is
    for convenience during manual mapping of Keras layers to scratch layers.
    """
    res = []
    for k, v in weights.items():
        if layer_name in k:
            res.append(v)
    return res


def assign_conv2d(layer, arrays: List[np.ndarray]) -> None:
    """Assign Conv2D weights (kernel [, bias]) to a Conv2D layer.

    Keras ordering is [kernel, bias] when use_bias=True.
    """
    if not arrays:
        raise ValueError("No arrays provided for Conv2D assignment")

    kernel = arrays[0]
    bias = None
    if len(arrays) > 1:
        bias = arrays[1]

    if bias is None:
        # if layer has attribute 'filters' try to construct zeros
        default_bias = np.zeros(getattr(layer, 'filters', 0), dtype=np.float32)
        layer.set_weights(kernel.astype(np.float32), default_bias)
    else:
        layer.set_weights(kernel.astype(np.float32), bias.astype(np.float32))


def assign_locally_connected(layer, arrays: List[np.ndarray]) -> None:
    """Assign LocallyConnected2D weights to layer.

    Keras often stores weights as [kernel, bias] where kernel shape is
    (out_h, out_w, kH, kW, in_ch, filters). We accept either that shape
    or a flattened (out_h*out_w, kH*kW*in_ch, filters).
    """
    if not arrays:
        raise ValueError("No arrays provided for LocallyConnected2D assignment")

    kernel = arrays[0]
    bias = arrays[1] if len(arrays) > 1 else None

    layer.set_weights(kernel.astype(np.float32), bias.astype(np.float32) if bias is not None else None)


def assign_embedding(layer, arrays: List[np.ndarray]) -> None:
    """Assign embedding matrix to Embedding layer.

    Expect single array shape (vocab_size, embed_dim).
    """
    if len(arrays) != 1:
        raise ValueError("Embedding assignment expects exactly one array")
    layer.set_weights(arrays[0].astype(np.float32))


def assign_simplernn(layer, arrays: List[np.ndarray]) -> None:
    """Assign weights to SimpleRNNCell.

    Keras ordering: [kernel, recurrent_kernel, bias].
    """
    if len(arrays) != 3:
        raise ValueError("SimpleRNN assignment expects 3 arrays: kernel, recurrent_kernel, bias")
    layer.set_weights(arrays[0].astype(np.float32), arrays[1].astype(np.float32), arrays[2].astype(np.float32))


def assign_lstm(layer, arrays: List[np.ndarray]) -> None:
    """Assign weights to LSTMCell.

    Keras ordering: [kernel, recurrent_kernel, bias] where kernel shape is
    (input_dim, 4*units) and recurrent_kernel (units, 4*units).
    """
    if len(arrays) != 3:
        raise ValueError("LSTM assignment expects 3 arrays: kernel, recurrent_kernel, bias")
    layer.set_weights(arrays[0].astype(np.float32), arrays[1].astype(np.float32), arrays[2].astype(np.float32))
