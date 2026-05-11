"""High-level mapping helpers: map Keras HDF5 weight dict -> scratch layer instances.

This module provides a small API that accepts a flat mapping of HDF5
dataset paths (as returned by load_h5_weights) and a user-provided
mapping that describes which HDF5 keys belong to which target layer
instances. It uses functions from keras_loader to perform per-layer
assignments.

Example mapping structure:

    mapping = {
        'conv2d': {'layer': conv_layer, 'kind': 'conv'},
        'embedding': {'layer': emb_layer, 'kind': 'embedding'},
        'rnn': {'layer': rnn_cell, 'kind': 'simplernn'},
    }

The keys are substrings matched against HDF5 dataset paths (case-sensitive).
If multiple arrays match a key, all are passed to the corresponding assign
helper in the same order Keras stores them (caller should verify shapes).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import keras_loader


def map_weights_dict_to_layers(weights: Dict[str, np.ndarray], mapping: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map arrays from weights to target layer instances described in mapping.

    Args:
        weights: dict mapping HDF5 dataset path -> ndarray
        mapping: dict where each key is a substring to search in HDF5 paths,
                 and value is dict with keys: layer (target instance) and
                 kind (one of: 'conv','locally','embedding','simplernn','lstm').

    Returns:
        report: dict mapping mapping-key -> status message
    """
    report: Dict[str, str] = {}

    for key_substr, spec in mapping.items():
        if 'layer' not in spec or 'kind' not in spec:
            report[key_substr] = 'invalid mapping spec; requires layer and kind'
            continue

        layer = spec['layer']
        kind = spec['kind']

        arrays = keras_loader.find_layer_weights(weights, key_substr)
        if not arrays:
            report[key_substr] = 'no matching arrays found'
            continue

        try:
            if kind == 'conv':
                keras_loader.assign_conv2d(layer, arrays)
            elif kind == 'locally':
                keras_loader.assign_locally_connected(layer, arrays)
            elif kind == 'embedding':
                keras_loader.assign_embedding(layer, arrays)
            elif kind == 'simplernn':
                keras_loader.assign_simplernn(layer, arrays)
            elif kind == 'lstm':
                keras_loader.assign_lstm(layer, arrays)
            else:
                report[key_substr] = f"unknown kind '{kind}'"
                continue

            report[key_substr] = f'assigned ({len(arrays)} arrays)'
        except Exception as e:
            report[key_substr] = f'error: {e}'

    return report


def load_h5_and_map(h5_path: str, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Load an HDF5 weight file and map arrays to layers using mapping.

    This is a convenience wrapper around load_h5_weights + map_weights_dict_to_layers.
    """
    weights = keras_loader.load_h5_weights(h5_path)
    return map_weights_dict_to_layers(weights, mapping)


def detect_layer_groups(weights: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """Group flat HDF5 weight dict by top-level group (layer name).

    Returns a mapping group_name -> {var_name: array}
    where var_name is the remainder after the first '/'.
    """
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for path, arr in weights.items():
        if '/' in path:
            group, var = path.split('/', 1)
        else:
            # fallback: put under root
            group, var = '', path

        groups.setdefault(group, {})[var] = arr

    return groups


def guess_layer_kind_from_group(group_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Heuristically guess layer kind from its arrays and shapes.

    Returns a dict with keys: kind (str) and info (details like shapes).
    Possible kinds: 'conv','locally','embedding','simplernn','lstm','dense','unknown'
    """
    info: Dict[str, Any] = {"vars": {k: v.shape for k, v in group_arrays.items()}}

    shapes = list(info["vars"].values())
    # embedding: single 2D array
    if len(group_arrays) == 1:
        s = shapes[0]
        if len(s) == 2:
            return {"kind": "embedding", "info": info}

    # dense: kernel (in,out) and bias (out,)
    if any('kernel' in k and arr.ndim == 2 for k, arr in group_arrays.items()):
        # check for LSTM vs Dense by bias length vs 4*units
        kernels = [arr for k, arr in group_arrays.items() if 'kernel' in k]
        biases = [arr for k, arr in group_arrays.items() if 'bias' in k]
        if kernels:
            k0 = kernels[0]
            if k0.ndim == 2:
                # possible dense or rnn/lstm kernel
                if biases:
                    b0 = biases[0]
                    if b0.ndim == 1:
                        # LSTM kernel usually has second dim divisible by 4
                        if k0.shape[1] % 4 == 0 or (b0.shape[0] % 4 == 0 and b0.shape[0] == k0.shape[1]):
                            return {"kind": "lstm", "info": info}
                        # simple RNN kernel shape (in, units) with bias (units,)
                        if b0.shape[0] == k0.shape[1]:
                            return {"kind": "simplernn", "info": info}
                        # dense fallback
                        return {"kind": "dense", "info": info}

    # conv: kernel shape (kH,kW,in,out)
    for k, arr in group_arrays.items():
        if arr.ndim == 4:
            # assume conv2d
            return {"kind": "conv", "info": info}

    # locally connected: 6D or flattened 3D
    for k, arr in group_arrays.items():
        if arr.ndim == 6:
            return {"kind": "locally", "info": info}
        if arr.ndim == 3:
            # potential flattened LC kernel: (out_h*out_w, kH*kW*C_in, filters)
            if arr.shape[1] > arr.shape[2]:
                return {"kind": "locally", "info": info}

    return {"kind": "unknown", "info": info}


def suggest_mapping(weights: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Detect groups and suggest a kind for each group.

    Returns mapping group -> {'kind': kind, 'vars': {var: shape}}
    """
    groups = detect_layer_groups(weights)
    suggestions: Dict[str, Dict[str, Any]] = {}
    for grp, arrays in groups.items():
        guess = guess_layer_kind_from_group(arrays)
        suggestions[grp] = {"kind": guess["kind"], "vars": guess["info"]["vars"]}
    return suggestions
