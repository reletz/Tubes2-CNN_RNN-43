# Normal imports when the package is available
try:
    from ..activations.Activation import Activation
    from ..initializers.Initializer import Initializer
except ImportError:  # pragma: no cover – fallback for direct file loading in tests
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
    Activation = _activation_mod.Activation
    Initializer = _initializer_mod.Initializer

import numpy as np

def get_xp(x):
    if type(x).__module__ == 'cupy':
        import cupy as cp
        return cp
    return np

class Layer:
    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation_fn: Activation,
            init_fn: Initializer
            ) -> None:
        self.weights: np.ndarray = init_fn.initialize((input_size, output_size))
        self.biases: np.ndarray = init_fn.initialize((1, output_size))
        self.weight_gradients: np.ndarray = None
        self.bias_gradients: np.ndarray = None
        self.activation: Activation = activation_fn
        self.input_cache: np.ndarray = None
        self.z_cache: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_cache = X
        z = X @ self.weights + self.biases
        self.z_cache = z
        return self.activation.forward(z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_cache is None:
            raise ValueError("Layer.backward() called before forward().")

        xp = get_xp(grad_output)
        batch_size = self.input_cache.shape[0]

        dz = self.activation.backward(grad_output)

        self.weight_gradients = self.input_cache.T @ dz / batch_size
        self.bias_gradients = xp.mean(dz, axis=0, keepdims=True)

        return dz @ self.weights.T
    
    def set_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
