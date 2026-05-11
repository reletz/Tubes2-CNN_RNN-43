from ..activations.Activation import Activation
from ..initializers.Initializer import Initializer
import numpy as np

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

        if self.rmsnorm is not None:
            z = self.rmsnorm.forward(z)

        return self.activation.forward(z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_cache is None:
            raise ValueError("Layer.backward() called before forward().")

        batch_size = self.input_cache.shape[0]

        dz = self.activation.backward(grad_output)

        if self.rmsnorm is not None:
            dz = self.rmsnorm.backward(dz)

        self.weight_gradients = self.input_cache.T @ dz / batch_size
        self.bias_gradients = np.mean(dz, axis=0, keepdims=True)

        return dz @ self.weights.T
    
    def set_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
