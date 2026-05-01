import numpy as np

class Optimizer:
    """Base class for optimization methods."""
    def update(self, params, grads):
        """Update parameters given gradients
        
        Args:
            params: List of NumPy arrays (weights and biases)
            grads: List of NumPy arrays with gradients (same structure as params)
            
        Returns:
            Updated parameters (modified in-place or returned)
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

class GradientDescent(Optimizer):
    """Standard gradient descent optimizer.
    
    Update rule: w = w - lr * gradient
    
    Args:
        learning_rate: default -> 0.01
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer
    
    Maintains per-parameter first and second moment estimates.
    
    Args:
        learning_rate: default -> 0.001
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 
        self.m = None  # First moment (mean) of gradients
        self.v = None  # Second moment (uncentered variance) of gradients
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
