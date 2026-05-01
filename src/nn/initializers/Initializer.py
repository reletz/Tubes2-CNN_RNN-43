import numpy as np

class Initializer:
    """Base class for weight initialization methods."""
    def initialize(self, shape):
        """Initialize weights/biases with given shape.
        
        Args:
            shape: Tuple of integers representing the desired array shape
            
        Returns:
            NumPy array of the specified shape with initialized values
        """
        raise NotImplementedError("Subclasses must implement initialize()")


class Zero(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)


class Uniform(Initializer):
    """Initialize with random values from uniform distribution.
    
    Args:
        low: Lower bound of the uniform distribution (default: -1.0)
        high: Upper bound of the uniform distribution (default: 1.0)
        seed: Random seed (default: None)
    """
    def __init__(self, low=-1.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def initialize(self, shape):
        """
        Returns:
            NumPy array with values sampled from Uniform[low, high]
        """
        return self.rng.uniform(self.low, self.high, shape)


class Normal(Initializer):
    def __init__(self, mean=0.0, variance=1.0, seed=None):
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def initialize(self, shape):
        """
        Returns:
            NumPy array with values sampled from N(mean, variance)
        """
        return self.rng.normal(self.mean, self.std, shape)

# Bonus
class Xavier(Initializer):
    """
    Samples weights from a uniform distribution within 
    [-sqrt(6/fan_in), sqrt(6/fan_in)] where fan_in is the number
    of input units in the weight tensor
    
    Args:
        gain: Scaling factor (default: 1.0)
    """
    def __init__(self, gain=1.0):
        self.gain = gain
    
    def initialize(self, shape):
        fan_in, fan_out = shape[0], shape[1]
        limit = self.gain * np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)


class He(Initializer):
    """
    Samples weights from a normal distribution with 
    mean 0 and standard deviation sqrt(2/fan_in)
    
    Args:
        scale: Scaling factor (default: 1.0)
    """
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def initialize(self, shape):
        fan_in = shape[0]
        std = self.scale * np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)