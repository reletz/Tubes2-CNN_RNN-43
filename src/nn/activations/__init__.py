from .Activation import Linear, ReLU, Sigmoid, Tanh, Softmax, ELU, LeakyReLU

ACTIVATION_REGISTRY = {
  "linear": Linear,
  "relu": ReLU,
  "sigmoid": Sigmoid,
  "tanh": Tanh,
  "softmax": Softmax,
  "leakyrelu": LeakyReLU,
  "elu": ELU
}

def get_activation(name):
  name_lower = name.lower()
  
  if name_lower not in ACTIVATION_REGISTRY:
    raise ValueError(f"Unknown activation: {name}. Tersedia: {list(ACTIVATION_REGISTRY.keys())}")
  
  return ACTIVATION_REGISTRY[name_lower]() 

__all__ = ["Linear", "ReLU", "Sigmoid", "Tanh", "Softmax", "ELU", "LeakyReLU", "get_activation"]