from .Loss import MSE, BCE, CCE

LOSS_REGISTRY = {
  "mse": MSE,
  "bce": BCE,
  "cce": CCE,
}

def get_loss(name):
  name_lower = name.lower()
  
  if name_lower not in LOSS_REGISTRY:
    raise ValueError(f"Unknown loss: {name}. Tersedia: {list(LOSS_REGISTRY.keys())}")
  
  return LOSS_REGISTRY[name_lower]() 

__all__ = ["MSE", "BCE", "CCE", "get_loss"]