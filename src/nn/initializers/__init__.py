from .Initializer import Initializer, Zero, Uniform, Normal, Xavier, He

INITIALIZER_REGISTRY = {
  "zero": Zero,
  "uniform": Uniform,
  "normal": Normal,
  "xavier": Xavier,
  "he": He,
}

def get_initializer(name, **kwargs):
    """
    **kwargs: Parameters to pass to the initializer constructor
    """
    if name not in INITIALIZER_REGISTRY:
        raise ValueError(
            f"Unknown initializer: {name}. "
            f"Available: {list(INITIALIZER_REGISTRY.keys())}"
        )
    return INITIALIZER_REGISTRY[name](**kwargs)

__all__ = ["Initializer", "Zero", "Uniform", "Normal", "Xavier", "He", "get_initializer"]