from .Optimizer import Optimizer, GradientDescent, Adam

OPTIMIZER_REGISTRY = {
    "gradient_descent": GradientDescent,
    "adam": Adam,
}


def get_optimizer(name, **kwargs):
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name](**kwargs)


__all__ = ["Optimizer", "GradientDescent", "Adam", "get_optimizer"]
