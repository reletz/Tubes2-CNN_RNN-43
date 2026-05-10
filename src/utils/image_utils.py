from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _validate_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    if len(target_size) != 2:
        raise ValueError("target_size must contain exactly 2 values: (height, width)")

    height, width = int(target_size[0]), int(target_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("target_size values must be positive")

    return height, width


def load_image(path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load one image, resize it, and normalize pixels to [0, 1].

    Args:
        path: Image file path.
        target_size: Output spatial size as (height, width).

    Returns:
        Image array with shape (H, W, 3) in float32.
    """
    height, width = _validate_target_size(target_size)

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


def load_batch(paths: Iterable[str | Path], target_size: tuple[int, int]) -> np.ndarray:
    """Load and stack images into a batch tensor.

    Args:
        paths: Iterable of image file paths.
        target_size: Output spatial size as (height, width).

    Returns:
        Batch tensor with shape (N, H, W, C) in float32.
    """
    image_arrays = [load_image(path, target_size) for path in paths]
    if not image_arrays:
        raise ValueError("paths is empty; cannot create a batch")

    return np.stack(image_arrays, axis=0)


def extract_features(
    paths: Iterable[str | Path],
    encoder,
    target_size: tuple[int, int],
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Extract image features using a frozen Keras encoder.

    Args:
        paths: Iterable of image file paths.
        encoder: Keras model with a .predict(...) method.
        target_size: Input spatial size expected by the encoder, (height, width).
        batch_size: Number of images per forward pass.
        output_path: Optional path to save extracted features as .npy.

    Returns:
        Feature tensor from the encoder.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    path_list = list(paths)
    if not path_list:
        raise ValueError("paths is empty; cannot extract features")

    feature_batches: list[np.ndarray] = []
    for start in range(0, len(path_list), batch_size):
        batch_paths = path_list[start : start + batch_size]
        batch_images = load_batch(batch_paths, target_size)
        batch_features = encoder.predict(batch_images, verbose=0)
        feature_batches.append(np.asarray(batch_features))

    features = np.concatenate(feature_batches, axis=0)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, features)

    return features
