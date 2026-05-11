from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

    if image.max() > 1.0 or image.min() < 0.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D heatmap to a target (H, W)."""
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got {heatmap.shape}")

    height, width = int(target_size[0]), int(target_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("target_size values must be positive")

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.min(heatmap))
    peak = float(np.max(heatmap))
    if peak > 0:
        heatmap = heatmap / peak

    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0), mode="L")
    heatmap_image = heatmap_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image, dtype=np.float32) / 255.0


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Map a normalized heatmap to a simple red-yellow color ramp."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got {heatmap.shape}")

    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 * heatmap, 0.0, 1.0)
    green = np.clip(1.5 * heatmap - 0.5, 0.0, 1.0)
    blue = np.clip(1.0 - 2.0 * heatmap, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend an image and a heatmap into a float32 RGB overlay in [0, 1]."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    image = _normalize_image(image)
    if heatmap.ndim == 2:
        heatmap_rgb = heatmap_to_rgb(heatmap)
    elif heatmap.ndim == 3 and heatmap.shape[-1] == 3:
        heatmap_rgb = np.asarray(heatmap, dtype=np.float32)
    else:
        raise ValueError(f"Expected heatmap shape (H, W) or (H, W, 3), got {heatmap.shape}")

    if heatmap_rgb.shape[:2] != image.shape[:2]:
        resized_heatmap = resize_heatmap(heatmap_rgb[..., 0], image.shape[:2]) if heatmap_rgb.ndim == 3 else resize_heatmap(heatmap_rgb, image.shape[:2])
        heatmap_rgb = heatmap_to_rgb(resized_heatmap)

    overlay = (1.0 - alpha) * image + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def save_overlay(image: np.ndarray, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> Path:
    """Save a Grad-CAM overlay image to disk."""
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(overlay * 255.0))).save(output_file)
    return output_file