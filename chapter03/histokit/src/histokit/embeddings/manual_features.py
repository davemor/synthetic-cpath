"""Manual 260-dimensional patch features for foreground/background work."""

from __future__ import annotations

import numpy as np
from PIL import Image


RGB_CHANNELS = ("r", "g", "b")
HSV_CHANNELS = ("h", "s", "v")
HIST_BINS = 8
THUMBNAIL_SIZE = 8
POOL_INPUT_SIZE = 224


def manual_feature_names() -> list[str]:
    names: list[str] = []
    for space, channels in (("rgb", RGB_CHANNELS), ("hsv", HSV_CHANNELS)):
        for stat in ("mean", "std"):
            for channel in channels:
                names.append(f"{space}_{channel}_{stat}")

    names.extend(
        [
            "gray_mean",
            "gray_std",
            "gray_p10",
            "gray_median",
            "gray_p90",
            "gray_entropy",
            "gray_edge_absdiff_mean",
            "gray_edge_absdiff_std",
        ]
    )

    for space, channels in (("rgb", RGB_CHANNELS), ("hsv", HSV_CHANNELS)):
        for channel in channels:
            for bin_index in range(HIST_BINS):
                names.append(f"{space}_{channel}_hist_bin_{bin_index:02d}")

    for row in range(THUMBNAIL_SIZE):
        for column in range(THUMBNAIL_SIZE):
            for channel in RGB_CHANNELS:
                names.append(f"thumbnail_{row}_{column}_{channel}")

    if len(names) != 260:
        raise RuntimeError(f"Expected 260 manual feature names, got {len(names)}")
    return names


def extract_manual_features(image: Image.Image) -> np.ndarray:
    rgb = _image_to_rgb_array(image)
    hsv = rgb_to_hsv(rgb)
    gray = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32, copy=False)

    features: list[float] = []
    for array in (rgb, hsv):
        features.extend(array.mean(axis=(0, 1)).astype(float).tolist())
        features.extend(array.std(axis=(0, 1)).astype(float).tolist())

    features.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(np.percentile(gray, 10)),
            float(np.median(gray)),
            float(np.percentile(gray, 90)),
            normalized_entropy(gray),
        ]
    )
    edge_values = gray_edge_absdiffs(gray)
    features.extend([float(edge_values.mean()), float(edge_values.std())])

    for array in (rgb, hsv):
        for channel_index in range(3):
            features.extend(channel_histogram(array[..., channel_index]).tolist())

    features.extend(pooled_thumbnail(rgb).reshape(-1).astype(float).tolist())
    out = np.asarray(features, dtype=np.float32)
    if out.shape != (260,):
        raise RuntimeError(f"Expected manual feature shape (260,), got {out.shape}")
    return out


def extract_manual_feature_batch(images: list[Image.Image]) -> np.ndarray:
    if not images:
        return np.empty((0, 260), dtype=np.float32)
    return np.stack([extract_manual_features(image) for image in images], axis=0)


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    if image.size != (POOL_INPUT_SIZE, POOL_INPUT_SIZE):
        image = image.resize((POOL_INPUT_SIZE, POOL_INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc, dtype=np.float32)
    nonzero = delta > 0
    red_max = nonzero & (maxc == red)
    green_max = nonzero & (maxc == green)
    blue_max = nonzero & (maxc == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue = hue / 6.0

    saturation = np.zeros_like(maxc, dtype=np.float32)
    positive_value = maxc > 0
    saturation[positive_value] = delta[positive_value] / maxc[positive_value]
    value = maxc.astype(np.float32, copy=False)
    return np.stack([hue, saturation, value], axis=-1).astype(np.float32, copy=False)


def normalized_entropy(gray: np.ndarray) -> float:
    counts, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0].astype(np.float64) / float(total)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / 8.0)


def gray_edge_absdiffs(gray: np.ndarray) -> np.ndarray:
    horizontal = np.abs(np.diff(gray, axis=1)).reshape(-1)
    vertical = np.abs(np.diff(gray, axis=0)).reshape(-1)
    if horizontal.size == 0 and vertical.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    return np.concatenate([horizontal, vertical]).astype(np.float32, copy=False)


def channel_histogram(channel: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(channel, bins=HIST_BINS, range=(0.0, 1.0))
    total = int(counts.sum())
    if total == 0:
        return np.zeros(HIST_BINS, dtype=np.float32)
    return (counts.astype(np.float32) / float(total)).astype(np.float32, copy=False)


def pooled_thumbnail(rgb: np.ndarray) -> np.ndarray:
    step = POOL_INPUT_SIZE // THUMBNAIL_SIZE
    return rgb.reshape(THUMBNAIL_SIZE, step, THUMBNAIL_SIZE, step, 3).mean(axis=(1, 3))
