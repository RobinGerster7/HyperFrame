from typing import Literal

import numpy as np
import cv2
import matplotlib.cm as cm

class Resize:
    """
    Resize the spatial dimensions of a NumPy array.

    Args:
        size (tuple): Desired output size (height, width).
        interpolation (str or int): Interpolation mode as string (e.g., "bilinear", "nearest") or OpenCV constant.
            Supported values: "nearest" → cv2.INTER_NEAREST, "bilinear" → cv2.INTER_LINEAR, "bicubic" → cv2.INTER_CUBIC.

    Outputs:
        np.ndarray: Resized array of shape (C, H_out, W_out) or (1, H_out, W_out) for ground truth.
    """

    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }

    def __init__(self, size, interpolation="bilinear"):
        self.size = size  # (height, width)
        if isinstance(interpolation, str):
            if interpolation not in self.INTERPOLATION_MAP:
                raise ValueError(
                    f"Invalid interpolation mode: {interpolation}. Supported: {list(self.INTERPOLATION_MAP.keys())}")
            self.interpolation = self.INTERPOLATION_MAP[interpolation]
        else:
            self.interpolation = interpolation  # Allow direct integer values

    def __call__(self, image):
        """
        Resize the input array to the specified size.

        Args:
            image (np.ndarray): Input array of shape (C, H, W).

        Returns:
            np.ndarray: Resized array of shape (C, H_out, W_out).
        """
        return np.stack([
            cv2.resize(image[c], self.size[::-1], interpolation=self.interpolation)
            for c in range(image.shape[0])
        ], axis=0)


class BaseNormalize:
    """
    Base class for normalization transforms.
    """
    apply_to_gt: bool = False

    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MinMaxNormalize(BaseNormalize):
    """
    Normalize a NumPy array either image-wise or pixel-wise to a specific range [min_val, max_val].

    Args:
        min_val (float): Minimum value of the normalized range.
        max_val (float): Maximum value of the normalized range.
        mode (str): 'image' for whole-image normalization, or 'pixel' for per-pixel normalization across bands.

    Returns:
        np.ndarray: Normalized array of the same shape.
    """
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, mode: Literal["image", "pixel"] = "image"):
        assert mode in ("image", "pixel"), "mode must be 'image' or 'pixel'"
        self.min_val = min_val
        self.max_val = max_val
        self.mode = mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "image":
            image_min = np.min(image)
            image_max = np.max(image)
            if image_max == image_min:
                return np.zeros_like(image)
            norm = (image - image_min) / (image_max - image_min)
            return norm * (self.max_val - self.min_val) + self.min_val

        elif self.mode == "pixel":
            # Assume image shape is (C, H, W)
            C, H, W = image.shape
            image_reshaped = image.reshape(C, -1).T  # shape (H*W, C)
            min_vals = image_reshaped.min(axis=1, keepdims=True)
            max_vals = image_reshaped.max(axis=1, keepdims=True)
            denom = (max_vals - min_vals)
            denom[denom == 0] = 1  # avoid division by zero
            norm = (image_reshaped - min_vals) / denom
            norm = norm.T.reshape(C, H, W)
            return norm * (self.max_val - self.min_val) + self.min_val


class L2Normalize(BaseNormalize):
    """
    Apply L2 normalization to a NumPy array either image-wise or pixel-wise.

    Args:
        mode (str): 'image' for whole-image normalization, or 'pixel' for per-pixel normalization across bands.

    Returns:
        np.ndarray: L2-normalized array of the same shape.
    """
    def __init__(self, mode: Literal["image", "pixel"] = "image"):
        assert mode in ("image", "pixel"), "mode must be 'image' or 'pixel'"
        self.mode = mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "image":
            norm = np.linalg.norm(image)
            return image / norm if norm != 0 else np.zeros_like(image)

        elif self.mode == "pixel":
            # Assume image shape is (C, H, W)
            C, H, W = image.shape
            reshaped = image.reshape(C, -1)  # shape (C, H*W)
            norms = np.linalg.norm(reshaped, axis=0, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            normalized = reshaped / norms
            return normalized.reshape(C, H, W)

class L1Normalize(BaseNormalize):
    """
    Apply L1 normalization to a NumPy array either image-wise or pixel-wise.

    Args:
        mode (str): 'image' for whole-image normalization, or 'pixel' for per-pixel normalization across bands.

    Returns:
        np.ndarray: L1-normalized array of the same shape.
    """
    def __init__(self, mode: Literal["image", "pixel"] = "image"):
        assert mode in ("image", "pixel"), "mode must be 'image' or 'pixel'"
        self.mode = mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "image":
            norm = np.sum(np.abs(image))
            return image / norm if norm != 0 else np.zeros_like(image)

        elif self.mode == "pixel":
            # Assume image shape is (C, H, W)
            C, H, W = image.shape
            reshaped = image.reshape(C, -1)  # shape (C, H*W)
            norms = np.sum(np.abs(reshaped), axis=0, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            normalized = reshaped / norms
            return normalized.reshape(C, H, W)

class ZScoreNormalize(BaseNormalize):
    """
    Normalize each spectral band using Z-score normalization across the entire image.

    Formula:
        x' = (x - μ) / σ

    Args:
        epsilon (float): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        np.ndarray: Z-score normalized array of the same shape.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization.

        Args:
            image (np.ndarray): Input hyperspectral image of shape (C, H, W).

        Returns:
            np.ndarray: Z-score normalized image of shape (C, H, W).
        """
        mean = image.mean(axis=(1, 2), keepdims=True)  # (C, 1, 1)
        std = image.std(axis=(1, 2), keepdims=True)    # (C, 1, 1)
        return (image - mean) / (std + self.epsilon)

class ClampChannels:
    """
    Retain only the first `max_channels` in a hyperspectral image.

    Args:
        max_channels (int): Number of channels to retain.

    Outputs:
        np.ndarray: Clamped array of shape (max_channels, H, W).
    """
    apply_to_gt = False

    def __init__(self, max_channels):
        self.max_channels = max_channels

    def __call__(self, image):
        return image[:self.max_channels, :, :]


class ChannelSubsampling:
    """
    Subsample spectral channels by keeping every N-th channel, determined by a specified ratio.

    Args:
        keep_ratio (float): Ratio of channels to retain (e.g., 0.5 keeps roughly half the channels).
                            Must be in the range (0, 1].

    Outputs:
        np.ndarray: Subsampled array with fewer spectral channels.
    """
    apply_to_gt = False

    def __init__(self, keep_ratio: float) -> None:
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in the range (0, 1].")
        self.keep_ratio = keep_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply channel subsampling to the input image.

        Args:
            image (np.ndarray): Hyperspectral image of shape (C, H, W).

        Returns:
            np.ndarray: Subsampled image of shape (C_sub, H, W).
        """
        step = max(1, int(round(1 / self.keep_ratio)))
        return image[::step, :, :]

class NonlinearBackgroundSuppression:
    """
    Apply Nonlinear Background Suppression (NBS) to suppress background in detection maps.

    The transformation is:
        μ_i' = exp(-(μ_i - 1)^2 / δ)

    Args:
        delta (float): Tunable parameter δ > 0 that controls suppression strength. Default is 0.2.

    Outputs:
        np.ndarray: Background-suppressed array of the same shape as input.
    """
    apply_to_gt = False

    def __init__(self, delta: float = 0.2):
        if delta <= 0:
            raise ValueError("Delta must be a positive value.")
        self.delta = delta

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the NBS transformation to the input.

        Args:
            image (np.ndarray): Input array of shape (C, H, W).

        Returns:
            np.ndarray: Suppressed image of same shape.
        """
        return np.exp(-((image - 1) ** 2) / self.delta)


class RelativeBandSelector:
    """
    Select specific spectral bands from a hyperspectral image using relative indices (0.0 to 1.0).

    Args:
        band_positions (tuple[float, float, float]): Relative positions for R, G, B bands (0.0 to 1.0).
    """
    apply_to_gt = False

    def __init__(self, band_positions=(0.75, 0.5, 0.25)):
        self.band_positions = band_positions

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Select RGB bands from the input image.

        Args:
            image (np.ndarray): Hyperspectral image (C, H, W)

        Returns:
            np.ndarray: RGB image (3, H, W)
        """
        C, H, W = image.shape
        indices = [min(C - 1, max(0, int(pos * C))) for pos in self.band_positions]
        rgb = np.stack([image[i] for i in indices], axis=0)  # (3, H, W)
        return rgb

class BrightnessScaling:
    """
    Apply brightness scaling to an RGB image.

    Args:
        factor (float): Brightness multiplier.
    """
    apply_to_gt = False

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def set_factor(self, factor: float):
        self.factor = factor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.clip(image * self.factor, 0.0, 1.0)

class ViridisColormap:
    """
    Applies the viridis colormap to input of shape (B, 1, H, W) and returns (B, 3, H, W).

    Raises:
        ValueError if input shape is not (B, 1, H, W).
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected input of shape (B, 1, H, W), got {image.shape}")

        B, _, H, W = image.shape
        output = np.empty((B, 3, H, W), dtype=np.uint8)

        for i in range(B):
            norm = (image[i, 0] - image[i, 0].min()) / (np.ptp(image[i, 0]) + 1e-8)
            colored = cm.viridis(norm)[:, :, :3]  # (H, W, 3)
            output[i] = (colored * 255).astype(np.uint8).transpose(2, 0, 1)  # (3, H, W)

        return output

