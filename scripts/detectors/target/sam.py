import numpy as np
from scripts.detectors.__base__ import Detector, TargetDetector


class SAM(TargetDetector):
    """
    Spectral Angle Mapper (SAM) detector.

    Reference:
        Kruse, Fred A., et al. "The spectral image processing system (SIPS)—interactive visualization
        and analysis of imaging spectrometer data." Remote sensing of environment 44.2-3 (1993): 145-163.
    """

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the SAM detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target spectrum of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W) with inverse angles.
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        img_flat = img.reshape(B, C, -1)  # Shape: (B, C, H*W)
        target_flat = target.reshape(B, C, 1)  # Shape: (B, C, 1)

        # Compute norms
        img_norms = np.linalg.norm(img_flat, axis=1, keepdims=True)  # Shape: (B, 1, H*W)
        tgt_norm = np.linalg.norm(target_flat, axis=1, keepdims=True)  # Shape: (B, 1, 1)

        # Compute cosine of spectral angles
        dot_product = np.sum(img_flat * target_flat, axis=1, keepdims=True)  # Shape: (B, 1, H*W)
        cos_angles = dot_product / (img_norms * tgt_norm + 1e-8)

        # Clamp values to valid acos range to avoid numerical issues
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)  # Shape: (B, 1, H*W)

        # Return inverse angle as similarity score (higher = more similar)
        result = 1.0 / (angles + 1e-6)  # Avoid division by zero
        result = result.reshape(B, 1, H, W)

        return result
