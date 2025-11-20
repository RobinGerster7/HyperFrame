import numpy as np
from scripts.detectors.__base__ import Detector, TargetDetector


class SID(TargetDetector):
    """
    Spectral Information Divergence (SID) detector.

    Reference:
        Chang, Chein-I. "An information-theoretic approach to spectral variability, similarity, and
        discrimination for hyperspectral image analysis." IEEE Transactions on Information Theory 46.5 (2000): 1927-1932.
    """

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the SID detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target spectrum of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W) with inverse SID scores (higher = more similar).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        eps = 1e-12

        img_flat = img.reshape(B, C, -1)  # (B, C, H*W)
        target_flat = target.reshape(B, C, 1)  # (B, C, 1)

        # Normalize to probability distributions (add eps before log/div to avoid zeros)
        pi = img_flat / (np.sum(img_flat, axis=1, keepdims=True) + eps)  # (B, C, H*W)
        di = target_flat / (np.sum(target_flat, axis=1, keepdims=True) + eps)  # (B, C, 1)

        # Ensure no log(0) or div by zero
        pi = np.clip(pi, eps, 1.0)
        di = np.clip(di, eps, 1.0)

        # Compute KL divergences
        kl_pi_di = np.sum(pi * np.log(pi / di), axis=1, keepdims=True)  # (B, 1, H*W)
        kl_di_pi = np.sum(di * np.log(di / pi), axis=1, keepdims=True)  # (B, 1, H*W)
        sid = kl_pi_di + kl_di_pi  # symmetric SID

        # Invert to convert to similarity (higher = better match)
        score = 1.0 / (sid + eps)  # (B, 1, H*W)
        score = score.reshape(B, 1, H, W)

        return score
