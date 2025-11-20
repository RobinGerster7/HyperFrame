import numpy as np
from scripts.detectors.__base__ import AnomalyDetector
import scipy.linalg
import warnings

class RX(AnomalyDetector):
    """
    Implementation of the Global Reed-Xiaoli (RX) anomaly detector.

    Reference:
        Reed, I. S., & Yu, X. (1990). Adaptive multiple-band CFAR detection of an optical pattern
        with unknown spectral distribution. IEEE Transactions on Acoustics, Speech, and Signal Processing.

    The RX detector computes the Mahalanobis distance from the global mean using the inverse covariance matrix.
    """

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Optimized forward pass for the RX anomaly detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).

        Returns:
            np.ndarray: Detection map of shape (B, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 is supported"

        img = img[0].reshape(C, H * W).astype(np.float32)  # (C, N)
        mu = img.mean(axis=1, keepdims=True)  # (C, 1)
        centered = img - mu  # (C, N)

        cov = (centered @ centered.T) / (H * W)  # (C, C)
        cov += np.eye(C, dtype=np.float32) * 1e-6
        inv_cov = scipy.linalg.inv(cov).astype(np.float32)

        # Safer Mahalanobis distance computation
        rx_scores = np.sum(centered * (inv_cov @ centered), axis=0)  # (N,)
        return rx_scores.reshape(1, 1, H, W)  # (B=1, 1, H, W)
