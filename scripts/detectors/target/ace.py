import numpy as np
import scipy.linalg
from scripts.detectors.__base__ import TargetDetector

class ACE(TargetDetector):
    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        X = img[0].reshape(C, H * W).astype(np.float32)  # (C, N)
        t = target[0].reshape(C, 1).astype(np.float32)   # (C, 1)

        mu = X.mean(axis=1, keepdims=True)
        X0 = X - mu
        t0 = t - mu

        R = (X0 @ X0.T) / (H * W)
        R += np.eye(C, dtype=np.float32) * 1e-6

        # Solve instead of inverting
        Rinv_t = scipy.linalg.solve(R, t0, check_finite=False)
        Rinv_X = scipy.linalg.solve(R, X0, check_finite=False)

        # Correct and stable ACE
        numerator = (Rinv_t.T @ X0) ** 2                # (1, N)
        denom_t = float(t0.T @ Rinv_t)                  # scalar
        denom_x = np.sum(X0 * Rinv_X, axis=0, keepdims=True)  # (1, N)

        ace = numerator / (denom_t * denom_x + 1e-8)
        ace_map = ace.reshape(1, 1, H, W)

        return ace_map
