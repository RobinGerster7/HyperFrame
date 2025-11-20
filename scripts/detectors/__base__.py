from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    """
    Abstract base class for all detectors using NumPy.

    All detectors operate on hyperspectral image input.

    Methods:
        forward(img): Abstract method for performing detection.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Perform the detection operation.

        Args:
            img (np.ndarray): Hyperspectral image of shape (B, C, H, W).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        pass


class TargetDetector(Detector):
    """
    Abstract base class for detectors requiring a target spectrum.

    Methods:
        forward(img, target): Abstract method for target-based detection.
    """

    @abstractmethod
    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Perform the target-based detection.

        Args:
            img (np.ndarray): Hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target spectrum of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        pass


class AnomalyDetector(Detector):
    """
    Abstract base class for detectors that do not use a target spectrum.

    Methods:
        forward(img): Abstract method for anomaly detection.
    """

    @abstractmethod
    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Perform anomaly detection.

        Args:
            img (np.ndarray): Hyperspectral image of shape (B, C, H, W).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        pass
