import numpy as np
from abc import ABC, abstractmethod


class TargetSpectrumGenerator(ABC):
    """
    Abstract base class for all target spectrum generators.

    These generators are used to produce a representative target spectral signature
    from one or more hyperspectral images and their corresponding ground truth masks.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes a target spectrum given input images and (optionally) a test image.

        Must be implemented by subclasses.
        """
        pass


class SingleSourceTargetGenerator(TargetSpectrumGenerator):
    """
    Abstract base class for target spectrum generators that operate on individual source-test image pairs.

    These generators take a single source image and its corresponding ground truth mask
    to compute a target spectrum for detecting targets in a test image.
    """

    @abstractmethod
    def forward(self, source_image: np.ndarray, source_gt: np.ndarray,
                test_image: np.ndarray = None):
        """
        Computes a target spectrum from a single source image and its ground truth.

        Args:
            source_image (np.ndarray): Input source image of shape (B, C, H, W).
            source_gt (np.ndarray): Binary ground truth mask of shape (B, 1, H, W).
            test_image (np.ndarray, optional): Test image of shape (B, C, H, W), used for test-adaptive generation.

        Returns:
            target_spectrum (np.ndarray): The computed target spectrum, typically of shape (C,).
            pixel_indices (np.ndarray): Indices of pixels contributing to the target spectrum,
                                        flattened to match the (H, W) layout.
        """
        pass


class MultiSourceTargetGenerator(TargetSpectrumGenerator):
    """
    Abstract base class for target spectrum generators that operate on multiple source images simultaneously.

    These generators aggregate information across all source images and their ground truth masks
    to compute a single target spectrum for use on a test image.
    """

    @abstractmethod
    def forward(self, source_images: np.ndarray, source_gts: np.ndarray,
                test_image: np.ndarray = None):
        """
        Computes a target spectrum from multiple source images and their masks.

        Args:
            source_images (np.ndarray): Array of shape (B, C, H, W) containing source images.
            source_gts (np.ndarray): Array of shape (B, 1, H, W) containing source ground truth masks.
            test_image (np.ndarray, optional): Test image of shape (B, C, H, W), if used in the generator.

        Returns:
            target_spectrum (np.ndarray): The computed target spectrum, typically of shape (C,).
            pixel_indices (np.ndarray): Flattened indices of selected pixels, if applicable.
        """
        pass
