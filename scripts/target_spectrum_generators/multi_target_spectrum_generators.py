import numpy as np

from scripts.target_spectrum_generators.__base__ import MultiSourceTargetGenerator
from scripts.target_spectrum_generators.single_target_spectrum_generators import SingleSourceMeanGenerator


class MultiSourceMeanGenerator(MultiSourceTargetGenerator):
    """
    Generator that computes the mean spectrum of the target region across all source images jointly.
    """

    def forward(self, source_images: np.ndarray, source_gts: np.ndarray, test_images: np.ndarray = None):
        """
        Computes the average spectrum over all target pixels across all images in the batch.

        Args:
            source_image (np.ndarray): Input image tensor of shape (B, C, H, W).
            source_gts (np.ndarray): Binary target mask of shape (B, 1, H, W).
            test_image (np.ndarray, optional): Not used in this method.

        Returns:
            tuple:
                - np.ndarray: Averaged target spectra of shape (B, C, 1, 1).
                - None
        """
        per_image_spectra, _ = SingleSourceMeanGenerator().forward(source_images, source_gts, test_images)
        mean_spectrum = per_image_spectra.mean(axis=0, keepdims=True)  # shape: (1, C, 1, 1)
        return mean_spectrum, None