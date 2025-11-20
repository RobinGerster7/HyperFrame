import numpy as np
from scripts.target_spectrum_generators.__base__ import SingleSourceTargetGenerator
from scripts.target_spectrum_generators.__base__ import MultiSourceTargetGenerator

class SingleBackgroundMeanSpectrumGenerator(SingleSourceTargetGenerator):
    """
    Generator that computes the mean spectrum of the background region for each source image separately.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, source_images: np.ndarray, source_gts: np.ndarray, test_images: np.ndarray = None):
        """
        Computes the average spectrum over all background pixels for each image in the batch.

        Args:
            source_images (np.ndarray): Input image tensor of shape (B, C, H, W).
            source_gts (np.ndarray): Binary target mask of shape (B, 1, H, W).
            test_images (np.ndarray, optional): Not used in this method.

        Returns:
            tuple:
                - np.ndarray: Averaged background spectra of shape (B, C, 1, 1).
                - None
        """
        B, C, H, W = source_images.shape

        img_flat = np.transpose(source_images, (0, 2, 3, 1)).reshape(B, -1, C)  # (B, H*W, C)
        gt_flat = source_gts.reshape(B, -1)  # (B, H*W)

        # Invert mask to get background (where gt == 0)
        bg_mask = ~gt_flat.astype(bool)

        # Extract and average background pixels
        bg_spectra = [img_flat[b][bg_mask[b]] if np.any(bg_mask[b]) else np.zeros((1, C)) for b in range(B)]
        avg_background = np.stack(
            [s.mean(axis=0) if s.shape[0] > 0 else np.zeros(C) for s in bg_spectra], axis=0
        )  # (B, C)

        return avg_background[:, :, np.newaxis, np.newaxis], None




class MultiBackgroundMeanGenerator(MultiSourceTargetGenerator):
    """
    Generator that computes the mean spectrum of the background region across all source images jointly.
    """

    def forward(self, source_images: np.ndarray, source_gts: np.ndarray, test_images: np.ndarray = None):
        """
        Computes the average background spectrum over all background pixels across all images in the batch.

        Args:
            source_images (np.ndarray): Input image tensor of shape (B, C, H, W).
            source_gts (np.ndarray): Binary target mask of shape (B, 1, H, W).
            test_images (np.ndarray, optional): Not used in this method.

        Returns:
            tuple:
                - np.ndarray: Averaged background spectra of shape (1, C, 1, 1).
                - None
        """
        per_image_spectra, _ = BackgroundMeanSpectrumGenerator().forward(source_images, source_gts, test_images)
        mean_spectrum = per_image_spectra.mean(axis=0, keepdims=True)  # shape: (1, C, 1, 1)
        return mean_spectrum, None
