import numpy as np

from scripts.target_spectrum_generators.__base__ import SingleSourceTargetGenerator


class SingleSourceMeanGenerator(SingleSourceTargetGenerator):
    """
    Generator that computes the mean spectrum of the target region for each source image separately.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, source_images: np.ndarray, source_gts: np.ndarray, test_images: np.ndarray = None):
        """
        Computes the average spectrum over all target pixels for each image in the batch.

        Args:
            source_images (np.ndarray): Input image tensor of shape (B, C, H, W).
            source_gts (np.ndarray): Binary target mask of shape (B, 1, H, W).
            test_images (np.ndarray, optional): Not used in this method.

        Returns:
            tuple:
                - np.ndarray: Averaged target spectra of shape (B, C, 1, 1).
                - None
        """
        B, C, H, W = source_images.shape

        # Corrected transpose function
        img_flat = np.transpose(source_images, (0, 2, 3, 1)).reshape(B, -1, C)  # (B, H*W, C)
        target_flat = source_gts.reshape(B, -1)  # (B, H*W)

        # Identify target indices
        mask = target_flat.astype(bool)
        ts = [img_flat[b][mask[b]] if np.any(mask[b]) else np.zeros((1, C)) for b in range(B)]  # Ensure non-empty list

        # Average target spectra for each batch
        avg_target_spectrum = np.stack(
            [t.mean(axis=0) if t.shape[0] > 0 else np.zeros(C) for t in ts], axis=0
        )  # (B, C)

        # Reshape result to (B, C, 1, 1)
        result = avg_target_spectrum[:, :, np.newaxis, np.newaxis]

        return result, None



