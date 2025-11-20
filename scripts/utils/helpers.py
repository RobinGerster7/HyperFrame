from typing import List
from scripts.utils.transforms import Resize, MinMaxNormalize
from typing import Callable
import numpy as np
from torch.utils.data import DataLoader
from scripts.datasets.hyperspectral_dataset import HyperspectralImageDataset

def create_dataloader(folder_path: str,
                      img_transforms: list[Callable[[np.ndarray], np.ndarray]],
                      gt_transforms: list[Callable[[np.ndarray], np.ndarray]],
                      batch_size=1,
                      shuffle=False) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a hyperspectral dataset.

    Args:
        folder_path (str): Path to the folder containing the hyperspectral dataset.
        img_transforms (list of Callable): List of transformation functions to apply to the hyperspectral images.
        gt_transforms (list of Callable): List of transformation functions to apply to the ground truth labels.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader that yields (image, ground truth) pairs.
    """

    dataset = HyperspectralImageDataset(
        folder_path,
        img_transforms=img_transforms,
        gt_transforms=gt_transforms,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)



def convert_to_rgb(image: np.ndarray, indices: List[float | int]) -> np.ndarray:
    """
    Converts a hyperspectral image to an RGB image using selected spectral channels.

    Args:
        image (np.ndarray): Hyperspectral image of shape (1, C, H, W),
                            where C is the number of spectral channels.
        indices (list[int]): List of three channel indices to extract for RGB visualization.
                             Should contain exactly 3 integers.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with values scaled to [0, 255] and dtype uint8.
    """
    assert image.ndim == 4 and image.shape[1] >= 3, "Expected input of shape (1, C, H, W) with C >= 3"
    assert len(indices) == 3, "Exactly three channel indices must be provided for RGB conversion"

    C = image.shape[1]
    indices = [int(round(i * (C - 1))) if isinstance(i, float) else i for i in indices]

    # Extract and rearrange channels to (H, W, 3)
    rgb_image = image[0, indices, :, :].transpose(1, 2, 0)

    # Normalize to range [0, 255]
    normalizer = MinMaxNormalize(min_val=0, max_val=255)
    return normalizer(rgb_image).astype(np.uint8)
