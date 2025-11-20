from .__base__ import BaseVisualizer
from scripts.datasets.hyperspectral_dataset import HyperspectralImageDataset
from scripts.loggers.experiment_logger import ExperimentLogger
from scripts.utils.helpers import convert_to_rgb

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class ImageVisualizer(BaseVisualizer):
    def __init__(
        self,
        logger: ExperimentLogger,
        verbose: bool = False
    ):
        super().__init__(logger, verbose)

        if logger.config is None:
            raise ValueError("Logger must contain config to initialize ImageVisualizer.")

        config = logger.config
        self.dataset = HyperspectralImageDataset(
            folder_path=config.test_folder,
            img_transforms=config.img_transforms,
            gt_transforms=config.gt_transforms
        )
        self.rgb_indices = config.rgb_indices

    def _resolve_output_path(self, run: int, test: int, source: Optional[int]) -> str:
        base = os.path.join("results", f"run_{run}", f"test_{test}")
        return os.path.join(base, f"source_{source}" if source is not None else "source_None")

    def _visualize_entry(
        self,
        run: int,
        test: int,
        source: Optional[int],
        result: dict,
        **kwargs
    ) -> None:
        image, gt = self.dataset[test]
        C = image.shape[0]
        abs_rgb = [int(round(p * (C - 1))) for p in self.rgb_indices]
        rgb_test = convert_to_rgb(image[None], abs_rgb)

        rgb_source = None
        if source is not None:
            source_img, _ = self.dataset[source]
            rgb_source = convert_to_rgb(source_img[None], abs_rgb)

        detection_map = result.get("detection_map")
        pixel_indices = result.get("pixel_indices")
        rows, cols = None, None
        if pixel_indices is not None:
            h, w = image.shape[1:]
            rows, cols = np.unravel_index(pixel_indices, (h, w))

        out_dir = self._resolve_output_path(run, test, source)
        os.makedirs(out_dir, exist_ok=True)

        plt.imsave(os.path.join(out_dir, "test_rgb.jpg"), rgb_test)
        plt.imsave(os.path.join(out_dir, "ground_truth.jpg"), gt.squeeze(), cmap="viridis")
        if rgb_source is not None:
            plt.imsave(os.path.join(out_dir, "source_rgb.jpg"), rgb_source)
        if detection_map is not None:
            plt.imsave(os.path.join(out_dir, "detection_map.jpg"), np.squeeze(detection_map), cmap="viridis")

        fig, axes = plt.subplots(1, 4 if rgb_source is not None else 3, figsize=(18, 5))
        axes[0].imshow(rgb_test)
        if rows is not None and cols is not None:
            axes[0].scatter(cols, rows, c='red', s=25)
        axes[0].set_title("Test RGB")
        axes[0].axis("off")

        idx = 1
        if rgb_source is not None:
            axes[1].imshow(rgb_source)
            axes[1].set_title("Source RGB")
            axes[1].axis("off")
            idx += 1

        axes[idx].imshow(gt.squeeze(), cmap="viridis")
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")

        if detection_map is not None:
            axes[idx + 1].imshow(np.squeeze(detection_map), cmap="viridis")
            axes[idx + 1].set_title("Detection Map")
            axes[idx + 1].axis("off")

        plt.tight_layout()
        if self.verbose or kwargs.get("show", False):
            plt.show()
        else:
            plt.close()
