import os
import random
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from scripts.experiments.__base__ import BaseExperiment, DemoExperiment
from scripts.experiments.configs.demo_config import DemoConfig
from scripts.target_spectrum_generators.__base__ import SingleSourceTargetGenerator, MultiSourceTargetGenerator
from scripts.target_spectrum_generators.background_spectrum_generators import SingleBackgroundMeanSpectrumGenerator, \
    SingleBackgroundMeanSpectrumGenerator
from scripts.target_spectrum_generators.single_target_spectrum_generators import SingleSourceMeanGenerator
from scripts.utils import helpers
from scripts.utils.metrics import evaluation_metrics

class OracleExperiment(DemoExperiment):
    """
    Performs a self-target evaluation where each image is used as both source and test.
    """

    def step_experiment(
        self,
        test_loader: DataLoader,
        source_loader: DataLoader,
        run_id: int,
        pbar: tqdm,
    ) -> None:
        generator = self.config.target_spectrum_generator
        if not isinstance(generator, (SingleSourceTargetGenerator, MultiSourceTargetGenerator)):
            raise TypeError("Target generator must be SingleSourceTargetGenerator or MultiSourceTargetGenerator.")

        all_images = [img.cpu().numpy() for img, _ in test_loader]
        all_gts = [gt.cpu().numpy() for _, gt in test_loader]

        if not all_images:
            raise ValueError("No data found in test_loader.")

        all_images = np.concatenate(all_images, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)

        for index in range(len(all_images)):
            test_image = all_images[index:index + 1]
            test_gt = all_gts[index:index + 1]

            source_spectrum, _ = SingleSourceMeanGenerator().forward(test_image, test_gt)
            background_spectrum, _ = SingleBackgroundMeanSpectrumGenerator().forward(test_image, test_gt)

            start_time = time.time()
            target_spectrum, pixel_indices = generator.forward(test_image, test_gt, test_image)
            detection_map = self.get_detection_map(test_image, target_spectrum)
            elapsed_time = round(time.time() - start_time, 4)

            metrics = evaluation_metrics(test_gt.reshape(-1), detection_map.reshape(-1))
            metric_dict = {
                metric: round(val, 5)
                for metric, val in zip(self.metrics[:-1], metrics)
            }
            metric_dict["Inference Time (s)"] = elapsed_time

            # Add custom fields if needed
            self.logger.record(
                run_id=run_id,
                test_index=index,
                source_index=index,
                data={
                    **metric_dict,
                    "pixel_indices": pixel_indices.tolist() if pixel_indices is not None else None,
                    "spectra": {
                        "source": source_spectrum.squeeze().flatten().tolist(),
                        "test": source_spectrum.squeeze().flatten().tolist(),
                        "optimized": target_spectrum.squeeze().flatten().tolist(),
                        "background": background_spectrum.squeeze().flatten().tolist()
                    },
                    "detection_map": detection_map,
                    "ground_truth_map": test_gt,
                },
            )
            pbar.update(1)