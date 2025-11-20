import shutil
import time
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.loggers.experiment_logger import ExperimentLogger
from scripts.experiments.configs.__base__ import BaseExperimentConfig
from scripts.experiments.configs.demo_config import DemoConfig
from scripts.target_spectrum_generators.__base__ import MultiSourceTargetGenerator, SingleSourceTargetGenerator
from scripts.utils import helpers


class BaseExperiment(ABC):
    def __init__(self, config: BaseExperimentConfig) -> None:
        self.config = config
        shutil.rmtree("results_hcem", ignore_errors=True)

    @abstractmethod
    def run(self) -> None:
        pass


class DemoExperiment(BaseExperiment):
    def __init__(self, config: DemoConfig) -> None:
        super().__init__(config)
        self.metrics = [
            "AUC (Pf, Pd)", "AUC (τ, Pd)", "AUC (τ, Pf)",
            "AUC OA", "AUC SNPR", "Inference Time (s)"
        ]
        self.logger = ExperimentLogger(config)

    @abstractmethod
    def step_experiment(
        self,
        test_loader: DataLoader,
        source_loader: DataLoader,
        run_id: int,
        pbar: tqdm,
    ) -> None:
        pass

    def get_detection_map(
        self,
        test_image: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        detection_map = self.config.detector.forward(test_image, target)
        if self.config.post_processing_transforms:
            for transform in self.config.post_processing_transforms:
                detection_map = transform(detection_map)
        return detection_map

    def run(self) -> None:
        time.sleep(0.1)

        # Load once to get number of test images
        test_loader = helpers.create_dataloader(
            self.config.test_folder,
            self.config.img_transforms,
            self.config.gt_transforms,
        )
        num_test_images = len(test_loader)

        is_single_source = isinstance(self.config.target_spectrum_generator, SingleSourceTargetGenerator)

        if is_single_source:
            total_steps = self.config.num_runs * num_test_images * (num_test_images - 1)
        else:
            total_steps = self.config.num_runs * num_test_images

        with tqdm(total=total_steps, desc="Experiment Progress",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]") as pbar:
            for run_id in range(self.config.num_runs):
                # reload loaders to avoid exhausted iterators
                test_loader = helpers.create_dataloader(
                    self.config.test_folder,
                    self.config.img_transforms,
                    self.config.gt_transforms,
                )
                source_loader = helpers.create_dataloader(
                    self.config.source_folder,
                    self.config.img_transforms,
                    self.config.gt_transforms,
                )
                self.step_experiment(test_loader, source_loader, run_id, pbar)



