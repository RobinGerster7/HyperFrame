from scripts.detectors.__base__ import Detector
from scripts.experiments.configs.__base__ import BaseExperimentConfig
from rich.console import Console
from rich.table import Table
from typing import Callable
import numpy as np

from scripts.target_spectrum_generators.__base__ import SingleSourceTargetGenerator, TargetSpectrumGenerator


class DemoConfig(BaseExperimentConfig):
    """
    Configuration class for the DemoExperiment.

    Args:
        source_folders (str): Path to source dataset folder.
        test_folders (str): Path to test dataset folder.
        pre_processing_transforms (list[Callable[[np.ndarray], np.ndarray]]): List of image transforms.
        gt_transforms (list[Callable[[np.ndarray], np.ndarray]]): List of ground truth transforms.
        post_processing_transforms (list[Callable[[np.ndarray], np.ndarray]]): List of optional post-processing transforms.
        detector (Detector): Detector model instance.
        target_spectrum_generator (TargetSpectrumGenerator): Target spectrum generator instance.
        num_runs (int): Number of experiment runs.
        rgb_indices (list[float], optional): List of 3 relative indices (in [0, 1]) for RGB visualization channels.
        fpr_range (tuple[float, float], optional): False positive rate range for partial AUC evaluation.
    """

    def __init__(self,
                 source_folder: str,
                 test_folder: str,
                 pre_processing_transforms: list[Callable[[np.ndarray], np.ndarray]],
                 gt_transforms: list[Callable[[np.ndarray], np.ndarray]],
                 post_processing_transforms: list[Callable[[np.ndarray], np.ndarray]],
                 detector: Detector,
                 target_spectrum_generator: TargetSpectrumGenerator,
                 num_runs: int,
                 rgb_indices: list[float] = None,
                 fpr_range: tuple[float, float] = (0, 1)) -> None:
        self.img_transforms = pre_processing_transforms
        self.gt_transforms = gt_transforms
        self.post_processing_transforms = post_processing_transforms
        self.detector = detector
        self.target_spectrum_generator = target_spectrum_generator
        self.num_runs = num_runs
        self.source_folder = source_folder
        self.test_folder = test_folder
        self.rgb_indices = rgb_indices if rgb_indices is not None else [0.25, 0.5, 0.75]
        self.fpr_range = fpr_range

    def display(self) -> None:
        """
        Displays the experiment configuration in a formatted table.
        """
        console = Console()
        table = Table(title="\U0001F3C1 Experiment Configuration", show_header=False)
        table.add_row("\U0001F50D Detector", self.detector.__class__.__name__)
        table.add_row("\U0001F3AF Target Generator", self.target_spectrum_generator.__class__.__name__)
        table.add_row("\U0001F4C2 Source Folder", self.source_folder)
        table.add_row("\U0001F4C2 Test Folder", self.test_folder)
        table.add_row("\U0001F501 Runs", str(self.num_runs))
        table.add_row("\U0001F3A8 RGB Indices (relative)", str(self.rgb_indices))
        table.add_row("\U0001F4CA FPR Range for AUC", f"{self.fpr_range[0]:.0e} to {self.fpr_range[1]:.0e}")

        console.print(table)

