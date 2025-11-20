import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.experiments.__base__ import DemoExperiment
from scripts.target_spectrum_generators.__base__ import (
    MultiSourceTargetGenerator,
    SingleSourceTargetGenerator,
)
from scripts.target_spectrum_generators.background_spectrum_generators import SingleBackgroundMeanSpectrumGenerator
from scripts.target_spectrum_generators.multi_target_spectrum_generators import MultiSourceMeanGenerator
from scripts.target_spectrum_generators.single_target_spectrum_generators import SingleSourceMeanGenerator
from scripts.utils.metrics import evaluation_metrics


class RotatingLeaveOneOutExperiment(DemoExperiment):
    """
    Performs a rotating leave-one-out evaluation over a set of hyperspectral images.

    For each run:
      - Each image is treated as the test image once.
      - Remaining images act as source images.

    Depending on the generator type:
      - MultiSourceTargetGenerator: all sources used to generate one target spectrum.
      - SingleSourceTargetGenerator: one target per source-test pair.
    """

    def step_experiment(
        self,
        test_loader: DataLoader,
        source_loader: DataLoader,
        run_id: int,
        pbar: tqdm,
    ) -> None:
        generator = self.config.target_spectrum_generator
        is_multi_source = isinstance(generator, MultiSourceTargetGenerator)
        is_single_source = isinstance(generator, SingleSourceTargetGenerator)

        if not (is_multi_source or is_single_source):
            raise TypeError("Target generator must be SingleSourceTargetGenerator or MultiSourceTargetGenerator.")

        # Load all images and GT masks into memory
        all_images = np.concatenate([img.cpu().numpy() for img, _ in test_loader], axis=0)
        all_gts = np.concatenate([gt.cpu().numpy() for _, gt in test_loader], axis=0)

        for test_index in range(len(all_images)):
            test_image = all_images[test_index:test_index + 1]
            test_gt = all_gts[test_index:test_index + 1]

            if is_multi_source:
                source_images = np.concatenate((all_images[:test_index], all_images[test_index + 1:]), axis=0)
                source_gts = np.concatenate((all_gts[:test_index], all_gts[test_index + 1:]), axis=0)

                source_spectrum, _ = MultiSourceMeanGenerator().forward(source_images, source_gts)
                test_spectrum, _ = SingleSourceMeanGenerator().forward(test_image, test_gt)
                background_spectrum, _ = SingleBackgroundMeanSpectrumGenerator().forward(test_image, test_gt)

                start_time = time.time()
                target_spectrum, pixel_indices = generator.forward(source_images, source_gts, test_image)
                detection_map = self.get_detection_map(test_image, target_spectrum)
                elapsed_time = round(time.time() - start_time, 4)

                metrics = evaluation_metrics(test_gt.reshape(-1), detection_map.reshape(-1), fpr_range=self.config.fpr_range)
                metric_dict = {
                    metric: round(val, 5)
                    for metric, val in zip(self.metrics[:-1], metrics)
                }
                metric_dict["Inference Time (s)"] = elapsed_time

                self.logger.record(
                    run_id=run_id,
                    test_index=test_index,
                    source_index=None,
                    data={
                        **metric_dict,
                        "pixel_indices": pixel_indices.tolist() if pixel_indices is not None else None,
                        "spectra": {
                            "source": source_spectrum.squeeze().flatten().tolist(),
                            "test": test_spectrum.squeeze().flatten().tolist(),
                            "optimized": target_spectrum.squeeze().flatten().tolist(),
                            "background": background_spectrum.flatten().tolist()
                        },
                        "detection_map": detection_map,
                        "ground_truth_map": test_gt,
                    },
                )
                pbar.update(1)

            elif is_single_source:
                detection_maps = []
                per_pair_metrics = []
                total_time = 0.0
                pixel_indices = None

                for source_index in range(len(all_images)):
                    if source_index == test_index:
                        continue

                    source_img = all_images[source_index:source_index + 1]
                    source_gt = all_gts[source_index:source_index + 1]

                    source_spectrum, _ = SingleSourceMeanGenerator().forward(source_img, source_gt)
                    test_spectrum, _ = SingleSourceMeanGenerator().forward(test_image, test_gt)
                    background_spectrum, _ = SingleBackgroundMeanSpectrumGenerator().forward(test_image, test_gt)

                    start_time = time.time()
                    target_spectrum, _indices = generator.forward(source_img, source_gt, test_image)
                    detection_map = self.get_detection_map(test_image, target_spectrum)
                    elapsed = time.time() - start_time
                    total_time += elapsed

                    metrics = evaluation_metrics(test_gt.reshape(-1), detection_map.reshape(-1), fpr_range=self.config.fpr_range)

                    per_pair_metrics.append(metrics)
                    detection_maps.append(detection_map)

                    if pixel_indices is None:
                        pixel_indices = _indices

                    metric_dict = {
                        metric: round(val, 5)
                        for metric, val in zip(self.metrics[:-1], metrics)
                    }
                    metric_dict["Inference Time (s)"] = round(elapsed, 5)
                    metric_dict["Inference Time (s)"] = round(elapsed, 5)

                    self.logger.record(
                        run_id=run_id,
                        test_index=test_index,
                        source_index=source_index,
                        data={
                            **metric_dict,
                            "pixel_indices": _indices.tolist() if _indices is not None else None,
                            "spectra": {
                                "source": source_spectrum.squeeze().flatten().tolist(),
                                "test": test_spectrum.squeeze().flatten().tolist(),
                                "optimized": target_spectrum.squeeze().flatten().tolist(),
                                "background": background_spectrum.flatten().tolist()
                            },
                            "detection_map": detection_map,
                            "ground_truth_map": test_gt,
                        },
                    )
                    pbar.update(1)


