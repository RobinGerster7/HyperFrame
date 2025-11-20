from scripts.detectors.target.cem import CEM
from scripts.target_spectrum_generators.macem import MACEM
from scripts.experiments.configs.demo_config import DemoConfig
from scripts.experiments.rotating_leave_one_out_experiment import RotatingLeaveOneOutExperiment
from scripts.target_spectrum_generators.tasr import TASR
from scripts.utils.transforms import Resize, MinMaxNormalize
from scripts.visualizers.spectrum_visualizer import SpectrumVisualizer

# RGB indices (as fractions)
# Camo:        [79/223, 94/223, 112/223]
# SanDiego:    [47/188, 94/188, 141/188]
# SSD:         [70/163, 50/163, 30/163],
# ABU-Airport: [38/95, 48/95, 67/95]

if __name__ == "__main__":
    image_size = (410,410)

    config = DemoConfig(
        source_folder="datasets/",
        test_folder="datasets/SSDE",
        pre_processing_transforms=[
            Resize(image_size, "bilinear"),
            MinMaxNormalize()
        ],
        gt_transforms=[Resize(image_size, "nearest")],
        post_processing_transforms=[MinMaxNormalize()],
        detector=CEM(),
        target_spectrum_generator=MACEM(),
        num_runs=1,
        fpr_range=(0, 1),
        rgb_indices=[70/163, 50/163, 30/163],
    )

    config.display()

    experiment = RotatingLeaveOneOutExperiment(config)
    experiment.run()

    # Print metrics
    experiment.logger.display_tabular_results()

    # Visualize spectra
    spectrum_vis = SpectrumVisualizer(experiment.logger, verbose=True)
    spectrum_vis.visualize()
    spectrum_vis.print_summary()
