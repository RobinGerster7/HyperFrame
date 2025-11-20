import numpy as np
from pygad import GA
from scripts.detectors.__base__ import Detector
from scripts.detectors.target.cem import CEM
from scripts.utils.metrics import evaluation_metrics
from scripts.utils.transforms import Resize
from scripts.target_spectrum_generators.__base__ import SingleSourceTargetGenerator
import matplotlib
matplotlib.use("Agg")  # prevents Tkinter use


class TASR(SingleSourceTargetGenerator):
    """
      TASR: Test-time Adaptive Spectrum Refinement using Genetic Algorithms.

      This generator refines a target spectrum by selecting an optimal subset of test image pixels using
      a genetic algorithm (GA). The optimization maximizes detection performance (AUC) while also promoting
      spectral separability from the background. The GA evolves sets of pixel indices whose spectra are averaged
      to form a target signature. Optional resizing is used to make optimization tractable.

      The `record_per_source_test_pair` flag (inherited from SingleSourceTargetGenerator) controls whether metrics
      are logged per source-test pair or aggregated during evaluation.

      Args:
          genome_length (int): Number of pixels (genes) in each GA solution.
          population_size (int): Number of candidate solutions per GA generation.
          generations (int): Number of generations to run the GA.
          mutation_rate (int): Percentage of genes to mutate in each child.
          tournament_size (int): Number of candidates in each tournament selection.
          keep_parents (int): Number of elite parents to carry over to the next generation.
          separability_weight (float): Weight given to spectral separability in the fitness function.
          size (tuple): Image size (H, W) used during optimization (resized from full resolution).
          detector (Detector): Detector used to compute detection map (e.g., CEM).
          mean_aggregation (bool): If False, instructs evaluation code (e.g., in leave-one-out setups)
            to record detection results_hcem and metrics separately for each source-test pair.
            If True, results_hcem are aggregated across all sources per test image.
            This flag does not affect the behavior of the forward() method.

      Returns:
          tuple:
              refined_spectrum (np.ndarray): Refined target spectrum of shape (1, C, 1, 1).
              selected_pixel_indices (np.ndarray): Flattened pixel indices of selected pixels in the original resolution.
      """

    def __init__(
        self,
        genome_length: int = 10,
        population_size: int = 30,
        generations: int = 50,
        mutation_rate: int = 25,
        tournament_size: int = 5,
        keep_parents: int = 1,
        separability_weight: float = 0.1,
        size: tuple = (100, 100),
        detector: Detector = CEM(),
    ):

        super().__init__()
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.keep_parents = keep_parents
        self.separability_weight = separability_weight
        self.detector = detector
        self.size = size

    def seperability_score(self, target_spectrum: np.ndarray, background_spectrum: np.ndarray) -> float:
        """
        Computes spectral separability between a target and background spectrum using the spectral angle mapper (SAM).

        Args:
            target_spectrum (np.ndarray): Target spectrum of shape (C,).
            background_spectrum (np.ndarray): Background spectrum of shape (C,).

        Returns:
            float: Angular difference (in radians) between the two spectra.
        """
        cos_theta = np.dot(target_spectrum, background_spectrum) / (
            np.linalg.norm(target_spectrum) * np.linalg.norm(background_spectrum)
        )
        return np.arccos(np.clip(cos_theta, -1, 1))

    def fitness_function(self, ga_instance, solution: list, solution_idx: int) -> float:
        """
        Fitness function used by the genetic algorithm.

        Evaluates a candidate solution (set of pixel indices) based on a combination of
        detection AUC and spectral separability from the background.

        Args:
            ga_instance: Unused reference to the GA instance (required by PyGAD).
            solution (list): List of flattened pixel indices (length = genome_length).
            solution_idx (int): Index of the solution in the current population.

        Returns:
            float: Fitness score combining AUC and separability.
        """
        pixel_indices = np.array(solution, dtype=np.int64)
        rows, cols = pixel_indices // self.image_width, pixel_indices % self.image_width
        spectra = self.test_image[:, rows, cols]

        optimized_spectrum = np.mean(spectra, axis=1, keepdims=True).reshape(1, -1, 1, 1)

        detection_map = self.detector.forward(self.source_image.reshape(1, *self.source_image.shape),
                                              optimized_spectrum)
        detection_map_flat = detection_map.squeeze().flatten()
        ground_truth_flat = self.ground_truth.flatten()

        auc_effect, *_ = evaluation_metrics(ground_truth_flat, detection_map_flat)
        background_pixels = self.test_image.reshape(self.test_image.shape[0], -1)
        separability = self.seperability_score(optimized_spectrum.squeeze(), np.mean(background_pixels, axis=1))

        return (auc_effect + self.separability_weight * separability).item()

    def forward(
        self,
        source_image: np.ndarray,
        ground_truth: np.ndarray,
        test_image: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the genetic algorithm to optimize a refined target spectrum from the test image.

        Args:
            source_image (np.ndarray): Source image of shape (1, C, H, W).
            ground_truth (np.ndarray): Ground truth mask of shape (1, 1, H, W).
            test_image (np.ndarray): Test image of shape (1, C, H, W).

        Returns:
            tuple:
                refined_spectrum (np.ndarray): Final refined spectrum of shape (1, C, 1, 1).
                selected_pixel_indices (np.ndarray): Flattened pixel indices in original resolution, shape (N,).
        """
        self.test_image_fullres = test_image.squeeze(0)  # (C, H_orig, W_orig)

        self.source_image = Resize(self.size, "bilinear")(source_image.squeeze(0))
        self.test_image = Resize(self.size, "bilinear")(self.test_image_fullres)
        self.ground_truth = Resize(self.size, "nearest")(ground_truth.squeeze(0))

        self.image_height, self.image_width = self.source_image.shape[1:]
        self.num_total_pixels = self.image_height * self.image_width

        ga_instance = GA(
            num_generations=self.generations,
            sol_per_pop=self.population_size,
            num_parents_mating=self.population_size // 2,
            fitness_func=self.fitness_function,
            num_genes=self.genome_length,
            gene_space=list(range(self.num_total_pixels)),
            mutation_percent_genes=self.mutation_rate,
            parent_selection_type="tournament",
            K_tournament=self.tournament_size,
            keep_parents=self.keep_parents,
            parallel_processing=["thread", 4],
            crossover_type="uniform",
            crossover_probability=1.0,
        )

        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()
        pixel_indices = np.array(solution, dtype=np.int64)
        rows, cols = pixel_indices // self.image_width, pixel_indices % self.image_width
        spectra = self.test_image[:, rows, cols]
        avg_spectrum = np.mean(spectra, axis=1, keepdims=True)

        scale_h = self.test_image_fullres.shape[1] / self.image_height
        scale_w = self.test_image_fullres.shape[2] / self.image_width
        original_rows = np.clip((rows * scale_h).astype(np.int64), 0, self.test_image_fullres.shape[1] - 1)
        original_cols = np.clip((cols * scale_w).astype(np.int64), 0, self.test_image_fullres.shape[2] - 1)
        original_pixel_indices = original_rows * self.test_image_fullres.shape[2] + original_cols

        return avg_spectrum.reshape(1, -1, 1, 1), original_pixel_indices

