import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.ndimage import convolve, label, uniform_filter
from scripts.detectors.target.cem import CEM
from scripts.detectors.target.ace import ACE  # Assuming you have an ACE class
from scripts.target_spectrum_generators.__base__ import MultiSourceTargetGenerator, SingleSourceTargetGenerator
from scripts.target_spectrum_generators.single_target_spectrum_generators import SingleSourceMeanGenerator


def local_variance(cem_map: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Approximate entropy using local variance of the normalized CEM map.
    """
    cem_norm = (cem_map - cem_map.min()) / (np.ptp(cem_map) + 1e-8)
    mean = uniform_filter(cem_norm, size=window_size)
    mean_sq = uniform_filter(cem_norm**2, size=window_size)
    return mean_sq - mean**2  # Variance



class MACEM(MultiSourceTargetGenerator):
    def __init__(
            self,
            search_space_ratio: float = 0.02,
            detector=CEM(),
            var_weight: float = 1 / 2,
            max_regions: int = 3,
            output_dir: str = "mira2_outputs",
            use_spatial_filtering: bool = True,
            connectivity: int = 8,
    ):
        super().__init__()
        self.search_space_ratio = search_space_ratio
        self.var_weight = var_weight
        self.max_regions = max_regions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_spatial_filtering = use_spatial_filtering
        self.connectivity = connectivity
        self.detector = detector

    def _vis_selected_pixels(self, H, W, selected_mask, gt_mask, name):
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        tp = selected_mask & gt_mask
        fp = selected_mask & ~gt_mask
        vis[tp] = (0, 255, 0)   # Green
        vis[fp] = (0, 0, 255)   # Red
        cv2.imwrite(str(self.output_dir / f"{name}.png"), vis)

    def _vis_kept_components(self, regions, name, H, W):
        output = np.zeros((H, W, 3), dtype=np.uint8)
        colors = np.random.randint(50, 255, size=(len(regions), 3), dtype=np.uint8)
        for i, (_, region_mask) in enumerate(regions):
            output[region_mask] = colors[i]
        cv2.imwrite(str(self.output_dir / f"{name}.png"), output)

    def _vis_viridis(self, array: np.ndarray, name: str):
        H, W = array.shape
        dpi = 100
        figsize = (W / dpi, H / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(array, cmap='viridis', aspect='equal')
        ax.axis('off')
        fig.savefig(str(self.output_dir / f"{name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def forward(
        self,
        source_images: np.ndarray,
        source_gts: np.ndarray,
        test_images: np.ndarray = None,
        test_gt: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        B, C, H, W = source_images.shape
        test_data = test_images.squeeze(0)
        test_gt_mask = (test_gt.squeeze() > 0) if test_gt is not None else np.zeros((H, W), dtype=bool)
        all_refined_spectra = []
        all_indices = []

        for b in range(B):
            source_target, _ = SingleSourceMeanGenerator().forward(
                source_images[b:b + 1], source_gts[b:b + 1]
            )

            # Compute detection maps
            det_map = self.detector.forward(test_images, source_target).squeeze()
            lv_map = local_variance(det_map)

            # Normalize each map
            det_norm = (det_map - det_map.min()) / (np.ptp(det_map) + 1e-8)
            lv_norm = (lv_map - lv_map.min()) / (np.ptp(lv_map) + 1e-8)

            # Weighted fusion
            fused = (
                (1-self.var_weight) * det_norm +
                self.var_weight * lv_norm
            )

            # Visualizations
            self._vis_viridis(det_norm, name=f"b{b}_det")
            self._vis_viridis(lv_norm, name=f"b{b}_lv")
            self._vis_viridis(fused, name=f"b{b}_fused")

            # Top-K selection
            top_k = max(1, int(H * W * self.search_space_ratio))
            flat_scores = fused.ravel()
            top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]
            ys_k, xs_k = np.unravel_index(top_indices, (H, W))
            topk_mask = np.zeros((H, W), dtype=bool)
            topk_mask[ys_k, xs_k] = True
            self._vis_selected_pixels(H, W, topk_mask, test_gt_mask, name=f"b{b}_topk")

            # Connected component filtering
            if self.use_spatial_filtering:
                # Define structure based on connectivity
                if self.connectivity == 4:
                    structure = np.array([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]], dtype=int)
                else:
                    structure = np.ones((3, 3), dtype=int)

                labeled, num = label(topk_mask, structure=structure)
                regions = []
                for i in range(1, num + 1):
                    region_mask = labeled == i
                    size = region_mask.sum()
                    regions.append((size, region_mask))

                if not regions:
                    max_idx = flat_scores.argmax()
                    ys = np.array([max_idx // W])
                    xs = np.array([max_idx % W])
                    final_mask = np.zeros((H, W), dtype=bool)
                    final_mask[ys[0], xs[0]] = True
                    kept_regions = []
                else:
                    regions = sorted(regions, key=lambda x: -x[0])
                    kept_regions = regions[:self.max_regions]
                    final_mask = np.logical_or.reduce([r[1] for r in kept_regions])
                    ys, xs = np.nonzero(final_mask)

                self._vis_selected_pixels(H, W, final_mask, test_gt_mask, name=f"b{b}_pruned")
                if kept_regions:
                    self._vis_kept_components(kept_regions, name=f"b{b}_components", H=H, W=W)

            else:
                final_mask = topk_mask
                ys, xs = np.nonzero(final_mask)
                self._vis_selected_pixels(H, W, final_mask, test_gt_mask, name=f"b{b}_pruned_nofilter")

            final_indices = ys * W + xs
            selected_spectra = test_data[:, ys, xs]
            selected_scores = fused[ys, xs]
            weights = selected_scores / (selected_scores.sum() + 1e-8)
            refined = np.sum(selected_spectra * weights[None, :], axis=1, keepdims=True)

            all_refined_spectra.append(refined[:, None])
            all_indices.append(final_indices)

        final_stack = np.concatenate(all_refined_spectra, axis=1)  # [C, B]
        final_refined = final_stack.mean(axis=1, keepdims=True)   # [C, 1]
        all_indices_unique = np.unique(np.concatenate(all_indices))

        return final_refined.reshape(1, C, 1, 1), all_indices_unique


