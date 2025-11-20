import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import convolve, label, uniform_filter

from scripts.detectors.target.cem import CEM
from scripts.detectors.target.ace import ACE  # Assuming you have an ACE class
from scripts.target_spectrum_generators.__base__ import MultiSourceTargetGenerator, SingleSourceTargetGenerator
from scripts.target_spectrum_generators.single_target_spectrum_generators import SingleSourceMeanGenerator
from scipy.spatial.distance import cosine

from scripts.utils.metrics import evaluation_metrics


def local_variance(cem_map: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Approximate entropy using local variance of the normalized CEM map.
    """
    cem_norm = (cem_map - cem_map.min()) / (np.ptp(cem_map) + 1e-8)
    mean = uniform_filter(cem_norm, size=window_size)
    mean_sq = uniform_filter(cem_norm ** 2, size=window_size)
    return mean_sq - mean ** 2  # Variance


def compute_similarity(source: np.ndarray, test: np.ndarray) -> float:
    """
    Compute similarity between source and test image using negative cosine distance.
    Both inputs are [C, H, W].
    """
    source_mean = source.reshape(source.shape[0], -1).mean(axis=1)
    test_mean = test.reshape(test.shape[0], -1).mean(axis=1)
    return 1 - cosine(source_mean, test_mean)  # Higher is more similar

def _softmax_with_temperature(x: np.ndarray, tau: float = 0.25, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float64)
    x = (x - np.max(x)) / max(tau, 1e-6)
    e = np.exp(x)
    w = e / (e.sum() + 1e-12)
    # small floor; renormalize
    w = w + eps
    return w / w.sum()

class MACEMMOD(MultiSourceTargetGenerator):
    def __init__(
        self,
        search_space_ratio: float = 0.02,
        detector=CEM(),
        var_weight: float = 1 / 2,
        max_regions: int = 3,
        output_dir: str = "mira2_outputs",
        use_spatial_filtering: bool = True,
        connectivity: int = 8,
        merge_method: str = "mean",  # "mean", "median", "similarity", or "auc_weighted"
        auc_fpr_range: tuple[float, float] = (0.0, 1.0),  # Pf range for AUC(Pf,Pd)
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
        self.merge_method = merge_method
        self.auc_fpr_range = auc_fpr_range

    # --- NEW: score a refined spectrum on its own source image/GT via AUC(Pf,Pd) ---
    def _score_refined_on_source(
        self,
        source_image: np.ndarray,   # [C,H,W]
        source_gt: np.ndarray,      # [H,W] binary {0,1}
        refined_spec: np.ndarray,   # [C,1] or [C]
    ) -> float:
        """
        Run detector on the source image using the refined spectrum that originated
        from this source, then compute AUC(Pf,Pd) over the chosen Pf range.
        """
        C, H, W = source_image.shape
        # shape to what detector.forward expects: [1,C,H,W] and [1,C,1,1]
        src_batch = source_image[None, ...]  # [1,C,H,W]
        if refined_spec.ndim == 1:
            refined_for_det = refined_spec.reshape(1, C, 1, 1)  # [1,C,1,1]
        elif refined_spec.ndim == 2:
            # [C,1] -> [1,C,1,1]
            refined_for_det = refined_spec.reshape(1, C, 1, 1)
        else:
            # already [1,C,1,1] or compatible
            refined_for_det = refined_spec

        det_map = self.detector.forward(src_batch, refined_for_det).squeeze()
        # Normalize to [0,1] for stable metrics
        det_map = (det_map - det_map.min()) / (np.ptp(det_map) + 1e-8)

        gt_flat = (source_gt > 0).astype(np.uint8).ravel()
        det_flat = det_map.astype(np.float32).ravel()

        auc_effect, _, _, _, _ = evaluation_metrics(
            gt_flat=gt_flat,
            detection_map_flat=det_flat,
            fpr_range=self.auc_fpr_range
        )
        # guard against zeros/NaNs
        if not np.isfinite(auc_effect) or auc_effect < 0:
            auc_effect = 0.0
        return float(auc_effect)

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

    def _similarity_weights(self, source_images, test_image):
        B, C, H, W = source_images.shape
        test_mean = test_image.reshape(C, -1).mean(axis=1)

        sims = np.zeros(B)
        for b in range(B):
            src_mean = source_images[b].reshape(C, -1).mean(axis=1)
            d = np.linalg.norm(test_mean - src_mean)  # Euclidean distance
            sims[b] = 1.0 / (d + 1e-8)  # closer → larger weight

        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        return sims / (sims.sum() + 1e-12)\

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

            # --- your existing selection / refinement flow (unchanged) ---
            det_map = self.detector.forward(test_images, source_target).squeeze()
            lv_map = local_variance(det_map)

            det_norm = (det_map - det_map.min()) / (np.ptp(det_map) + 1e-8)
            lv_norm = (lv_map - lv_map.min()) / (np.ptp(lv_map) + 1e-8)

            fused = (1 - self.var_weight) * det_norm + self.var_weight * lv_norm

            self._vis_viridis(det_norm, name=f"b{b}_det")
            self._vis_viridis(lv_norm, name=f"b{b}_lv")
            self._vis_viridis(fused, name=f"b{b}_fused")

            top_k = max(1, int(H * W * self.search_space_ratio))
            flat_scores = fused.ravel()
            top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]
            ys_k, xs_k = np.unravel_index(top_indices, (H, W))
            topk_mask = np.zeros((H, W), dtype=bool)
            topk_mask[ys_k, xs_k] = True
            self._vis_selected_pixels(H, W, topk_mask, test_gt_mask, name=f"b{b}_topk")

            if self.use_spatial_filtering:
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
            refined = np.sum(selected_spectra * weights[None, :], axis=1, keepdims=True)  # [C,1]

            all_refined_spectra.append(refined[:, None])  # [C,1,1] -> we keep as [C,1,1] style
            all_indices.append(final_indices)

        final_stack = np.concatenate(all_refined_spectra, axis=1)  # [C, B, 1]

        # --- merge across sources ---
        if self.merge_method == "mean":
            final_refined = final_stack.mean(axis=1, keepdims=False)  # [C, 1]

        elif self.merge_method == "median":
            final_refined = np.median(final_stack, axis=1, keepdims=False)  # [C, 1]

        elif self.merge_method == "similarity":
            sim_w = self._similarity_weights(source_images, test_data)  # [B]
            print(f"[MACEMMOD] Similarity weights: {sim_w}")
            final_refined = (final_stack * sim_w[None, :, None]).sum(axis=1, keepdims=False)  # [C,1]

        # --- NEW: AUC(Pf,Pd)-weighted merge (evaluated on each source) ---
        elif self.merge_method == "auc_weighted":
            weights = np.zeros(B, dtype=np.float64)
            for b in range(B):
                # refined spectrum for source b: [C,1] (squeeze trailing dim)
                refined_b = final_stack[:, b, 0]  # [C]
                score_b = self._score_refined_on_source(
                    source_image=source_images[b],
                    source_gt=source_gts[b],
                    refined_spec=refined_b
                )
                weights[b] = max(score_b, 0.0)

            # Normalize weights (fallback to uniform if all zeros)
            w_sum = weights.sum()
            if w_sum <= 0:
                weights[:] = 1.0 / B
            else:
                weights /= w_sum

            print(f"[MACEMMOD] AUC(Pf,Pd) weights (Pf range {self.auc_fpr_range}): {weights}")
            final_refined = (final_stack * weights[None, :, None]).sum(axis=1, keepdims=False)  # [C,1]


        elif self.merge_method == "best_auc":
            scores = np.zeros(B, dtype=np.float64)
            for b in range(B):
                refined_b = final_stack[:, b, 0]
                scores[b] = self._score_refined_on_source(
                    source_image=source_images[b],
                    source_gt=source_gts[b],
                    refined_spec=refined_b
                )
            b_star = int(np.argmax(scores))
            print(f"[MACEMMOD] best source by AUC: {b_star} (score={scores[b_star]:.5f})")
            final_refined = final_stack[:, b_star, :]  # [C,1]
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")



        all_indices_unique = np.unique(np.concatenate(all_indices))
        return final_refined.reshape(1, C, 1, 1), all_indices_unique