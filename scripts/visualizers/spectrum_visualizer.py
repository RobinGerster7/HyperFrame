from .__base__ import BaseVisualizer
from scripts.loggers.experiment_logger import ExperimentLogger

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


class SpectrumVisualizer(BaseVisualizer):
    def __init__(
        self,
        logger: ExperimentLogger,
        verbose: bool = False
    ):
        super().__init__(logger, verbose)
        self._all_metrics = []

        self.colors = {
            "source": "#E41A1C",
            "optimized": "#4DAF4A",
            "test": "#377EB8",
            "background": "#984EA3"
        }

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

    def _rel_improvement(self, opt: float, src: float, better: str = "lower") -> float:
        if src == 0:
            return 0.0
        return (src - opt) / abs(src) if better == "lower" else (opt - src) / abs(src)

    def _resolve_output_path(self, run: int, test: int, source: Optional[int]) -> str:
        base = os.path.join("results_hcem", f"run_{run}", f"test_{test}")
        return os.path.join(base, f"source_{source}" if source is not None else "source_None")

    def _visualize_entry(
        self,
        run: int,
        test: int,
        source: Optional[int],
        result: dict,
        **kwargs
    ) -> None:
        spectra = result.get("spectra", {})
        if not {"source", "test", "optimized"}.issubset(spectra):
            return

        source_spec = np.squeeze(np.array(spectra["source"]))
        test_spec = np.squeeze(np.array(spectra["test"]))
        opt_spec = np.squeeze(np.array(spectra["optimized"]))

        bands = np.arange(len(test_spec))
        width = 0.4
        source_label = "Multi" if source is None else str(source)

        # Metrics
        mae_src = np.mean(np.abs(test_spec - source_spec))
        mae_opt = np.mean(np.abs(test_spec - opt_spec))
        mse_src = mean_squared_error(test_spec, source_spec)
        mse_opt = mean_squared_error(test_spec, opt_spec)
        cos_src = self._cosine_sim(source_spec, test_spec)
        cos_opt = self._cosine_sim(opt_spec, test_spec)

        self._all_metrics.append({
            "run": run,
            "test": test,
            "source": source_label,
            "mae_source": mae_src,
            "mae_optimized": mae_opt,
            "mse_source": mse_src,
            "mse_optimized": mse_opt,
            "cosine_source": cos_src,
            "cosine_optimized": cos_opt,
        })

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

        # Spectral curves
        axes[0].plot(bands, source_spec, "--", label="Source", color=self.colors["source"], linewidth=2)
        axes[0].plot(bands, test_spec, "-.", label="Test", color=self.colors["test"], linewidth=2.5)
        axes[0].plot(bands, opt_spec, "-", label="Optimized", color=self.colors["optimized"], linewidth=3)

        axes[0].set_ylabel("Reflectance",fontsize=14)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Absolute error
        axes[1].bar(bands - width / 2, np.abs(test_spec - source_spec), width=width,
                    label="|Source - Test|", color=self.colors["source"], edgecolor="black", alpha=0.8)
        axes[1].bar(bands + width / 2, np.abs(test_spec - opt_spec), width=width,
                    label="|Optimized - Test|", color=self.colors["optimized"], edgecolor="black", alpha=0.8)

        axes[1].set_xlabel("Spectral Band",fontsize=14)
        axes[1].set_ylabel("Abs Error",fontsize=14)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_dir = self._resolve_output_path(run, test, source)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"spectra_plot.png")
        plt.savefig(save_path, dpi=300)

        if self.verbose or kwargs.get("show", False):
            plt.show()
        else:
            plt.close()

    def print_summary(self):
        if not self._all_metrics:
            return

        df = pd.DataFrame(self._all_metrics)

        print("\n=== Overall Mean Relative Improvements ===")
        for metric, better in [("mae", "lower"), ("mse", "lower"), ("cosine", "higher")]:
            src_mean = df[f"{metric}_source"].mean()
            opt_mean = df[f"{metric}_optimized"].mean()
            rel = self._rel_improvement(opt_mean, src_mean, better)
            arrow = "↑" if rel > 0 else "↓" if rel < 0 else "→"
            print(f"{metric.upper()}: Source = {src_mean:.3f}, Opt = {opt_mean:.3f}, Δ = {rel:+.1%} {arrow}")
