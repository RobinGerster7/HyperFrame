from scripts.loggers.experiment_logger import ExperimentLogger
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import roc_curve

# Set plot style to match seaborn barplot
sns.set(style="whitegrid", context="paper", font_scale=1.6)  # Increased from 1.3
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 15,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False
})

class ROCCurveVisualizer:
    def __init__(
        self,
        loggers: dict[str, ExperimentLogger],
        verbose: bool = False,
        save_dir: str = "results_hcem/roc_curves"
    ):
        self.loggers = loggers
        self.verbose = verbose
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._auc_records = []

    def run_all(self, average_across_tests: bool = False) -> None:
        if average_across_tests:
            self._plot_combined_roc_all_tests()
        else:
            all_tests = self._get_all_test_indices()
            for test_index in all_tests:
                self._plot_combined_roc_for_test(test_index)

    def _get_all_test_indices(self) -> list[int]:
        test_indices = set()
        for logger in self.loggers.values():
            for run in logger.get().values():
                test_indices.update(run.keys())
        return sorted(test_indices)

    def _plot_combined_roc_for_test(self, test_index: int) -> None:
        fpr_interp = np.logspace(-4, 0, 300)
        plt.figure(figsize=(10, 6), facecolor="white")

        methods = list(self.loggers.items())
        palette = sns.color_palette("Set2", n_colors=len(methods) - 1)

        for i, (method, logger) in enumerate(methods):
            is_last = i == len(methods) - 1
            color = "green" if is_last else palette[i % len(palette)]

            all_tpr = []
            for run_id, run_data in logger.get().items():
                if test_index not in run_data:
                    continue
                for source_index, result in run_data[test_index].items():
                    y_score = np.ravel(result["detection_map"])
                    y_true = np.ravel(result["ground_truth_map"])
                    if np.all(y_true == 0) or np.all(y_true == 1):
                        continue
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    tpr_interp = np.interp(fpr_interp, fpr, tpr, left=0.0, right=1.0)
                    all_tpr.append(tpr_interp)

            if all_tpr:
                mean_tpr = np.mean(all_tpr, axis=0)
                auc_score = np.trapz(mean_tpr, fpr_interp)
                self._auc_records.append({"method": method, "test": test_index, "auc": auc_score})
                plt.plot(
                    fpr_interp, mean_tpr,
                    label=f"{method}",
                    linewidth=2.5,
                    color=color
                )

        plt.xscale("log")
        plt.xlim([1e-4, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate (log scale)")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves for Test Image {test_index}", pad=12, fontsize=14, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Method", loc="lower right", frameon=False)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"roc_test_{test_index}.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        if self.verbose:
            plt.show()
        plt.close()

    def _plot_combined_roc_all_tests(self) -> None:
        fpr_interp = np.logspace(-4, 0, 300)
        plt.figure(figsize=(10, 6), facecolor="white")

        methods = list(self.loggers.items())
        palette = sns.color_palette("Set2", n_colors=len(methods) - 1)

        for i, (method, logger) in enumerate(methods):
            is_last = i == len(methods) - 1
            color = "green" if is_last else palette[i % len(palette)]

            all_tpr = []
            for run_data in logger.get().values():
                for test_data in run_data.values():
                    for result in test_data.values():
                        y_score = np.ravel(result["detection_map"])
                        y_true = np.ravel(result["ground_truth_map"])
                        if np.all(y_true == 0) or np.all(y_true == 1):
                            continue
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        tpr_interp = np.interp(fpr_interp, fpr, tpr, left=0.0, right=1.0)
                        all_tpr.append(tpr_interp)

            if all_tpr:
                mean_tpr = np.mean(all_tpr, axis=0)
                auc_score = np.trapz(mean_tpr, fpr_interp)
                self._auc_records.append({"method": method, "test": "ALL", "auc": auc_score})
                plt.plot(
                    fpr_interp, mean_tpr,
                    label=f"{method}",
                    linewidth=2.5,
                    color=color
                )

        plt.xscale("log")
        plt.xlim([1e-4, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate (log scale)")
        plt.ylabel("True Positive Rate")
        #plt.title("Combined ROC Curves Across All Test Images", pad=12, fontsize=14, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Method", loc="lower right", frameon=False)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"roc_all_tests.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        if self.verbose:
            plt.show()
        plt.close()

    def print_summary(self) -> None:
        if not self._auc_records:
            print("No AUC records available.")
            return

        df = pd.DataFrame(self._auc_records)
        summary = df.pivot_table(index="method", columns="test", values="auc").round(4)
        overall = df.groupby("method")["auc"].mean().round(4)

        print("\n=== AUC Summary (per Test) ===")
        print(summary)
        print("\n=== Overall Mean AUC ===")
        print(overall)
