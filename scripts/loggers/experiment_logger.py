# loggers/experiment_logger.py
from typing import Any, Literal, Optional
import numpy as np
import pickle
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from pathlib import Path
from scripts.experiments.configs.__base__ import BaseExperimentConfig


class ExperimentLogger:
    def __init__(self, config: BaseExperimentConfig) -> None:
        self.config = config
        self.results: dict[int, dict[int, dict[int | None, dict[str, Any]]]] = {}

    def record(self, run_id: int, test_index: int, source_index: int | None, data: dict[str, Any]) -> None:
        self.results.setdefault(run_id, {}).setdefault(test_index, {})[source_index] = data

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"results_hcem": self.results, "config": self.config}, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.results = data["results_hcem"]
            self.config = data.get("config")

    def get(self) -> dict:
        return self.results

    def print(self) -> None:
        Console().print(Pretty(self.results, expand_all=True))

    def display_tabular_results(self) -> None:
        flat: dict[str, list[float]] = {}
        for run in self.results.values():
            for test in run.values():
                for result in test.values():
                    for k, v in result.items():
                        if isinstance(v, (float, int)):
                            flat.setdefault(k, []).append(v)

        table = Table(title="🏆 Overall Final Metrics", show_header=True)
        table.add_column("Metric")
        table.add_column("Mean", justify="center")
        table.add_column("Std Dev", justify="center")
        table.add_column("Min", justify="center")
        table.add_column("Max", justify="center")

        for metric, values in flat.items():
            arr = np.array(values)
            table.add_row(metric, f"{arr.mean():.3f}", f"{arr.std():.3f}", f"{arr.min():.3f}", f"{arr.max():.3f}")
        Console().print(table)

    def get_metric_stats(
            self,
            metric_name: str,
            stat: Literal["mean", "std", "min", "max", "all"] = "all",
            run_filter: Optional[int] = None
    ) -> float | dict[str, float] | None:
        """
        Returns requested statistic(s) for a given metric, optionally filtered by run ID.
        """
        values = []

        for run_id, run in self.results.items():
            if run_filter is not None and run_id != run_filter:
                continue
            for test in run.values():
                for result in test.values():
                    val = result.get(metric_name)
                    if isinstance(val, (float, int)):
                        values.append(val)

        if not values:
            return None

        arr = np.array(values)

        if stat == "mean":
            return float(arr.mean())
        elif stat == "std":
            return float(arr.std())
        elif stat == "min":
            return float(arr.min())
        elif stat == "max":
            return float(arr.max())
        elif stat == "all":
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        else:
            raise ValueError(f"Unknown stat '{stat}'. Choose from mean, std, min, max, all.")

