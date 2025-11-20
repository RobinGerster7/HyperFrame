import os
from abc import ABC, abstractmethod
from typing import Optional

from scripts.loggers.experiment_logger import ExperimentLogger


class BaseVisualizer(ABC):
    def __init__(
        self,
        logger: ExperimentLogger,
        verbose: bool = False
    ):
        self.logger = logger
        self.verbose = verbose

    def visualize(
        self,
        run: Optional[int] = None,
        test: Optional[int] = None,
        source: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Iterates over (run, test, source) entries and dispatches to `_visualize_entry`.
        """
        results = self.logger.get()
        runs = [run] if run is not None else results.keys()

        for r in runs:
            tests = [test] if test is not None else results[r].keys()
            for t in tests:
                sources = [source] if source is not None else results[r][t].keys()
                for s in sources:
                    entry = results[r][t][s]
                    self._visualize_entry(r, t, s, entry, **kwargs)

    @abstractmethod
    def _visualize_entry(self, run: int, test: int, source: Optional[int], result: dict, **kwargs):
        pass
