import abc
from pathlib import Path

import numpy as np


class ImageReader(abc.ABC):
    @abc.abstractmethod
    def read(self, filepath: str | Path) -> np.ndarray:
        ...
