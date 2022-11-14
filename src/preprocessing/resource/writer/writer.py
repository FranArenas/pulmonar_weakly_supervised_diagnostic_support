import abc
from pathlib import Path

import numpy as np


class ImageWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, image: np.ndarray, filepath: Path | str):
        ...
