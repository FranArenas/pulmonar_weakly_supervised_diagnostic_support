import abc
from pathlib import Path

import numpy as np

from preprocessing.entity.color_mode import ColorMode


class ImageReader(abc.ABC):
    @abc.abstractmethod
    def read(self, filepath: str | Path, mode: ColorMode = ColorMode.COLOR) -> np.ndarray:
        ...
